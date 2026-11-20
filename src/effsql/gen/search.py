"""Cost-guided search for cheaper correct queries (train only): `python -m effsql.gen.search`.

The first pool found a query at least 2x cheaper than gold for only 6% of train questions: each
rewrite was one blind attempt by the 7B model, which never saw a cost. This search gives the model
what an engineer would look at, and several tries:

  round r, for every target question without a usable query at most stop_rel_cost x gold's cost:
    prompt = schema, evidence, question, the cheapest usable correct query known so far with its
             step count and SQLite plan, and up to `history` earlier attempts with what went wrong
             (different result, error, not cheaper, matches only by chance);
    n answers are extracted, deduplicated against everything known for the question, executed
    (correct = same result as gold under the official 30 s limit; then VM steps, capped at 50x
    gold like screening) and checked like the pool: the arbitrary-row-pick check of
    pool/quality.py and constants that appear nowhere in the question, evidence or gold.
  "usable" (the strict rule, as in the RL reward; version 2, 2026-11-21) = correct, cost counted, no
  arbitrary-row-pick construct that gold does not use (a "first-row pick": `x = (SELECT ...)` with a
  subquery that can return several rows, LIMIT without ORDER BY; about half of the pilot's 2x wins
  were these) and no constants found nowhere in the question, evidence or gold. Only usable queries
  become the next reference or count as found.

The start is the base pool (pool_v1): its cheapest usable correct query per question, and its
known candidates are not executed again. Outputs in runs/<run>/ (resumable at every step):
  search_targets_train.json     the questions (a pilot is a fixed random sample)
  search_gen_train.jsonl        model answers, one record per (question, round)
  search_exec_train.jsonl       one record per distinct new query
  search_summary_train.json     headroom before and after every round, yield, throughput, and at the end
                                for the training and the validation questions separately
  cheap_targets_train.jsonl     per question: the cheapest usable correct query known (gold, pool or
                                search) and the cheapest from the base pool alone. Used by DPO (the
                                chosen answers) and RL (the cost yardstick of the reward).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.exec.engine import batched, connect_readonly, execute_many, execute_two_pass
from effsql.gen.backend import Backend, GenParams, make_backend, stable_seed
from effsql.gen.parse import cand_id, canonicalize, extract_sql
from effsql.gen.prompts import fill, fit_messages, load_templates
from effsql.gen.rewrite import plan_text
from effsql.gen.sample import Request, is_current_gen, run_requests, schemas_for
from effsql.io import append_jsonl, latest_by, read_json, write_json
from effsql.measure.correctness import hashes_match
from effsql.measure.cost import rel_cost
from effsql.measure.gold import ensure_gold_costs
from effsql.paths import Paths, load_paths
from effsql.pool.quality import analyze, foreign_literals, verdict

log = logging.getLogger(__name__)
SEARCH_VERSION = 2  # 2: the strict usable rule (first-row picks never usable)
SPLIT = "train"
CHUNK = 1000
OK_QUALITY = ("ok", "verified", "gold")


def is_current_search(rec: dict[str, Any]) -> bool:
    return rec.get("search_version") == SEARCH_VERSION


# Start state from the base pool ----------------------------------------------------------------
def strict_ok(sql: str, example: Any, gold_flags: set[str]) -> bool:
    """No arbitrary-row-pick construct that gold does not use, no constant found nowhere in the
    question, evidence or gold (the rule of the RL reward)."""
    return not (set(analyze(sql)["flags"]) - gold_flags) and not foreign_literals(sql, example.gold_sql,
                                                                                 example.question, example.evidence)


def base_state(paths: Paths, base_run: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """(best, known): best usable correct query per question with gold ok (strict rule); every known
    candidate per question by canonical SQL (its correctness and cost, so a repeat is not executed again)."""
    df = pd.read_parquet(paths.run_dir(base_run) / f"pool_{SPLIT}.parquet")
    df = df[df["gold_status"] == "ok"]
    examples = {e.qid: e for e in load_examples(SPLIT, paths)}
    gold_flags: dict[str, set[str]] = {}
    known: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    best: dict[str, dict[str, Any]] = {}
    for r in df.itertuples(index=False):
        usable = bool(r.correct) and r.quality in OK_QUALITY and not r.steps_capped and not pd.isna(r.rel_cost)
        if usable and not r.is_gold and r.rel_cost < 1:  # only a query cheaper than gold can become the best
            if r.qid not in gold_flags:
                gold_flags[r.qid] = set(analyze(examples[r.qid].gold_sql)["flags"])
            usable = strict_ok(r.sql_original, examples[r.qid], gold_flags[r.qid])
        rec = {"sql": r.sql_original, "correct": bool(r.correct), "usable": usable,
               "vm_steps": None if pd.isna(r.vm_steps) else int(r.vm_steps),
               "rel_cost": None if pd.isna(r.rel_cost) else float(r.rel_cost),
               "gold_vm_steps": int(r.gold_vm_steps), "source": r.source}
        if r.sql_canonical is not None:
            known[r.qid][r.sql_canonical] = rec
        if usable and (r.qid not in best or rec["rel_cost"] < best[r.qid]["rel_cost"]):
            best[r.qid] = rec
    return best, known


def pick_targets(paths: Paths, run: str, cfg: dict[str, Any], best: dict[str, Any], db_of: dict[str, str],
                 limit: int | None) -> list[str]:
    """All questions with a usable start, or a fixed random sample of `limit` (pilot) outside the
    validation databases. Saved on the first call; a later call with other settings is refused."""
    path = paths.run_dir(run) / f"search_targets_{SPLIT}.json"
    want = {"base_run": cfg["base_run"], "limit": limit, "seed": cfg["seed"]}
    if path.exists():
        saved = read_json(path)
        if {k: saved.get(k) for k in want} != want:
            raise RuntimeError(f"{path} was made with {[saved.get(k) for k in want]}, not {list(want.values())}: "
                               "use another --run")
        return saved["qids"]
    qids = sorted(best)
    excluded: list[str] = []
    if limit:
        exp_meta = paths.run_dir(cfg["pilot_exclude_export"], create=False) / "export_meta.json"
        excluded = sorted(read_json(exp_meta)["holdout_db_ids"]) if exp_meta.exists() else []
        pool = [q for q in qids if db_of[q] not in set(excluded)]
        rng = np.random.default_rng(stable_seed(cfg["seed"], "search_pilot"))
        qids = sorted(rng.choice(pool, size=min(limit, len(pool)), replace=False).tolist())
    write_json(path, {**want, "excluded_dbs": excluded, "n": len(qids), "qids": qids})
    return qids


# Prompt -----------------------------------------------------------------------------------------
def attempt_note(rec: dict[str, Any], best_steps: int) -> str:
    if not rec["correct"]:
        if rec.get("error_type"):
            return f"fails ({rec['error_type']}): {(rec.get('error_msg') or '')[:160]}"
        return "runs, but returns a different result than the reference query"
    if rec.get("foreign_literals"):
        return f"uses constants that are not in the question or evidence: {', '.join(rec['foreign_literals'][:3])}"
    if rec.get("vm_steps") is None:  # counting stopped at the cap
        return "correct, but far more expensive than the reference"
    if not rec["usable"]:
        return "matches the result only because SQLite picks an arbitrary row (not a safe equivalent)"
    return f"correct, but {rec['vm_steps']} steps: not cheaper than the reference ({best_steps})"


def history_text(attempts: list[dict[str, Any]], best_steps: int, k: int) -> str:
    if not attempts or k <= 0:
        return ""
    lines = ["", "Earlier attempts that did not improve on the reference (do not repeat them):"]
    for a in attempts[:k]:
        sql = " ".join(a["sql"].split())
        lines.append(f"- {sql[:600]}\n  -> {attempt_note(a, best_steps)}")
    return "\n".join(lines) + "\n"


def build_search_prompt(templates: dict[str, str], example: Any, schema: dict[str, Any], best: dict[str, Any],
                        plan: str, history: str, budget: int, count_tokens: Any) -> tuple[list[dict[str, str]], str, int]:
    def make(schema_text: str) -> list[dict[str, str]]:
        user = fill(templates["search_user"], {
            "db_id": example.db_id, "schema": schema_text, "evidence": example.evidence or "(none)",
            "question": example.question, "ref_steps": best["vm_steps"], "gold_steps": best["gold_vm_steps"],
            "ref_sql": best["sql"], "plan": plan, "history": history})
        return [{"role": "system", "content": templates["search_system"]}, {"role": "user", "content": user}]

    return fit_messages(make, schema, budget, count_tokens)


# Execution of one round's answers ---------------------------------------------------------------
def execute_round(paths: Paths, run: str, round_: int, examples: dict[str, Any], golds: dict[str, dict[str, Any]],
                  known: dict[str, dict[str, dict[str, Any]]], exec_cfg: dict[str, Any], pool_cfg: dict[str, Any],
                  workers: int | None) -> None:
    run_dir = paths.run_dir(run)
    out = run_dir / f"search_exec_{SPLIT}.jsonl"
    done = latest_by(out, key="cand_id", where=is_current_search)
    gens = [g for g in latest_by(run_dir / f"search_gen_{SPLIT}.jsonl", key="key", where=is_current_gen).values()
            if g["round"] == round_]
    c = exec_cfg["progress_n"]
    new: dict[str, dict[str, Any]] = {}
    repeats: list[dict[str, Any]] = []
    for g in sorted(gens, key=lambda g: g["key"]):
        q = g["qid"]
        for comp in g["completions"]:
            sql, how = extract_sql(comp["text"])
            if sql is None:
                continue
            canon, _ = canonicalize(sql)
            cid = cand_id(q, canon)
            if cid in done or cid in new:
                continue
            base = {"cand_id": cid, "qid": q, "db_id": examples[q].db_id, "round": round_, "sql": sql,
                    "sql_canonical": canon, "extraction": how, "model": g.get("model"),
                    "gold_vm_steps": golds[q]["vm_steps"], "search_version": SEARCH_VERSION}
            if canon in known.get(q, {}):  # already in the base pool: reuse its verdict
                k = known[q][canon]
                repeats.append({**base, "known": True, "correct": k["correct"], "usable": k["usable"],
                                "vm_steps": k["vm_steps"], "rel_cost": k["rel_cost"], "error_type": None})
                new[cid] = repeats[-1]
            else:
                new[cid] = {**base, "known": False}
    if repeats:
        append_jsonl(out, repeats)
    todo = [r for r in new.values() if not r["known"]]
    log.info("search round %d: %d new distinct queries to execute, %d repeats of known ones", round_, len(todo),
             len(repeats))
    cap_factor = pool_cfg.get("candidate_step_cap_factor", 50)
    gold_flags = {q: set(analyze(examples[q].gold_sql)["flags"]) for q in {r["qid"] for r in todo}}

    def is_correct(r: dict[str, Any]) -> bool:
        return bool(r["ok"] and hashes_match(r["result_hash"], golds[new[r["key"]]["qid"]]["result_hash"]))

    for chunk in batched(todo, CHUNK):
        jobs = [{"key": r["cand_id"], "db_path": paths.local_db_path(SPLIT, r["db_id"]), "sql": r["sql"],
                 "count_exec": {"step_budget": int(cap_factor * max(r["gold_vm_steps"] or 0, 1000))}} for r in chunk]
        res = execute_two_pass(jobs, paths, exec_cfg, count_if=is_correct, workers=workers)
        # Quality: the deterministic variant of flagged correct queries, executed like quality.py does.
        analyses = {r["cand_id"]: analyze(r["sql"]) for r in chunk if is_correct(res[r["cand_id"]])}
        qjobs = []
        for cid, a in analyses.items():
            if a["flags"] and a["roundtrip"] is not None and not a["untransformable"]:
                db = paths.local_db_path(SPLIT, new[cid]["db_id"])
                qjobs += [{"key": f"{cid}|rt", "db_path": db, "sql": a["roundtrip"]},
                          {"key": f"{cid}|var", "db_path": db, "sql": a["variant"]}]
        qres = {r["key"]: r for r in execute_many(qjobs, paths, workers=workers, timeout_s=exec_cfg["timeout_s"],
                                                   max_rows=exec_cfg["max_rows"], count_steps=False)}
        recs = []
        for r in chunk:
            x = res[r["cand_id"]]
            ok = is_correct(x)
            rec = {**r, "correct": ok, "error_type": x["error_type"], "error_msg": (x.get("error_msg") or "")[:300] or None,
                   "vm_steps": x.get("vm_steps"), "steps_capped": x.get("steps_error") == "step_budget",
                   "quality": None, "quality_flags": [], "foreign_literals": [], "usable": False, "rel_cost": None}
            if ok:
                e = examples[r["qid"]]
                a = analyses[r["cand_id"]]
                v, _ = verdict(a, golds[r["qid"]]["result_hash"], qres.get(f"{r['cand_id']}|rt"),
                               qres.get(f"{r['cand_id']}|var"))
                fl = foreign_literals(r["sql"], e.gold_sql, e.question, e.evidence)
                rec.update(quality=v, quality_flags=a["flags"], foreign_literals=fl)
                if x.get("vm_steps") is not None:
                    rec["rel_cost"] = rel_cost(x["vm_steps"], r["gold_vm_steps"], c)
                    rec["usable"] = v in OK_QUALITY and not fl and not (set(a["flags"]) - gold_flags[r["qid"]])
            recs.append(rec)
        append_jsonl(out, recs)
        log.info("search round %d: %d queries executed", round_, len(recs))


# State after the rounds so far -----------------------------------------------------------------
def state_after(base_best: dict[str, dict[str, Any]], execs: list[dict[str, Any]], round_: int,
                ) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Best usable query per question and the attempts of rounds <= round_ that did not become best
    (most recent first)."""
    best = dict(base_best)
    per_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in sorted((e for e in execs if e["round"] <= round_), key=lambda e: (e["round"], e["cand_id"])):
        per_q[e["qid"]].append(e)
        if e["usable"] and e["rel_cost"] is not None and e["rel_cost"] < best[e["qid"]]["rel_cost"]:
            best[e["qid"]] = {**e, "source": f"search:r{e['round']}"}
    attempts = {q: [e for e in reversed(v) if e["cand_id"] != best[q].get("cand_id")] for q, v in per_q.items()}
    return best, attempts


def headroom(best: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    r = np.array([best[q]["rel_cost"] for q in qids])
    return {"n": len(qids), "pct_2x": round(100 * float(np.mean(r <= 0.5)), 2),
            "pct_10pct": round(100 * float(np.mean(r <= 0.9)), 2),
            "geomean_rel_cost": round(float(np.exp(np.mean(np.log(r)))), 4)}


def summarize(run_dir: Any, base_best: dict[str, Any], qids: list[str], execs: list[dict[str, Any]],
              rounds_done: int, gen_stats: list[dict[str, Any]], model: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model": model, "n_questions": len(qids), "start": headroom(base_best, qids), "rounds": []}
    for r in range(1, rounds_done + 1):
        best, _ = state_after(base_best, execs, r)
        ex = [e for e in execs if e["round"] == r]
        new = [e for e in ex if not e.get("known")]
        prev, _ = state_after(base_best, execs, r - 1)
        out["rounds"].append({
            "round": r, **headroom(best, qids),
            "n_distinct_answers": len(ex), "n_new": len(new),
            "pct_new_correct": round(100 * float(np.mean([e["correct"] for e in new])), 1) if new else None,
            "n_correct_not_usable": sum(1 for e in new if e["correct"] and not e["usable"]),
            "n_foreign_literals": sum(1 for e in new if e.get("foreign_literals")),
            "n_questions_improved": sum(1 for q in qids if best[q]["rel_cost"] < prev[q]["rel_cost"]),
        })
    out["generation"] = gen_stats
    write_json(run_dir / f"search_summary_{SPLIT}.json", out)
    return out


# Main loop --------------------------------------------------------------------------------------
def search(run: str, backend: Backend, *, cfg: dict[str, Any] | None = None, paths: Paths | None = None,
           limit: int | None = None, workers: int | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = cfg or load_config("search.yaml")
    exec_cfg = load_config("evaluate.yaml")["exec"]
    pool_cfg = load_config("pool.yaml")
    workers = workers or exec_cfg.get("workers")
    run_dir = paths.run_dir(run)
    examples = {e.qid: e for e in load_examples(SPLIT, paths)}
    base_best, known = base_state(paths, cfg["base_run"])
    qids = pick_targets(paths, run, cfg, base_best, {q: e.db_id for q, e in examples.items()}, limit)
    dbs = {examples[q].db_id for q in qids}
    missing = [d for d in sorted(dbs) if not paths.local_db_path(SPLIT, d).exists()]
    if missing:
        raise RuntimeError(f"Databases not staged: {missing}. Run `effsql stage --split {SPLIT}`.")
    golds = ensure_gold_costs(SPLIT, {q: (examples[q].db_id, examples[q].gold_sql) for q in qids}, paths, exec_cfg,
                              workers)
    schemas = schemas_for(SPLIT, dbs, paths)
    templates = load_templates()
    sp = cfg["sampling"]
    exec_path = run_dir / f"search_exec_{SPLIT}.jsonl"
    gen_stats = []
    for r in range(1, cfg["rounds"] + 1):
        execs = list(latest_by(exec_path, key="cand_id", where=is_current_search).values())
        best, attempts = state_after(base_best, execs, r - 1)
        active = [q for q in qids if best[q]["rel_cost"] > cfg["stop_rel_cost"]]
        log.info("search round %d: %d of %d questions still without a %.2fx query", r, len(active), len(qids),
                 cfg["stop_rel_cost"])
        requests = []
        conns: dict[str, Any] = {}
        try:
            for q in sorted(active, key=lambda q: (examples[q].db_id, q)):
                e = examples[q]
                if e.db_id not in conns:
                    conns[e.db_id] = connect_readonly(paths.local_db_path(SPLIT, e.db_id), paths)
                b = best[q]
                hist = history_text(attempts.get(q, []), b["vm_steps"], cfg["history"])
                msgs, level, n_tok = build_search_prompt(templates, e, schemas[e.db_id], b,
                                                         plan_text(conns[e.db_id], b["sql"]), hist,
                                                         cfg["max_prompt_tokens"], backend.count_tokens)
                params = GenParams(n=sp["n"], temperature=sp["temperature"], top_p=sp["top_p"],
                                   max_tokens=sp["max_tokens"], seed=stable_seed(cfg["seed"], q, "search", r))
                requests.append(Request(f"{q}|search|r{r}", msgs, params,
                                        {"qid": q, "split": SPLIT, "db_id": e.db_id, "source": f"search:r{r}",
                                         "variant": "search", "round": r, "prompt_level": level,
                                         "prompt_tokens": n_tok, "ref_rel_cost": b["rel_cost"],
                                         "analysis_only": False}))
        finally:
            for conn in conns.values():
                conn.close()
        st = run_requests(requests, backend, run_dir / f"search_gen_{SPLIT}.jsonl", cfg["chunk_size"],
                          f"search round {r}")
        gen_stats.append({**st, "round": r, "n_active": len(active)})
        execute_round(paths, run, r, examples, golds, known, exec_cfg, pool_cfg, workers)
        execs = list(latest_by(exec_path, key="cand_id", where=is_current_search).values())
        summ = summarize(run_dir, base_best, qids, execs, r, gen_stats, backend.model_id)
        log.info("search round %d done: %s", r, json.dumps(summ["rounds"][-1]))
    execs = list(latest_by(exec_path, key="cand_id", where=is_current_search).values())
    best, _ = state_after(base_best, execs, cfg["rounds"])
    held = holdout_db_ids(paths, cfg)
    summ["final"] = write_targets(run_dir, best, base_best, qids, {q: examples[q].db_id for q in qids}, held)
    write_json(run_dir / f"search_summary_{SPLIT}.json", summ)
    return summ


def holdout_db_ids(paths: Paths, cfg: dict[str, Any]) -> list[str]:
    """The validation databases of the SFT export (never used for training)."""
    f = paths.run_dir(cfg["pilot_exclude_export"], create=False) / "export_meta.json"
    return sorted(read_json(f)["holdout_db_ids"]) if f.exists() else []


def write_targets(run_dir: Any, best: dict[str, dict[str, Any]], base_best: dict[str, dict[str, Any]],
                  qids: list[str], db_of: dict[str, str], held: list[str]) -> dict[str, Any]:
    """cheap_targets_train.jsonl (one row per searched question) and the headroom of the training and the
    validation questions: with the base pool alone and with the search."""
    rows = []
    for q in qids:
        b, p = best[q], base_best[q]
        rows.append({"qid": q, "db_id": db_of[q], "part": "val" if db_of[q] in held else "train",
                     "sql": b["sql"], "vm_steps": b["vm_steps"], "gold_vm_steps": b["gold_vm_steps"],
                     "rel_cost": b["rel_cost"], "source": b.get("source"),
                     "pool_sql": p["sql"], "pool_vm_steps": p["vm_steps"], "pool_rel_cost": p["rel_cost"],
                     "pool_source": p.get("source"), "search_version": SEARCH_VERSION})
    path = run_dir / f"cheap_targets_{SPLIT}.jsonl"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    out: dict[str, Any] = {"holdout_db_ids": held}
    for part in ("train", "val"):
        qs = [r["qid"] for r in rows if r["part"] == part]
        if qs:
            out[part] = {"pool_only": headroom(base_best, qs), "with_search": headroom(best, qs),
                         "by_source": dict(Counter(source_kind(best[q].get("source")) for q in qs))}
    return out


def source_kind(source: Any) -> str:
    """gold, pool (any base pool candidate) or search."""
    s = str(source)
    return "gold" if s == "gold" else "search" if s.startswith("search") else "pool"


def read_targets(paths: Paths, run: str) -> dict[str, dict[str, Any]]:
    """qid -> row of cheap_targets_train.jsonl of a finished search run."""
    path = paths.run_dir(run, create=False) / f"cheap_targets_{SPLIT}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the search (`python -m effsql.gen.search --run {run}`)")
    rows = {r["qid"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}
    old = [q for q, r in rows.items() if r.get("search_version") != SEARCH_VERSION]
    if old:
        raise RuntimeError(f"{len(old)} targets in {path} come from an older search version: rerun the search")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m effsql.gen.search", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="output run, e.g. search_pilot_32b")
    ap.add_argument("--model", default="32b", help="a key of search.yaml models")
    ap.add_argument("--limit", type=int, help="pilot: a fixed random sample of this many questions")
    ap.add_argument("--rounds", type=int, help="override search.yaml rounds")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config("search.yaml")
    if a.rounds:
        cfg["rounds"] = a.rounds
    backend = make_backend({"model": cfg["models"][a.model], "seed": cfg["seed"]})
    print(json.dumps(search(a.run, backend, cfg=cfg, limit=a.limit, workers=a.workers), indent=2, default=str))


if __name__ == "__main__":
    main()
