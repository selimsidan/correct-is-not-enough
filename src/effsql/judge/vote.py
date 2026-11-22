"""Accuracy with execution at answer time (notebook 83, CPU).

`python -m effsql.judge exec`: every distinct candidate of a split is compiled against the schema (EXPLAIN,
no table data read) and run once with the answer-time cap (judge.yaml vote.timeout_s), in parallel, as it
would be when answering. runs/<cand_run>/exec_<split>.jsonl: {qid, results: [{compiles, ok, error_type,
result_hash, n_rows, elapsed_ms, preview}]} in candidate order (resumable by question).

`python -m effsql.judge vote`: one answer per question by each method, from what is known at answer time
only (the candidates, how often each was written, the judge's p, the execution results):
  judge          the highest p (the judge alone, as <judge sel>_p)
  judge_compile  the highest p among the candidates that compile (all candidates when none does)
  result_vote    candidates that ran are grouped by their result; the group written most often wins (ties:
                 the group with the higher best p); the answer is that group's highest-p candidate
  weighted_vote  the same, each written copy counting with its p
  <vote>_nonempty  either vote, where candidates with an empty result do not vote when some candidate
                 returned rows (an empty group wins only when nothing returned rows)
The vote methods fall back to judge_compile when no candidate ran. The method with the highest validation EX
(from the labels of `python -m effsql.judge label`) is kept; ties go to the simpler one (earlier in
vote.methods). Dev and mini-dev are never used to choose.

Prediction runs (read by `eval-val`, `effsql evaluate` and `report` like any model): <out>_<method> for
every method and <out> for the kept one. selection.json / selection.txt in <out>: label-based EX per method
and split with McNemar tests against the generator's own answer, the oracle, and answer-time execution
statistics (the time a question waits when its candidates run in parallel).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from effsql.config import load_config
from effsql.io import append_jsonl, done_ids, latest_by, write_json
from effsql.judge.candidates import SPLITS, data_split, read_candidates, read_labels
from effsql.judge.score import read_scores
from effsql.judge.select import greedy_index, oracle, outcome, write_predictions
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
METHODS = ("judge", "judge_compile", "result_vote", "weighted_vote", "result_vote_nonempty",
           "weighted_vote_nonempty")
EXEC_FIELDS = ("compiles", "ok", "error_type", "result_hash", "n_rows", "elapsed_ms", "preview")


def exec_jobs(recs: list[dict[str, Any]], split: str, paths: Paths, preview: int) -> list[dict[str, Any]]:
    """One job per candidate, keys numbered in order (question by question, candidate by candidate)."""
    jobs = []
    for r in recs:
        for c in r["cands"]:
            jobs.append({"key": len(jobs), "db_path": paths.local_db_path(data_split(split), r["db_id"]),
                         "sql": c["sql"], "compile": True, "preview": preview})
    return jobs


def exec_candidates(run: str, split: str, *, paths: Paths | None = None, jcfg: dict[str, Any] | None = None,
                    workers: int | None = None, chunk: int = 200) -> dict[str, Any]:
    from effsql.exec.engine import batched, execute_many

    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    vc = jcfg["vote"]
    cands = read_candidates(paths, run, split)
    out = paths.run_dir(run) / f"exec_{split}.jsonl"
    done = done_ids(out)
    todo = [r for r in cands if r["qid"] not in done]
    log.info("exec %s: %d questions to run (%d done)", split, len(todo), len(done))
    for part in batched(todo, chunk):
        jobs = exec_jobs(part, split, paths, vc["preview_rows"])
        res = {x["key"]: x for x in execute_many(jobs, paths, workers=workers, timeout_s=vc["timeout_s"],
                                                 max_rows=vc["max_rows"], count_steps=False)} if jobs else {}
        recs, k = [], 0
        for r in part:
            rows = []
            for _ in r["cands"]:
                rows.append({f: res[k].get(f) for f in EXEC_FIELDS})
                k += 1
            recs.append({"qid": r["qid"], "results": rows})
        append_jsonl(out, recs)
        log.info("exec %s: %d more questions", split, len(recs))
    return {"split": split, "n_questions": len(cands), "n_run_now": len(todo)}


def read_exec(paths: Paths, run: str, split: str) -> dict[str, dict[str, Any]]:
    path = paths.run_dir(run, create=False) / f"exec_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m effsql.judge exec` first")
    return latest_by(path)


def method_index(method: str, cands: list[dict[str, Any]], p: list[float] | None,
                 res: list[dict[str, Any]] | None) -> int | None:
    """The chosen candidate of one question. Without judge scores the generator's own answer ranks first;
    without execution results the vote methods fall back to the judge."""
    n = len(cands)
    if not n:
        return None
    if p is None or len(p) != n:
        p = [float("greedy" in c["sources"]) for c in cands]
    if res is not None and len(res) != n:
        res = None

    def best(idx: list[int]) -> int:
        return max(idx, key=lambda i: (p[i], -i))

    everyone = list(range(n))
    if method == "judge":
        return best(everyone)
    compiled = [i for i in everyone if res and res[i].get("compiles")]
    if method == "judge_compile":
        return best(compiled or everyone)
    base, nonempty = method.removesuffix("_nonempty"), method.endswith("_nonempty")
    if base not in ("result_vote", "weighted_vote"):
        raise ValueError(f"unknown method {method!r}")
    ran = [i for i in everyone if res and res[i].get("ok")]
    if nonempty and any(res[i].get("n_rows") for i in ran):
        ran = [i for i in ran if res[i].get("n_rows")]
    groups: dict[str, list[int]] = {}
    for i in ran:
        groups.setdefault(res[i]["result_hash"], []).append(i)
    if not groups:
        return best(compiled or everyone)
    weight = [cands[i]["n"] * (p[i] if base == "weighted_vote" else 1.0) for i in everyone]
    top = max(groups.values(), key=lambda g: (sum(weight[i] for i in g), max(p[i] for i in g)))
    return best(top)


def exec_stats(ex: dict[str, dict[str, Any]], qids: list[str], cap_s: float) -> dict[str, Any]:
    """Answer-time execution: a question waits for its slowest candidate when all run in parallel."""
    wait, capped, comp, ran, n = [], 0, 0, 0, 0
    for q in qids:
        rs = ex.get(q, {}).get("results", [])
        if not rs:
            continue
        ms = [min(float(r.get("elapsed_ms") or 0.0), 1000 * cap_s) for r in rs]
        wait.append(max(ms))
        capped += any(r.get("error_type") == "timeout" for r in rs)
        comp += sum(bool(r.get("compiles")) for r in rs)
        ran += sum(bool(r.get("ok")) for r in rs)
        n += len(rs)
    w = np.asarray(wait) if wait else np.zeros(1)
    return {"wait_ms_median": float(np.median(w)), "wait_ms_p90": float(np.percentile(w, 90)),
            "wait_ms_max": float(w.max()), "pct_questions_capped": 100 * capped / max(len(wait), 1),
            "pct_candidates_compile": 100 * comp / max(n, 1), "pct_candidates_ran": 100 * ran / max(n, 1)}


def vote(cand_run: str, judge_run: str, out: str, *, paths: Paths | None = None,
         jcfg: dict[str, Any] | None = None, splits: tuple[str, ...] = SPLITS) -> dict[str, Any]:
    from effsql.train.select import mcnemar_p

    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    vc = jcfg["vote"]
    methods = [m for m in vc["methods"] if m in METHODS]
    res: dict[str, Any] = {"cand_run": cand_run, "judge_run": judge_run, "methods": methods,
                           "timeout_s": vc["timeout_s"], "splits": {}}
    picks: dict[str, dict[str, dict[str, int | None]]] = {}
    by_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in splits:
        cands = read_candidates(paths, cand_run, split)
        scores, ex, labels = read_scores(paths, cand_run, judge_run, split), read_exec(paths, cand_run, split), \
            read_labels(paths, cand_run, split)
        qids = [r["qid"] for r in cands]
        by_q = {r["qid"]: r["cands"] for r in cands}
        by_split[split] = by_q
        picks[split] = {m: {q: method_index(m, by_q[q], (scores.get(q) or {}).get("p"),
                                            (ex.get(q) or {}).get("results")) for q in qids} for m in methods}
        s: dict[str, Any] = {"n_questions": len(qids), "execution": exec_stats(ex, qids, vc["timeout_s"])}
        if labels is not None and all(q in labels for q in qids):
            greedy = {q: greedy_index(by_q[q]) for q in qids}
            g = outcome(greedy, labels, qids)
            s["greedy"] = {k: v for k, v in g.items() if k != "correct"}
            s["methods"] = {}
            for m in methods:
                o = outcome(picks[split][m], labels, qids, greedy)
                o["mcnemar_p_vs_greedy"] = mcnemar_p(g["correct"], o["correct"])
                o["n_changed"] = sum(picks[split][m][q] != greedy[q] for q in qids)
                s["methods"][m] = o
            s["oracle"] = oracle(labels, qids)
            ran = {q: {**labels[q], "labels": [lab for lab, r in zip(labels[q]["labels"], ex[q]["results"])
                                               if r.get("ok")]} for q in qids if q in ex}
            s["oracle_ran"] = oracle(ran, [q for q in qids if q in ran])
        res["splits"][split] = s
    val = res["splits"].get("val", {}).get("methods")
    if not val:
        raise RuntimeError("validation needs labels and execution results: run `label` and `exec` for val first")
    chosen = max(methods, key=lambda m: (val[m]["ex"], -methods.index(m)))
    res["chosen"] = chosen
    info = {"cand_run": cand_run, "judge_run": judge_run, "generator": jcfg["candidates"]["generator"],
            "timeout_s": vc["timeout_s"], "chosen_on": "val"}
    for split, by_q in by_split.items():
        for m in methods:
            write_predictions(paths, f"{out}_{m}", split, m, by_q, picks[split][m], info)
        write_predictions(paths, out, split, chosen, by_q, picks[split][chosen], info)
    for s in res["splits"].values():
        for o in s.get("methods", {}).values():
            o.pop("correct", None)
    write_json(paths.run_dir(out) / "selection.json", res)
    (paths.run_dir(out) / "selection.txt").write_text(format_vote(res) + "\n", encoding="utf-8")
    return res


def _f(x: Any, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def format_vote(res: dict[str, Any]) -> str:
    lines = [f"method kept on validation: {res['chosen']} (highest validation EX; ties: the simpler method)", ""]
    for split, s in res["splits"].items():
        e = s["execution"]
        lines.append(f"== {split} ({s['n_questions']} questions){'' if split == 'val' else '  [not used to choose]'} ==")
        if "methods" not in s:
            lines.append("  no labels (run `python -m effsql.judge label`)")
        else:
            lines.append(f"  {'method':<24}{'EX':>7}{'diff':>7}{'p':>8}{'changed':>9}")
            g = s["greedy"]["ex"]
            lines.append(f"  {'generator alone':<24}{_f(g, '7.1f')}")
            for m, o in s["methods"].items():
                lines.append(f"  {m:<24}{_f(o['ex'], '7.1f')}{o['ex'] - g:>+7.1f}{o['mcnemar_p_vs_greedy']:>8.3f}"
                             f"{o['n_changed']:>9}")
            lines.append(f"  {'oracle':<24}{_f(s['oracle']['ex'], '7.1f')}   (some candidate correct; "
                         f"{_f(s['oracle_ran']['ex'], '.1f')} among those that ran within the cap)")
        lines.append(f"  answer-time execution (cap {res['timeout_s']} s, candidates in parallel): wait median "
                     f"{e['wait_ms_median']:.0f} ms, p90 {e['wait_ms_p90']:.0f} ms, max {e['wait_ms_max']:.0f} ms; "
                     f"{e['pct_questions_capped']:.1f}% of questions had a candidate stopped at the cap; "
                     f"{e['pct_candidates_compile']:.1f}% of candidates compile, {e['pct_candidates_ran']:.1f}% ran")
        lines.append("")
    lines.append("EX from executing the candidates with the evaluation's limits (labels). diff and p: against the "
                 "generator's own answer on the same questions (exact McNemar). changed: questions whose answer "
                 "differs from the generator's.")
    return "\n".join(lines)
