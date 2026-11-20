"""Preference pairs for "cost first, then accuracy" (CPU): `python -m effsql.train pairs`.

Chosen answers come from the cheap targets of the search (`cheap_targets_train.jsonl`): per question the
cheapest usable correct query known (strict rule: no first-row pick that gold does not use, no constant
found nowhere in the question, evidence or gold). Rejected answers are, by default, the current model's
own answers (`rl --settings own_answers`), so DPO pushes away from what the model really writes.

Per question (only questions with at least one efficiency pair, as in the first DPO):
  efficiency   chosen = the target; rejected = one of the model's own correct answers costing at least
               min_ratio x the target and min_abs_steps more (a correct answer whose counting stopped at
               the cap counts as costlier). Up to `efficiency_pairs`.
  correctness  chosen = the target; rejected = one of its own answers that is wrong, fails, times out,
               or matches gold only by an arbitrary row pick or a foreign constant. Up to
               `correctness_pairs`.
Rejected answers are distinct queries (canonical SQL), the most frequent first, each used once.

Ablations (same rules, other sources):
  --chosen gold      chosen = gold (the generated queries are not used at all)
  --chosen pool      chosen = the cheapest usable query of the base pool alone (no search)
  --rejected pool    rejected = base pool candidates (other models' queries, the first DPO's choice)

Validation pairs come from the held-out databases of the SFT export (they only monitor the DPO loss).
Output: runs/<run>/data/dpo_{train,val}.jsonl and export_meta.json, read by `python -m effsql.train dpo`.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.gen.backend import stable_seed
from effsql.gen.parse import cand_id, canonicalize
from effsql.gen.search import read_targets, source_kind
from effsql.io import append_jsonl, read_json, read_jsonl, write_json
from effsql.paths import Paths, load_paths
from effsql.runmeta import write_run_meta
from effsql.train.data import EXPORT_VERSION, _quantiles

log = logging.getLogger(__name__)
CORRECT = "correct"
NO_SQL = ("no_sql", "cut_off")
CHOSEN = ("search", "pool", "gold")
REJECTED = ("own", "pool")


def canon(sql: str) -> str:
    c, _ = canonicalize(sql)
    return c or " ".join(sql.split())


def own_answers(paths: Paths, own_run: str) -> dict[str, list[dict[str, Any]]]:
    """qid -> distinct answers of the sampling run: sql, status, steps, count (most frequent first)."""
    run_dir = paths.run_dir(own_run, create=False)
    files = sorted((run_dir / "iters").glob("*/rewards.jsonl"))
    if not files:
        raise FileNotFoundError(f"no scored answers in {run_dir}: run `python -m effsql.train rl --run {own_run} "
                                "--settings own_answers`")
    out: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        for r in read_jsonl(f):
            by: dict[str, dict[str, Any]] = {}
            for sql, st, steps in zip(r["sql"], r["status"], r["steps"]):
                if not sql or st in NO_SQL:
                    continue
                c = canon(sql)
                if c in by:
                    by[c]["count"] += 1
                else:
                    by[c] = {"sql": sql, "canon": c, "status": st, "steps": steps, "count": 1}
            out[r["qid"]] = sorted(by.values(), key=lambda a: (-a["count"], a["canon"]))
    return out


def pool_answers(paths: Paths, pool_run: str, qids: set[str], seed: int) -> dict[str, list[dict[str, Any]]]:
    """qid -> the base pool's model candidates in the same form (status, steps), in a seeded random order."""
    df = pd.read_parquet(paths.run_dir(pool_run, create=False) / "pool_train.parquet",
                         columns=["qid", "cand_id", "is_gold", "correct", "quality", "sql_original", "sql_canonical",
                                  "vm_steps", "steps_capped", "steps_lower_bound", "error_type"])
    df = df[df["qid"].isin(qids) & ~df["is_gold"]]
    out: dict[str, list[dict[str, Any]]] = {}
    for q, g in df.groupby("qid"):
        rows = []
        for r in g.sort_values("cand_id").itertuples(index=False):
            if not isinstance(r.sql_original, str) or not r.sql_original.strip() or r.error_type == "parse_fail":
                continue
            if r.correct:
                status = "first_row_pick" if r.quality in ("fragile", "unverifiable") else CORRECT
            else:
                status = r.error_type if isinstance(r.error_type, str) else "wrong_result"
            steps = None if r.steps_capped or pd.isna(r.vm_steps) else int(r.vm_steps)
            rows.append({"sql": r.sql_original, "canon": r.sql_canonical or canon(r.sql_original),
                         "status": status, "steps": steps, "count": 1})
        rng = np.random.default_rng(stable_seed(seed, "pairs_pool", q))
        out[q] = [rows[i] for i in rng.permutation(len(rows))]
    return out


def chosen_for(t: dict[str, Any], kind: str, gold_sql: str) -> tuple[str, int, str]:
    """(sql, steps, source) of the chosen answer."""
    if kind == "search":
        return t["sql"], int(t["vm_steps"]), str(t["source"])
    if kind == "pool":
        return t["pool_sql"], int(t["pool_vm_steps"]), str(t["pool_source"])
    return gold_sql, int(t["gold_vm_steps"]), "gold"


def question_pairs(answers: list[dict[str, Any]], ch_sql: str, ch_steps: int, cfg: dict[str, Any]
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """(efficiency rejected, correctness rejected, outcome) for one question; outcome says why there is no
    efficiency pair (no_answers, writes_cheap_form: every own correct answer within the margin, only_wrong)."""
    c = canon(ch_sql)
    eff, corr = [], []
    any_correct = False
    for a in answers:
        if a["canon"] == c:
            any_correct = True
            continue
        if a["status"] == CORRECT:
            any_correct = True
            s = a["steps"]
            if s is None or (s >= cfg["min_ratio"] * ch_steps and s - ch_steps >= cfg["min_abs_steps"]):
                eff.append(a)
        else:
            corr.append(a)
    outcome = "pairs" if eff else ("no_answers" if not answers else "writes_cheap_form" if any_correct else "only_wrong")
    return eff[: cfg["efficiency_pairs"]], corr[: cfg["correctness_pairs"]], outcome


def _rec(part: str, kind: str, q: str, db: str, ch: tuple[str, int, str], rj: dict[str, Any], rej_source: str
         ) -> dict[str, Any]:
    sql, steps, src = ch
    return {"arm": "dpo", "part": part, "epoch": 0, "qid": q, "db_id": db, "pair_type": kind,
            "chosen_cand": cand_id(q, canon(sql)), "rejected_cand": cand_id(q, rj["canon"]),
            "chosen_sql": sql, "rejected_sql": rj["sql"], "chosen_source": src, "rejected_source": rej_source,
            "chosen_steps": steps, "rejected_steps": rj["steps"], "rejected_status": rj["status"],
            "rejected_count": rj["count"],
            "cost_ratio": None if rj["steps"] is None or kind != "efficiency" else rj["steps"] / max(steps, 1),
            "export_version": EXPORT_VERSION}


def build_pairs(targets_run: str, run: str, *, own_run: str | None = None, chosen: str = "search",
                rejected: str = "own", sft_export: str = "sft_v1", pool_run: str = "pool_v1",
                paths: Paths | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")["pairs"]
    if chosen not in CHOSEN or rejected not in REJECTED:
        raise ValueError(f"--chosen one of {CHOSEN}, --rejected one of {REJECTED}")
    if rejected == "own" and not own_run:
        raise ValueError("--own-run is needed for rejected = own")
    smeta = read_json(paths.run_dir(sft_export, create=False) / "export_meta.json")
    held = set(smeta["holdout_db_ids"])
    targets = read_targets(paths, targets_run)
    examples = {e.qid: e for e in load_examples("train", paths)}
    answers = own_answers(paths, own_run) if rejected == "own" else pool_answers(paths, pool_run, set(targets), cfg["seed"])
    recs: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    outcomes: dict[str, Counter] = defaultdict(Counter)
    for q in sorted(targets):
        t = targets[q]
        part = "val" if t["db_id"] in held else "train"
        ch = chosen_for(t, chosen, examples[q].gold_sql)
        eff, corr, outcome = question_pairs(answers.get(q, []), ch[0], ch[1], cfg)
        outcomes[part][outcome] += 1
        if not eff:
            continue
        recs[part] += [_rec(part, "efficiency", q, t["db_id"], ch, a, rejected) for a in eff]
        recs[part] += [_rec(part, "correctness", q, t["db_id"], ch, a, rejected) for a in corr]

    run_dir = paths.run_dir(run)
    stats: dict[str, Any] = {}
    files: dict[str, str] = {}
    for part, rs in recs.items():
        path = run_dir / "data" / f"dpo_{part}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.unlink(missing_ok=True)
        append_jsonl(tmp, rs)
        tmp.replace(path)
        files[f"dpo_{part}"] = str(path)
        first = {r["qid"]: r for r in rs}
        stats[f"dpo_{part}"] = {
            "n_examples": len(rs), "n_questions": len(first), "by_type": dict(Counter(r["pair_type"] for r in rs)),
            "questions": dict(outcomes[part]),
            "chosen_source": dict(Counter(source_kind(r["chosen_source"]) for r in first.values())),
            "efficiency_cost_ratio": _quantiles([r["cost_ratio"] for r in rs if r["cost_ratio"] is not None]),
            "rejected_status": dict(Counter(r["rejected_status"] for r in rs)),
            "rejected_count": _quantiles([r["rejected_count"] for r in rs]),
        }
    meta = {"pool_run": smeta["pool_run"], "pool_sha256": smeta["pool_sha256"], "export_version": EXPORT_VERSION,
            "arms": ["dpo"], "kind": "pairs", "targets_run": targets_run, "own_run": own_run, "chosen": chosen,
            "rejected": rejected, "pairs_cfg": cfg, "sft_export": sft_export,
            "holdout_db_ids": sorted(held), "val_qids": smeta["val_qids"],
            "n_questions": {p: s["n_questions"] for p, s in ((p[4:], s) for p, s in stats.items())},
            "files": files, "stats": stats}
    write_json(run_dir / "export_meta.json", meta)
    write_run_meta(run_dir, extra={"command": "pairs", "targets_run": targets_run, "own_run": own_run,
                                   "chosen": chosen, "rejected": rejected})
    log.info("pairs %s: %s", run, {p: (s["n_questions"], s["by_type"]) for p, s in stats.items()})
    return meta
