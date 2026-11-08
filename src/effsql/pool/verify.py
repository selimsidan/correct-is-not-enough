"""`effsql verify-pool`: does the step-count ordering agree with real wall-clock time?

Samples questions whose correct candidates span at least 2 efficiency levels, times each sampled
question's correct candidates with the quick protocol (interleaved, sequential), and reports:
  * Spearman correlation between vm_steps and median time, pooled over all timed candidates
    (log scale) and the mean of per-question correlations (questions with >= 3 candidates);
  * pairwise agreement: share of candidate pairs (same question, different steps) where the
    cheaper-by-steps candidate is also faster, overall, for pairs whose step ratio reaches
    level_ratio, and for pairs where both medians are at least min_time_ms (sub-ms timings are noisy).
Run on a CPU runtime: timings are only comparable on the same runtime type.
"""

from __future__ import annotations

import logging
import math
import random
from itertools import combinations
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from effsql.config import load_config
from effsql.data.stage import stage
from effsql.exec.engine import connect_readonly
from effsql.io import append_jsonl, done_ids, latest_by, write_json
from effsql.measure.timing import time_interleaved
from effsql.paths import Paths, load_paths
from effsql.runmeta import cpu_info

log = logging.getLogger(__name__)
VERIFY_VERSION = 1


def _is_current(r: dict[str, Any]) -> bool:
    return r.get("verify_version") == VERIFY_VERSION


def pick_candidates(group: pd.DataFrame, k: int) -> pd.DataFrame:
    """Cheapest, most expensive and evenly spread candidates in between (at most k)."""
    g = group.sort_values("vm_steps")
    if len(g) <= k:
        return g
    idx = sorted({round(i * (len(g) - 1) / (k - 1)) for i in range(k)})
    return g.iloc[idx]


def agreement_stats(questions: list[dict[str, Any]], level_ratio: float, min_time_s: float) -> dict[str, Any]:
    pairs = {"all": [0, 0], "step_ratio_ge_level": [0, 0], "both_slower_than_min_time": [0, 0]}
    xs, ys, per_q = [], [], []
    for q in questions:
        cs = [c for c in q["candidates"] if c.get("median_s")]
        for c in cs:
            xs.append(math.log(c["vm_steps"] + 1))
            ys.append(math.log(c["median_s"]))
        if len(cs) >= 3:
            rho = spearmanr([c["vm_steps"] for c in cs], [c["median_s"] for c in cs]).statistic
            if rho == rho:  # not NaN
                per_q.append(rho)
        for a, b in combinations(cs, 2):
            if a["vm_steps"] == b["vm_steps"]:
                continue
            agree = (a["vm_steps"] < b["vm_steps"]) == (a["median_s"] < b["median_s"])
            buckets = ["all"]
            lo, hi = sorted((a["vm_steps"], b["vm_steps"]))
            if (hi + 1) >= level_ratio * (lo + 1):
                buckets.append("step_ratio_ge_level")
            if min(a["median_s"], b["median_s"]) >= min_time_s:
                buckets.append("both_slower_than_min_time")
            for k in buckets:
                pairs[k][0] += agree
                pairs[k][1] += 1
    return {
        "n_questions": len(questions), "n_candidates_timed": len(xs),
        "spearman_pooled_log": spearmanr(xs, ys).statistic if len(xs) >= 3 else None,
        "spearman_per_question_mean": sum(per_q) / len(per_q) if per_q else None,
        "n_questions_with_per_question_rho": len(per_q),
        "pairwise_agreement": {k: {"agree": v[0], "n_pairs": v[1], "rate": v[0] / v[1] if v[1] else None}
                               for k, v in pairs.items()},
    }


def verify_pool(split: str, run: str, *, sample: int | None = None, paths: Paths | None = None,
                pool_cfg: dict[str, Any] | None = None, timing_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = pool_cfg or load_config("pool.yaml")
    vcfg = cfg["verify"]
    tq = (timing_cfg or load_config("evaluate.yaml")["timing"])["quick"]
    run_dir = paths.run_dir(run)
    pool = pd.read_parquet(run_dir / f"pool_{split}.parquet")
    timed = pool[pool["correct"] & pool["vm_steps"].notna() & ~pool["steps_capped"]]
    levels = timed.groupby("qid")["efficiency_level"].nunique()
    eligible = sorted(levels[levels >= 2].index)
    n = sample or vcfg["sample"]
    rng = random.Random(vcfg.get("seed", 0))
    chosen = sorted(rng.sample(eligible, min(n, len(eligible))))
    log.info("verify %s: %d eligible questions, timing %d", split, len(eligible), len(chosen))
    if not chosen:
        return {"n_questions": 0, "note": "no question with 2 or more efficiency levels"}

    db_of = timed.drop_duplicates("qid").set_index("qid")["db_id"]
    stage(split, db_ids=sorted({db_of[q] for q in chosen}), paths=paths)  # idempotent

    out = run_dir / f"verify_{split}.jsonl"
    done = done_ids(out, where=_is_current)
    conns: dict[str, Any] = {}
    try:
        for k, q in enumerate([q for q in chosen if q not in done], 1):
            g = pick_candidates(timed[timed["qid"] == q], vcfg["max_candidates"])
            db = db_of[q]
            if db not in conns:
                conns[db] = connect_readonly(paths.local_db_path(split, db), paths)
            res = time_interleaved(conns[db], list(g["sql_original"]), warmup=tq["warmup"], runs=tq["runs"],
                                   timeout_s=tq["timeout_s"])
            cands = [{"cand_id": cid, "vm_steps": int(st), "median_s": t.median, "cv": t.cv}
                     for cid, st, t in zip(g["cand_id"], g["vm_steps"], res.timings)] if not res.error else []
            append_jsonl(out, [{"qid": q, "candidates": cands, "error": res.error, "verify_version": VERIFY_VERSION}])
            if k % 20 == 0:
                log.info("verify: %d questions timed", k)
    finally:
        for c in conns.values():
            c.close()

    recs = [r for q, r in latest_by(out, where=_is_current).items() if q in set(chosen)]
    stats = agreement_stats([r for r in recs if not r["error"]], cfg["level_ratio"], vcfg["min_time_ms"] / 1000)
    stats.update(n_errors=sum(1 for r in recs if r["error"]), machine=cpu_info())
    write_json(run_dir / f"verify_stats_{split}.json", stats)
    return stats
