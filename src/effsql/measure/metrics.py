"""Aggregate per-question evaluation records into the reported metrics (overall and by difficulty).

Per-question record fields used here: qid, difficulty, correct, correct_tied (None when no tied
gold exists), error_type, rel_cost (correct predictions only).
Timing runs: one dict qid -> record per repeat; official records carry reward and time_ratio,
quick records carry quick_ratio and unstable.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from effsql.measure.bird_official import rves_score, ves_score

DIFFICULTIES = ("simple", "moderate", "challenging")


def _pct(xs: list[bool]) -> float | None:
    return 100.0 * sum(xs) / len(xs) if xs else None


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def group_metrics(
    recs: list[dict[str, Any]],
    timing_runs: list[dict[str, dict[str, Any]]],
    preset: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    n = len(recs)
    correct = [bool(r["correct"]) for r in recs]
    costs = [r["rel_cost"] for r in recs if r["correct"] and r.get("rel_cost") is not None]
    out: dict[str, Any] = {"n": n, "ex": _pct(correct)}

    if any(r.get("correct_tied") is not None for r in recs):
        out["ex_tied"] = _pct([bool(r["correct"] or r.get("correct_tied")) for r in recs])
        out["n_with_tied_gold"] = sum(r.get("correct_tied") is not None for r in recs)

    out["rel_cost_median"] = statistics.median(costs) if costs else None
    out["rel_cost_geomean"] = math.exp(statistics.fmean(math.log(c) for c in costs)) if costs else None
    thr = cfg.get("faster_than_gold", 0.9)
    out["faster_than_gold_rate"] = _pct([c < thr for c in costs])
    for tau in cfg.get("taus", [1.0, 1.5, 2.0]):
        out[f"eff_acc@{tau}"] = _pct(
            [bool(r["correct"] and r.get("rel_cost") is not None and r["rel_cost"] <= tau) for r in recs]
        )
    errors = Counter(r["error_type"] or ("wrong_result" if not r["correct"] else "correct") for r in recs)
    out["outcomes"] = dict(errors)
    out["gap_to_oracle"] = None  # filled once a candidate pool exists (stage 5)

    qids = [r["qid"] for r in recs]
    if preset == "official" and timing_runs:
        rves = [rves_score([run.get(q, {}).get("reward", 0) for q in qids]) for run in timing_runs]
        ves = [ves_score([run.get(q, {}).get("time_ratio", 0.0) for q in qids]) for run in timing_runs]
        out["rves_mean"], out["rves_std"] = _mean_std(rves)
        out["ves_mean"], out["ves_std"] = _mean_std(ves)
        out["n_repeats"] = len(timing_runs)
    elif preset == "quick" and timing_runs:
        per_run = []
        unstable = []
        for run in timing_runs:
            ratios = [run[q]["quick_ratio"] for q in qids if run.get(q, {}).get("quick_ratio")]
            per_run.append(statistics.median(ratios) if ratios else None)
            unstable += [bool(run[q].get("unstable")) for q in qids if q in run and "unstable" in run[q]]
        out["quick_ratio_median_mean"], out["quick_ratio_median_std"] = _mean_std(per_run)
        out["quick_unstable_rate"] = _pct(unstable)
        out["n_repeats"] = len(timing_runs)
    return out


def compute_metrics(
    recs: list[dict[str, Any]],
    timing_runs: list[dict[str, dict[str, Any]]],
    preset: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out = {"overall": group_metrics(recs, timing_runs, preset, cfg), "by_difficulty": {}}
    for d in DIFFICULTIES:
        sub = [r for r in recs if r.get("difficulty") == d]
        if sub:
            out["by_difficulty"][d] = group_metrics(sub, timing_runs, preset, cfg)
    return out
