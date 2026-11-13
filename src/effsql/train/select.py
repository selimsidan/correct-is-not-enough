"""Choosing the SFT method on the validation databases (never on dev or minidev, which stay a clean test).

Rule (fixed before any run, 2026-11-14):
  1. Correctness first: keep the methods whose validation EX is not significantly below the best
     method's (exact two-sided McNemar test on the questions both answered, alpha 0.05).
  2. Among those, the most efficient wins: the largest share of questions answered correctly with a
     query at least 2x cheaper than gold (rel_cost <= 0.5, smoothed steps); ties by the lower median
     cost relative to gold among correct answers, then by EX.
Input: eval_train.jsonl of `effsql evaluate --split train` per method (the validation predictions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from effsql.evaluate import is_current
from effsql.io import latest_by, write_json

log = logging.getLogger(__name__)
CHEAP = 0.5  # rel_cost <= 0.5: at least 2x cheaper than gold


def method_stats(recs: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    rows = [recs[q] for q in qids]
    correct = [bool(r["correct"]) for r in rows]
    costs = [r["rel_cost"] for r in rows if r["correct"] and r.get("rel_cost") is not None]
    return {
        "n": len(rows),
        "ex": 100 * float(np.mean(correct)) if rows else None,
        "cheap_2x": 100 * float(np.mean([bool(r["correct"] and r.get("rel_cost") is not None and r["rel_cost"] <= CHEAP)
                                         for r in rows])) if rows else None,
        "rel_cost_median": float(np.median(costs)) if costs else None,
        "slower_2x": 100 * float(np.mean([bool(r["correct"] and r.get("rel_cost") is not None and r["rel_cost"] >= 2.0)
                                          for r in rows])) if rows else None,
    }


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar p-value: does b differ from a on the same questions?"""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    return 1.0 if n10 + n01 == 0 else float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def select_method(eval_dirs: dict[str, Path], alpha: float = 0.05, out: Path | None = None) -> dict[str, Any]:
    """eval_dirs: method name -> run dir holding eval_train.jsonl. Returns the table and the winner."""
    recs = {m: latest_by(d / "eval_train.jsonl", where=is_current) for m, d in eval_dirs.items()}
    missing = [m for m, r in recs.items() if not r]
    if missing:
        raise RuntimeError(f"no validation evaluation for {missing}")
    qids = sorted(set.intersection(*(set(r) for r in recs.values())))
    stats = {m: method_stats(r, qids) for m, r in recs.items()}
    best_ex = max(stats, key=lambda m: (stats[m]["ex"], m))
    ref = [bool(recs[best_ex][q]["correct"]) for q in qids]
    for m in stats:
        mine = [bool(recs[m][q]["correct"]) for q in qids]
        p = 1.0 if m == best_ex else mcnemar_p(ref, mine)
        stats[m]["p_vs_best_ex"] = p
        stats[m]["eligible"] = m == best_ex or p >= alpha
    eligible = [m for m in stats if stats[m]["eligible"]]
    winner = max(eligible, key=lambda m: (stats[m]["cheap_2x"],
                                          -(stats[m]["rel_cost_median"] if stats[m]["rel_cost_median"] is not None else 9e9),
                                          stats[m]["ex"], m))
    res = {"n_questions": len(qids), "alpha": alpha, "best_ex_method": best_ex, "winner": winner,
           "methods": stats, "rule": __doc__.split("Rule", 1)[1].split("Input:")[0].strip()}
    if out:
        write_json(out, res)
    return res


def format_table(res: dict[str, Any]) -> str:
    lines = [f"{'method':<28}{'EX':>7}{'>=2x cheaper':>14}{'median cost':>13}{'>=2x slower':>13}{'p vs best':>11}  eligible"]
    for m, s in res["methods"].items():
        med = f"{s['rel_cost_median']:.3f}" if s["rel_cost_median"] is not None else "-"
        lines.append(f"{m:<28}{s['ex']:>7.1f}{s['cheap_2x']:>13.1f}%{med:>13}{s['slower_2x']:>12.1f}%"
                     f"{s['p_vs_best_ex']:>11.3f}  {'yes' if s['eligible'] else 'no'}")
    lines.append(f"winner: {res['winner']} ({res['n_questions']} validation questions)")
    return "\n".join(lines)
