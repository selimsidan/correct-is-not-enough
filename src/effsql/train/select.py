"""Choosing between trained models on the validation databases (never on dev or minidev, which stay a clean test).

Rule (2026-11-15; replaces the first rule, which ranked by the share of answers at least 2x cheaper
than gold: about 14 questions per model on validation, too few to separate models):
  1. Correctness first: keep the models whose validation EX is not significantly below the best
     model's (exact two-sided McNemar test on the same questions, alpha 0.05).
  2. Among those, the highest step-based R-VES wins (BIRD's reward per question, with step counts in
     place of seconds; gold = 100, wrong = 0). Ties by efficiency per correct answer, then by EX.
Also reported, not used to choose: VES (unbounded, driven by a few extreme speed-ups), answers >= 2x
cheaper or slower than gold, median cost, and every model against the winner on the same questions
(R-VES difference with a Wilcoxon signed-rank test; how often each is cheaper when both are correct,
with a sign test).
Input: eval_train.jsonl of `effsql evaluate --split train` per model (the validation predictions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from effsql.evaluate import is_current
from effsql.io import latest_by, write_json
from effsql.measure.metrics import step_scores

log = logging.getLogger(__name__)
CHEAP = 0.5  # rel_cost <= 0.5: at least 2x cheaper than gold
SAME = 0.1  # both correct: costs within 10% count as equal in the cheaper/costlier counts


def method_stats(recs: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    rows = [recs[q] for q in qids]
    correct = [bool(r["correct"]) for r in rows]
    costs = [r["rel_cost"] for r in rows if r["correct"] and r.get("rel_cost") is not None]
    scores = [step_scores(r) for r in rows]
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "ex": 100 * float(np.mean(correct)),
        "rves": float(np.mean([s[0] for s in scores])),
        "eff_per_correct": float(np.mean([s[0] for s, c in zip(scores, correct) if c])) if any(correct) else None,
        "ves": float(np.mean([s[1] for s in scores])),
        "cheap_2x": 100 * float(np.mean([bool(r["correct"] and r.get("rel_cost") is not None and r["rel_cost"] <= CHEAP)
                                         for r in rows])),
        "slower_2x": 100 * float(np.mean([bool(r["correct"] and r.get("rel_cost") is not None and r["rel_cost"] >= 2.0)
                                          for r in rows])),
        "rel_cost_median": float(np.median(costs)) if costs else None,
        "rel_cost_geomean": float(np.exp(np.mean(np.log(costs)))) if costs else None,
    }


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar p-value: does b differ from a on the same questions?"""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    return 1.0 if n10 + n01 == 0 else float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def paired(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    """Model b against model a on the same questions."""
    ra = np.array([step_scores(a[q])[0] for q in qids])
    rb = np.array([step_scores(b[q])[0] for q in qids])
    d = rb - ra
    p_rves = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    cheaper = costlier = 0
    ratios = []
    for q in qids:
        x, y = a[q], b[q]
        if x["correct"] and y["correct"] and x.get("rel_cost") and y.get("rel_cost"):
            r = y["rel_cost"] / x["rel_cost"]
            ratios.append(r)
            cheaper += r < 1 - SAME
            costlier += r > 1 / (1 - SAME)
    return {
        "rves_diff": float(d.mean()), "p_rves": p_rves,
        "p_ex": mcnemar_p([bool(a[q]["correct"]) for q in qids], [bool(b[q]["correct"]) for q in qids]),
        "cheaper": int(cheaper), "costlier": int(costlier), "n_both_correct": len(ratios),
        # geometric mean of b's cost / a's cost on the questions both answer correctly
        "cost_ratio": float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
        "p_cost": float(binomtest(cheaper, cheaper + costlier, 0.5).pvalue) if cheaper + costlier else 1.0,
    }


def select_method(eval_dirs: dict[str, Path], alpha: float = 0.05, out: Path | None = None) -> dict[str, Any]:
    """eval_dirs: model name -> run dir holding eval_train.jsonl. Returns the table and the winner."""
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
    winner = max(eligible, key=lambda m: (round(stats[m]["rves"], 6), stats[m]["eff_per_correct"] or 0.0,
                                          stats[m]["ex"], m))
    for m in stats:
        stats[m]["vs_winner"] = None if m == winner else paired(recs[winner], recs[m], qids)
    res = {"n_questions": len(qids), "alpha": alpha, "best_ex_method": best_ex, "winner": winner,
           "methods": stats, "rule": __doc__.split("Rule", 1)[1].split("Input:")[0].strip()}
    if out:
        write_json(out, res)
    return res


def _f(x: float | None, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def format_table(res: dict[str, Any]) -> str:
    """The selection table, then each model against the winner on the same questions."""
    w = max(28, max(len(m) for m in res["methods"]) + 2)
    lines = [f"{'model':<{w}}{'EX':>7}{'R-VES':>8}{'R-VES/EX':>10}{'VES':>8}{'>=2x cheaper':>14}"
             f"{'>=2x slower':>13}{'median cost':>13}{'p EX vs best':>14}  eligible"]
    for m, s in res["methods"].items():
        lines.append(f"{m:<{w}}{s['ex']:>7.1f}{s['rves']:>8.1f}{_f(s['eff_per_correct'], '.1f'):>10}{s['ves']:>8.1f}"
                     f"{s['cheap_2x']:>13.1f}%{s['slower_2x']:>12.1f}%{_f(s['rel_cost_median'], '.3f'):>13}"
                     f"{s['p_vs_best_ex']:>14.3f}  {'yes' if s['eligible'] else 'no'}")
    lines.append(f"winner: {res['winner']} (highest R-VES among eligible; {res['n_questions']} validation questions)")
    lines.append("")
    lines.append(f"against the winner, same questions:")
    lines.append(f"{'model':<{w}}{'R-VES diff':>11}{'p':>8}{'p EX':>8}{'cheaper':>9}{'costlier':>10}{'p':>8}")
    for m, s in res["methods"].items():
        v = s["vs_winner"]
        if v:
            lines.append(f"{m:<{w}}{v['rves_diff']:>+11.2f}{v['p_rves']:>8.3f}{v['p_ex']:>8.3f}"
                         f"{v['cheaper']:>9}{v['costlier']:>10}{v['p_cost']:>8.3f}")
    lines.append("(cheaper/costlier: questions both answer correctly where this model's query costs at least 10% "
                 "less / more than the winner's)")
    return "\n".join(lines)


def select_cost(ref: str, eval_dirs: dict[str, Path], alpha: float = 0.05, out: Path | None = None) -> dict[str, Any]:
    """The rule of "cost first, then accuracy" (fixed 2026-11-21, before any of its models were trained),
    on the validation databases against the current model `ref`:
      1. Accuracy: a candidate is allowed unless its EX is lower than ref's with a significant exact
         McNemar test (alpha 0.05).
      2. Cost: among allowed candidates, the lowest cost relative to ref on the questions both answer
         correctly (geometric mean of the ratio) wins, if it is below 1. Otherwise ref is kept.
    eval_dirs: model name -> run dir holding eval_train.jsonl, ref included."""
    recs = {m: latest_by(d / "eval_train.jsonl", where=is_current) for m, d in eval_dirs.items()}
    missing = [m for m, r in recs.items() if not r]
    if missing:
        raise RuntimeError(f"no validation evaluation for {missing}")
    qids = sorted(set.intersection(*(set(r) for r in recs.values())))
    stats = {m: method_stats(r, qids) for m, r in recs.items()}
    for m, st in stats.items():
        v = None if m == ref else paired(recs[ref], recs[m], qids)
        st["vs_ref"] = v
        st["eligible"] = m != ref and not (st["ex"] < stats[ref]["ex"] and v["p_ex"] < alpha)
    ok = [m for m in stats if stats[m]["eligible"] and stats[m]["vs_ref"]["cost_ratio"] is not None]
    best = min(ok, key=lambda m: (stats[m]["vs_ref"]["cost_ratio"], -stats[m]["ex"], m)) if ok else None
    winner = best if best is not None and stats[best]["vs_ref"]["cost_ratio"] < 1 else ref
    res = {"n_questions": len(qids), "alpha": alpha, "ref": ref, "winner": winner, "methods": stats,
           "rule": select_cost.__doc__.split("\n", 1)[1].split("eval_dirs:")[0].strip()}
    if out:
        write_json(out, res)
    return res


def format_cost_table(res: dict[str, Any]) -> str:
    w = max(28, max(len(m) for m in res["methods"]) + 2)
    lines = [(f"{'model':<{w}}{'EX':>7}{'EX diff':>9}{'p EX':>8}{'cost ratio':>12}{'cheaper':>9}{'costlier':>10}"
              f"{'p':>8}{'>=2x cheaper':>14}{'geomean cost':>14}  allowed")]
    for m, s in res["methods"].items():
        v = s["vs_ref"] or {}
        ref_ex = res["methods"][res["ref"]]["ex"]
        lines.append(f"{m:<{w}}{s['ex']:>7.1f}{s['ex'] - ref_ex:>+9.1f}{_f(v.get('p_ex'), '.3f'):>8}"
                     f"{_f(v.get('cost_ratio'), '.3f'):>12}{v.get('cheaper', '-'):>9}{v.get('costlier', '-'):>10}"
                     f"{_f(v.get('p_cost'), '.3f'):>8}{s['cheap_2x']:>13.1f}%{_f(s['rel_cost_geomean'], '.3f'):>14}"
                     f"  {'ref' if m == res['ref'] else ('yes' if s['eligible'] else 'no')}")
    lines.append(f"winner: {res['winner']} ({res['n_questions']} validation questions; cost ratio = geometric mean of "
                 f"this model's cost / {res['ref']}'s on questions both answer correctly)")
    return "\n".join(lines)
