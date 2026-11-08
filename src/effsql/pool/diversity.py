"""`effsql diversity`: are a question's correct candidates at clearly different efficiency levels?

Levels: sort correct candidates by cost (steps + c, c = progress_n; censored candidates use their
lower bound). A new level starts when a cost is at least level_ratio x the current level's first
cost AND at least min_abs_steps more. A question is diverse with >= min_levels levels. Otherwise
it is flagged with a reason: gold_not_ok, no_correct, one_correct or same_cost.
Each call writes diversity_<split>_r<round>.jsonl and appends a summary to diversity_summary.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from effsql.config import load_config
from effsql.io import latest_by, read_json, write_json
from effsql.paths import Paths, load_paths
from effsql.pool.screen import is_current_screen, load_candidates

log = logging.getLogger(__name__)


def candidate_cost(s: dict[str, Any]) -> int | None:
    """Steps used for levels: counted steps, or the lower bound for censored candidates."""
    if s.get("vm_steps") is not None:
        return s["vm_steps"]
    return s.get("steps_lower_bound")


def assign_levels(costs: list[int], c: float, ratio: float, min_abs: float) -> list[int]:
    """Level index for each cost (input order preserved)."""
    order = sorted(range(len(costs)), key=lambda i: costs[i])
    levels = [0] * len(costs)
    level, base = 0, None
    for i in order:
        x = costs[i]
        if base is None:
            base = x
        elif (x + c) >= ratio * (base + c) and x - base >= min_abs:
            level += 1
            base = x
        levels[i] = level
    return levels


def question_record(qid: str, rows: list[dict[str, Any]], gold_status: str, cfg: dict[str, Any], c: float) -> dict[str, Any]:
    correct = [r for r in rows if r["screen"]["correct"] and candidate_cost(r["screen"]) is not None]
    costs = [candidate_cost(r["screen"]) for r in correct]
    levels = assign_levels(costs, c, cfg["level_ratio"], cfg["min_abs_steps"]) if costs else []
    n_levels = (max(levels) + 1) if levels else 0
    sigs = {(r["screen"].get("plan") or {}).get("plan_signature") for r in correct}
    if gold_status != "ok":
        reason = "gold_not_ok"
    elif not correct:
        reason = "no_correct"
    elif len(correct) == 1:
        reason = "one_correct"
    elif n_levels < cfg["min_levels"]:
        reason = "same_cost"
    else:
        reason = "ok"
    return {
        "qid": qid, "gold_status": gold_status, "n_candidates": len(rows), "n_correct": len(correct),
        "n_correct_generated": sum(1 for r in correct if r["cand"]["sources"] != ["gold"]),
        "n_levels": n_levels, "n_plan_signatures": len(sigs - {None}),
        "diverse": reason == "ok", "reason": reason,
        "min_cost": min(costs) if costs else None, "max_cost": max(costs) if costs else None,
        "levels": {r["cand"]["cand_id"]: lv for r, lv in zip(correct, levels)},
    }


def diversity(split: str, run: str, round_: int = 0, *, paths: Paths | None = None,
              pool_cfg: dict[str, Any] | None = None, progress_n: int | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = pool_cfg or load_config("pool.yaml")
    c = progress_n or load_config("evaluate.yaml")["exec"]["progress_n"]
    run_dir = paths.run_dir(run)
    cands = load_candidates(paths, run, split)
    screen = latest_by(run_dir / f"screen_{split}.jsonl", key="cand_id", where=is_current_screen)
    unscreened = [cid for cid, cd in cands.items() if cid not in screen]
    if unscreened:
        raise RuntimeError(f"{len(unscreened)} candidates not screened yet. Run `effsql screen` first.")
    by_q: dict[str, list[dict[str, Any]]] = {}
    gold_status: dict[str, str] = {}
    for cid, cd in cands.items():
        by_q.setdefault(cd["qid"], []).append({"cand": cd, "screen": screen[cid]})
        gold_status[cd["qid"]] = screen[cid]["gold_status"]
    recs = [question_record(q, rows, gold_status[q], cfg, c) for q, rows in sorted(by_q.items())]

    out = run_dir / f"diversity_{split}_r{round_}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    usable = [r for r in recs if r["gold_status"] == "ok"]
    summary = {
        "round": round_, "n_questions": len(recs), "n_gold_not_ok": len(recs) - len(usable),
        "n_diverse": sum(r["diverse"] for r in usable),
        "pct_diverse_of_usable": 100.0 * sum(r["diverse"] for r in usable) / max(len(usable), 1),
        "reasons": dict(Counter(r["reason"] for r in recs)),
        "mean_correct_per_question": sum(r["n_correct"] for r in usable) / max(len(usable), 1),
        "mean_levels": sum(r["n_levels"] for r in usable) / max(len(usable), 1),
    }
    spath = run_dir / f"diversity_summary_{split}.json"
    allsum = [s for s in (read_json(spath) if spath.exists() else []) if s["round"] != round_] + [summary]
    write_json(spath, sorted(allsum, key=lambda s: s["round"]))
    log.info("diversity %s round %d: %.1f%% of usable questions diverse; reasons %s",
             split, round_, summary["pct_diverse_of_usable"], summary["reasons"])
    return summary


def latest_round(paths: Paths, run: str, split: str) -> int:
    rounds = sorted(int(p.stem.rsplit("_r", 1)[1]) for p in paths.run_dir(run).glob(f"diversity_{split}_r*.jsonl"))
    if not rounds:
        raise RuntimeError("No diversity results. Run `effsql diversity` first.")
    return rounds[-1]
