"""`effsql summary`: one report for a pipeline run (written as JSON and printed).

Combines generation, parse, screening, diversity (per round), pool, verification and the
evaluation runs named <run>_<split>_<prompt>.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import pandas as pd

from effsql.config import load_config
from effsql.io import latest_by, read_json, write_json
from effsql.paths import Paths, load_paths
from effsql.gen.sample import is_current_gen


def _maybe(path):
    return read_json(path) if path.exists() else None


def dpo_pair_counts(pool: pd.DataFrame, level_ratio: float, min_abs: float, c: float) -> dict[str, int]:
    """Potential preference pairs per question: (cheaper correct, pricier correct) at different levels,
    and (correct, incorrect)."""
    eff = corr_inc = 0
    for _, g in pool.groupby("qid"):
        ok = g[g["correct"] & g["vm_steps"].notna()]["vm_steps"].tolist()
        for a, b in combinations(sorted(ok), 2):
            if (b + c) >= level_ratio * (a + c) and b - a >= min_abs:
                eff += 1
        corr_inc += int(g["correct"].sum()) * int((~g["correct"] & (g["error_type"] != "parse_fail")).sum())
    return {"efficiency_pairs": eff, "correct_vs_incorrect_pairs": corr_inc}


def summarize(split: str, run: str, *, paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    pool_cfg = load_config("pool.yaml")
    c = load_config("evaluate.yaml")["exec"]["progress_n"]
    run_dir = paths.run_dir(run)
    out: dict[str, Any] = {"run": run, "split": split}

    gens = list(latest_by(run_dir / f"generations_{split}.jsonl", key="key", where=is_current_gen).values())
    if gens:
        out["generation"] = {
            "stats": _maybe(run_dir / f"gen_stats_{split}.json"),
            "prompt_levels": dict(Counter(g["prompt_level"] for g in gens)),
            "finish_reasons": dict(Counter(c_["finish_reason"] for g in gens for c_ in g["completions"])),
        }
    out["parse"] = _maybe(run_dir / f"parse_stats_{split}.json")
    out["diversity_rounds"] = _maybe(run_dir / f"diversity_summary_{split}.json")

    pool_path = run_dir / f"pool_{split}.parquet"
    if pool_path.exists():
        pool = pd.read_parquet(pool_path)
        exe = pool[pool["error_type"] != "parse_fail"]
        by_source = exe.assign(src=exe["sources"].str.split(";")).explode("src")
        out["pool"] = {
            "n_candidates": len(pool), "n_questions": int(pool["qid"].nunique()),
            "candidates_per_question_by_source": (by_source.groupby("src")["cand_id"].count()
                                                  / pool["qid"].nunique()).round(3).to_dict(),
            "pct_correct_by_source": (100 * by_source.groupby("src")["correct"].mean()).round(1).to_dict(),
            "error_types": exe["error_type"].fillna("none").value_counts().to_dict(),
            "n_steps_capped": int(pool["steps_capped"].sum()),
            "eff_score_quantiles": (pool["eff_score"].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                                    .round(3).to_dict() if pool["eff_score"].notna().any() else None),
            "n_correct_cheaper_than_gold_x1_5": int((pool["rel_cost"] <= 1 / 1.5).sum()),
            "potential_pairs": dpo_pair_counts(pool, pool_cfg["level_ratio"], pool_cfg["min_abs_steps"], c),
        }
    out["verify"] = _maybe(run_dir / f"verify_stats_{split}.json")
    evals = {}
    for p in sorted(paths.runs_dir.glob(f"{run}_*")):
        for m in sorted(p.glob("metrics_*.json")):
            o = read_json(m)["overall"]
            evals[f"{p.name}/{m.stem}"] = {k: o.get(k) for k in ("n", "ex", "ex_tied", "rves_mean", "rves_std",
                                                                  "ves_mean", "quick_ratio_median_mean",
                                                                  "rel_cost_median", "eff_acc@1.5", "outcomes")}
    out["evaluations"] = evals
    write_json(run_dir / f"summary_{split}.json", out)
    return out
