"""`effsql build-pool`: one table, one row per unique candidate, for the later training phases.

eff_score = log2((gold_steps + c) / (steps + c)) for correct candidates (positive = cheaper than
gold), from measure.cost.eff_score. Censored candidates (counting stopped at the cap) use their
lower bound, so their eff_score is an upper bound; they are marked steps_capped.
eff_rank: 1 = cheapest correct candidate of the question; eff_percentile: 1.0 = cheapest,
0.0 = most expensive (1.0 when the question has a single correct candidate).
Questions whose gold is not ok are dropped unless keep_bad_gold.
quality: verdict of `effsql.pool.quality` for correct model candidates (ok, verified, fragile,
unverifiable; None if the check has not run), "gold" for gold rows, None for incorrect rows.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from effsql.config import load_config
from effsql.io import latest_by, read_jsonl
from effsql.measure.cost import eff_score, rel_cost
from effsql.paths import Paths, load_paths
from effsql.pool.diversity import candidate_cost, latest_round
from effsql.pool.quality import is_current_quality
from effsql.pool.screen import is_current_screen, load_candidates

log = logging.getLogger(__name__)
PLAN_KEYS = ("n_scan", "n_search", "n_temp_btree", "n_correlated", "n_subquery", "n_automatic_index",
             "n_covering_index")
TEXT_KEYS = ("n_joins", "n_selects", "nesting_depth", "n_ctes", "has_window", "n_tokens")


def build_pool(split: str, run: str, *, paths: Paths | None = None, pool_cfg: dict[str, Any] | None = None,
               progress_n: int | None = None) -> pd.DataFrame:
    paths = paths or load_paths()
    cfg = pool_cfg or load_config("pool.yaml")
    c = progress_n or load_config("evaluate.yaml")["exec"]["progress_n"]
    run_dir = paths.run_dir(run)
    cands = load_candidates(paths, run, split)
    screen = latest_by(run_dir / f"screen_{split}.jsonl", key="cand_id", where=is_current_screen)
    div = {r["qid"]: r for r in read_jsonl(run_dir / f"diversity_{split}_r{latest_round(paths, run, split)}.jsonl")}
    qpath = run_dir / f"quality_{split}.jsonl"
    qual = latest_by(qpath, key="cand_id", where=is_current_quality) if qpath.exists() else {}

    rows = []
    for cid, cd in cands.items():
        s = screen.get(cid)
        if s is None:
            raise RuntimeError(f"Candidate {cid} not screened. Run `effsql screen` first.")
        if s["gold_status"] != "ok" and not cfg.get("keep_bad_gold", False):
            continue
        cost, gold_steps = candidate_cost(s), s.get("gold_vm_steps")
        has_cost = s["correct"] and cost is not None and gold_steps is not None
        plan, text = s.get("plan") or {}, s.get("text") or {}
        q = qual.get(cid, {})
        if cd["is_gold"]:
            quality, qflags = "gold", None
        elif s["correct"]:
            quality, qflags = q.get("verdict"), ";".join(q.get("flags") or []) or None
        else:
            quality, qflags = None, None
        rows.append({
            "qid": cd["qid"], "split": split, "db_id": cd["db_id"], "difficulty": cd["difficulty"],
            "cand_id": cid, "source": cd["first_source"], "sources": ";".join(cd["sources"]),
            "is_gold": cd["is_gold"], "model": "gold" if cd["first_source"] == "gold" else cd.get("model"),
            "prompt_variant": cd["first_variant"], "variants": ";".join(cd["variants"]),
            "n_occurrences": cd["n_occurrences"],
            "sql_original": cd["sql_original"], "sql_canonical": cd["sql_canonical"],
            "canonical_ok": cd["canonical_ok"],
            "correct": bool(s["correct"]), "error_type": s.get("error_type"), "gold_status": s["gold_status"],
            "vm_steps": s.get("vm_steps"), "steps_capped": bool(s.get("steps_capped")),
            "steps_lower_bound": s.get("steps_lower_bound"), "gold_vm_steps": gold_steps,
            "rel_cost": rel_cost(cost, gold_steps, c) if has_cost else None,
            "eff_score": eff_score(cost, gold_steps, c) if has_cost else None,
            "_cost": cost if has_cost else None,
            "efficiency_level": div.get(cd["qid"], {}).get("levels", {}).get(cid),
            "question_diverse": div.get(cd["qid"], {}).get("diverse"),
            "quality": quality, "quality_flags": qflags,
            "plan_signature": plan.get("plan_signature"),
            **{f"plan_{k}": plan.get(k) for k in PLAN_KEYS},
            **{f"text_{k}": text.get(k) for k in TEXT_KEYS},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Pool is empty")
    has = df["_cost"].notna()
    df["eff_rank"] = df[has].groupby("qid")["_cost"].rank(method="min").astype("Int64")
    n = df[has].groupby("qid")["_cost"].transform("count")
    df.loc[has, "eff_percentile"] = (1.0 - (df.loc[has, "eff_rank"].astype(float) - 1) / (n - 1).where(n > 1, 1)).where(n > 1, 1.0)
    df["n_correct_in_question"] = df.groupby("qid")["correct"].transform("sum")
    df = df.drop(columns="_cost").sort_values(["qid", "eff_rank", "cand_id"], na_position="last")

    df.to_parquet(run_dir / f"pool_{split}.parquet", index=False)
    with open(run_dir / f"pool_{split}.jsonl", "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps({k: (None if pd.isna(v) else v) if not isinstance(v, (list, dict)) else v
                                for k, v in rec.items()}, default=str) + "\n")
    log.info("pool %s: %d candidates over %d questions (%d correct)", split, len(df), df["qid"].nunique(),
             int(df["correct"].sum()))
    return df
