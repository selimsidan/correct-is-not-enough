"""`effsql quality`: find correct candidates that match gold only because SQLite picked a row arbitrarily.

"Correct" means the result set equals gold's on this database (official EX). Two SQL constructs let
SQLite pick rows arbitrarily, so a query that uses them can match gold by luck and be fast for the
wrong reason:

  scalar_subquery   a comparison with a subquery that can return several rows, e.g.
                    `WHERE id = (SELECT id FROM t WHERE name = 'x')`. SQLite silently uses the first
                    row the subquery returns and ignores the rest.
  unordered_limit   LIMIT without ORDER BY in the same SELECT: which rows are kept is unspecified.

Not flagged: subqueries that return one row by construction (aggregates without GROUP BY, such as
`(SELECT MAX(x) FROM t)`) and ordered picks (`ORDER BY ... LIMIT`), the BIRD convention for
"highest / lowest" questions that gold uses too.

Each flagged candidate is tested on its database with a deterministic variant of itself:
  `x = (subquery)`  -> `x IN (subquery)`       (every row instead of the first)
  `x <> (subquery)` -> `NOT x IN (subquery)`
  unordered LIMIT   -> removed
Verdicts:
  ok            nothing flagged (no execution needed)
  verified      the variant still returns gold's result: the arbitrary pick did not matter
                (the subquery returns one row, or all its rows lead to the same answer)
  fragile       the variant returns a different result: the match depended on the row SQLite
                happened to pick
  unverifiable  the construct cannot be rewritten (e.g. `x > (subquery)`), the variant fails, or
                sqlglot's re-rendering alone changes the result
Training never uses fragile or unverifiable candidates (train.data drops them). The variant is only
a test: candidates keep their original SQL.

Resumable by cand_id (quality_version). Needs the staged train databases.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.exec.engine import batched, execute_many
from effsql.io import append_jsonl, done_ids, latest_by, write_json
from effsql.measure.correctness import hashes_match
from effsql.measure.gold import ensure_gold_costs
from effsql.paths import Paths, load_paths
from effsql.pool.diversity import candidate_cost
from effsql.pool.screen import is_current_screen, load_candidates
from effsql.pool.summary import dpo_pair_counts

log = logging.getLogger(__name__)
logging.getLogger("sqlglot").setLevel(logging.ERROR)
QUALITY_VERSION = 1
CHUNK = 1000
BAD_VERDICTS = ("fragile", "unverifiable")
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def is_current_quality(rec: dict[str, Any]) -> bool:
    return rec.get("quality_version") == QUALITY_VERSION


# SQL analysis ---------------------------------------------------------------------------------
def _single_row(q: exp.Expression) -> bool:
    """True if the subquery returns at most one row by construction."""
    if not isinstance(q, exp.Select):
        return False
    if q.args.get("order") and q.args.get("limit"):
        return True  # ordered pick: deterministic up to ties, same convention as gold
    if q.args.get("group") or q.args.get("limit"):
        return False
    projs = q.expressions
    return bool(projs) and all(p.find(exp.AggFunc) is not None and p.find(exp.Window) is None for p in projs)


def _scalar_comparison(tree: exp.Expression) -> tuple[exp.Expression, exp.Subquery, exp.Expression] | None:
    """First comparison against a subquery that may return several rows: (node, subquery, other side)."""
    for node in tree.find_all(*_COMPARISONS):
        for side, other in ((node.expression, node.this), (node.this, node.expression)):
            if isinstance(side, exp.Subquery) and not _single_row(side.this):
                return node, side, other
    return None


def analyze(sql: str) -> dict[str, Any]:
    """Flags and the deterministic variant of one query. `variant` is None if nothing is flagged or
    a flagged construct cannot be rewritten (then `untransformable` names it)."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:  # noqa: BLE001 (sqlglot raises several error types)
        return {"flags": ["parse_error"], "roundtrip": None, "variant": None,
                "untransformable": f"sqlglot cannot parse: {type(e).__name__}"}
    roundtrip = tree.sql(dialect="sqlite")
    var = tree.copy()
    flags: list[str] = []
    untransformable = None
    for _ in range(100):  # one comparison per pass; rewritten nodes are no longer comparisons
        hit = _scalar_comparison(var)
        if hit is None:
            break
        node, sub, other = hit
        flags.append("scalar_subquery")
        if isinstance(node, exp.EQ):
            node.replace(exp.In(this=other.copy(), query=sub.copy()))
        elif isinstance(node, exp.NEQ):
            node.replace(exp.Not(this=exp.In(this=other.copy(), query=sub.copy())))
        else:
            untransformable = f"{type(node).__name__} comparison with a multi-row subquery"
            break
    for sel in list(var.find_all(exp.Select)):
        if sel.args.get("limit") and not sel.args.get("order"):
            flags.append("unordered_limit")
            sel.set("limit", None)
            sel.set("offset", None)
    if not flags:
        return {"flags": [], "roundtrip": roundtrip, "variant": None, "untransformable": None}
    variant = None if untransformable else var.sql(dialect="sqlite")
    return {"flags": sorted(set(flags)), "roundtrip": roundtrip, "variant": variant,
            "untransformable": untransformable}


def verdict(analysis: dict[str, Any], gold_hash: str | None, rt: dict[str, Any] | None,
            var: dict[str, Any] | None) -> tuple[str, str | None]:
    """(verdict, detail) from the analysis and the execution records of round trip and variant."""
    if not analysis["flags"]:
        return "ok", None
    if analysis["untransformable"]:
        return "unverifiable", analysis["untransformable"]
    if rt is None or not rt["ok"] or not hashes_match(rt["result_hash"], gold_hash):
        return "unverifiable", "re-rendering by sqlglot alone changes the result"
    if var is None:
        return "unverifiable", "variant not executed"
    if var["ok"]:
        if hashes_match(var["result_hash"], gold_hash):
            return "verified", None
        return "fragile", f"variant returns {var['n_rows']} rows with a different result"
    if var["error_type"] == "too_many_rows":
        return "fragile", "variant returns more rows than the limit"
    return "unverifiable", f"variant failed: {var['error_type']}"


# Run ------------------------------------------------------------------------------------------
def quality(split: str, run: str, *, paths: Paths | None = None, exec_cfg: dict[str, Any] | None = None,
            pool_cfg: dict[str, Any] | None = None, workers: int | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    exec_cfg = exec_cfg or load_config("evaluate.yaml")["exec"]
    pool_cfg = pool_cfg or load_config("pool.yaml")
    workers = workers or exec_cfg.get("workers")
    run_dir = paths.run_dir(run)
    cands = load_candidates(paths, run, split)
    screen = latest_by(run_dir / f"screen_{split}.jsonl", key="cand_id", where=is_current_screen)
    todo_all = [c for c in cands.values() if not c["is_gold"] and screen.get(c["cand_id"], {}).get("correct")]

    out = run_dir / f"quality_{split}.jsonl"
    done = done_ids(out, key="cand_id", where=is_current_quality)
    todo = [c for c in todo_all if c["cand_id"] not in done]
    examples = {e.qid: e for e in load_examples(split, paths)}

    analyses = {c["cand_id"]: analyze(c["sql_original"]) for c in todo}
    flagged = [c for c in todo if analyses[c["cand_id"]]["flags"]]
    log.info("quality %s: %d correct candidates, %d already checked, %d to check, %d flagged",
             split, len(todo_all), len(done), len(todo), len(flagged))

    unflagged = [c for c in todo if not analyses[c["cand_id"]]["flags"]]
    if unflagged:
        append_jsonl(out, [{"cand_id": c["cand_id"], "qid": c["qid"], "flags": [], "verdict": "ok", "detail": None,
                            "quality_version": QUALITY_VERSION} for c in unflagged])

    if flagged:
        need = sorted({examples[c["qid"]].db_id for c in flagged})
        missing = [d for d in need if not paths.local_db_path(split, d).exists()]
        if missing:
            raise RuntimeError(f"Databases not staged: {missing}. Run `effsql stage --split {split}`.")
        qids = sorted({c["qid"] for c in flagged})
        golds = ensure_gold_costs(split, {q: (examples[q].db_id, examples[q].gold_sql) for q in qids},
                                  paths, exec_cfg, workers)
        for chunk in batched(flagged, CHUNK):
            jobs = []
            for c in chunk:
                a = analyses[c["cand_id"]]
                db = paths.local_db_path(split, c["db_id"])
                if a["roundtrip"] is not None and not a["untransformable"]:
                    jobs.append({"key": f"{c['cand_id']}|rt", "db_path": db, "sql": a["roundtrip"]})
                    jobs.append({"key": f"{c['cand_id']}|var", "db_path": db, "sql": a["variant"]})
            res = {r["key"]: r for r in execute_many(jobs, paths, workers=workers, timeout_s=exec_cfg["timeout_s"],
                                                      max_rows=exec_cfg["max_rows"], count_steps=False)}
            recs = []
            for c in chunk:
                a = analyses[c["cand_id"]]
                v, detail = verdict(a, golds[c["qid"]].get("result_hash"), res.get(f"{c['cand_id']}|rt"),
                                    res.get(f"{c['cand_id']}|var"))
                recs.append({"cand_id": c["cand_id"], "qid": c["qid"], "flags": a["flags"], "verdict": v,
                             "detail": detail, "variant_sql": a["variant"], "quality_version": QUALITY_VERSION})
            append_jsonl(out, recs)
            log.info("quality %s: %d flagged candidates checked", split, len(recs))

    stats = quality_stats(paths, run, split, pool_cfg, load_config("evaluate.yaml")["exec"]["progress_n"])
    write_json(run_dir / f"quality_stats_{split}.json", stats)
    return stats


def quality_stats(paths: Paths, run: str, split: str, pool_cfg: dict[str, Any], c: float) -> dict[str, Any]:
    """Verdict counts and how much the filter changes the efficiency picture (questions where a model
    query beats gold, efficiency pairs at 1.5x and 2x), before and after dropping bad verdicts."""
    run_dir = paths.run_dir(run)
    cands = load_candidates(paths, run, split)
    screen = latest_by(run_dir / f"screen_{split}.jsonl", key="cand_id", where=is_current_screen)
    q = latest_by(run_dir / f"quality_{split}.jsonl", key="cand_id", where=is_current_quality)
    rows = []
    for cid, cd in cands.items():
        s = screen.get(cid)
        if s is None or s["gold_status"] != "ok":
            continue
        cost = candidate_cost(s) if s["correct"] else None
        rows.append({"qid": cd["qid"], "cand_id": cid, "is_gold": cd["is_gold"], "correct": bool(s["correct"]),
                     "error_type": s.get("error_type"),
                     "vm_steps": s.get("vm_steps") if not s.get("steps_capped") else None,
                     "cost": cost, "gold": s.get("gold_vm_steps"),
                     "verdict": "gold" if cd["is_gold"] else q.get(cid, {}).get("verdict")})
    df = pd.DataFrame(rows)
    n_q = int(df["qid"].nunique())
    model_correct = df[~df["is_gold"] & df["correct"]]
    ratio, min_abs = pool_cfg["level_ratio"], pool_cfg["min_abs_steps"]

    def beats(d: pd.DataFrame) -> int:
        d = d[d["cost"].notna() & d["gold"].notna()]
        hit = ((d["gold"] + c) >= ratio * (d["cost"] + c)) & (d["gold"] - d["cost"] >= min_abs)
        return int(d[hit]["qid"].nunique())

    kept = df[~df["verdict"].isin(BAD_VERDICTS)]
    pairs = {f"x{r}": {"before": dpo_pair_counts(df, r, min_abs, c)["efficiency_pairs"],
                       "after": dpo_pair_counts(kept, r, min_abs, c)["efficiency_pairs"]} for r in (1.5, 2.0)}
    flags = Counter(f for cid in model_correct["cand_id"] for f in q.get(cid, {}).get("flags", []))
    return {
        "n_correct_model_candidates": len(model_correct),
        "n_checked": int(model_correct["verdict"].notna().sum()),
        "verdicts": model_correct["verdict"].fillna("not_checked").value_counts().to_dict(),
        "flags": dict(flags),
        "pct_dropped": round(100 * model_correct["verdict"].isin(BAD_VERDICTS).mean(), 2) if len(model_correct) else 0.0,
        "questions": n_q,
        "questions_beating_gold": {"before": beats(model_correct),
                                   "after": beats(model_correct[~model_correct["verdict"].isin(BAD_VERDICTS)])},
        "efficiency_pairs": pairs,
        "quality_version": QUALITY_VERSION,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m effsql.pool.quality", description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="train")
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    print(json.dumps(quality(args.split, args.run, workers=args.workers), indent=2, default=str))


if __name__ == "__main__":
    main()
