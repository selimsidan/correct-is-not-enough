"""`effsql evaluate`: score a predictions file on dev or minidev.

Pipeline (each pass resumable, outputs in runs/<run>/):
  1. gold costs (cached per split): gold result hash, vm_steps, plan;
  2. correctness + vm_steps of each prediction, in parallel -> eval_<split>.jsonl;
  3. optional timing, strictly sequential, `repeat` times -> timing_<preset>_r<i>.jsonl
     (quick: gold/pred ratio of medians; official: exact R-VES port);
  4. metrics.json and one appended row in results/results.csv (the experiment ledger).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effsql.config import load_config, save_config
from effsql.data.bird import Example, load_examples, load_tied_gold
from effsql.exec.engine import batched, connect_readonly, execute_many
from effsql.io import append_jsonl, done_ids, iter_jsonl, write_json
from effsql.measure.bird_official import iterated_execute_sql
from effsql.measure.correctness import hashes_match
from effsql.measure.cost import rel_cost
from effsql.measure.gold import ensure_gold_costs, golds_for
from effsql.measure.metrics import compute_metrics
from effsql.measure.timing import time_pair_quick
from effsql.paths import Paths, load_paths
from effsql.runmeta import write_run_meta

log = logging.getLogger(__name__)
OFFICIAL_SEP = "\t----- bird -----\t"
TIMING_PRESETS = ("none", "quick", "official")
LEDGER_COLUMNS = [
    "timestamp", "run", "git_commit", "git_dirty", "model", "method", "split", "n", "timing", "repeats",
    "ex", "ex_tied", "rves_mean", "rves_std", "ves_mean", "ves_std", "quick_ratio_median",
    "rel_cost_median", "rel_cost_geomean", "faster_than_gold_rate", "eff_acc@1.5",
    "cpu_model", "cpu_count", "pred",
]


# Predictions -----------------------------------------------------------------------------------

def load_predictions(pred: str, examples: list[Example]) -> dict[str, str]:
    """qid -> SQL. Accepts `gold` (gold vs gold sanity check), our JSONL ({"qid", "sql"} per line)
    or the official JSON ({"0": "SQL\\t----- bird -----\\tdb_id", ...}), aligned by position as
    the official script does."""
    if pred == "gold":
        return {e.qid: e.gold_sql for e in examples}
    path = Path(pred)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        values = list(data.values())
        out = {}
        for e, v in zip(examples, values):
            if not isinstance(v, str):
                out[e.qid] = ""
                continue
            sql, _, db = v.partition(OFFICIAL_SEP)
            if db and db.strip() != e.db_id:
                raise ValueError(f"{path}: position of {e.qid} has db_id {db.strip()!r}, expected {e.db_id!r}")
            out[e.qid] = sql.strip()
        return out
    out = {}
    for rec in iter_jsonl(path):
        out[rec["qid"]] = rec.get("sql") or ""
    return out


# Passes ----------------------------------------------------------------------------------------

def correctness_pass(
    examples: list[Example], preds: dict[str, str], golds: dict[str, dict[str, Any]], out_path: Path,
    paths: Paths, split: str, exec_cfg: dict[str, Any], workers: int | None,
) -> None:
    done = done_ids(out_path)
    by_qid = {e.qid: e for e in examples}
    jobs = [
        {"key": e.qid, "db_path": paths.local_db_path(split, e.db_id), "sql": preds.get(e.qid, "")}
        for e in examples if e.qid not in done
    ]
    if not jobs:
        return
    log.info("correctness: %d to run (%d already done)", len(jobs), len(done))
    c = exec_cfg["progress_n"]
    stream = execute_many(
        jobs, paths, workers=workers, timeout_s=exec_cfg["timeout_s"],
        step_budget=exec_cfg.get("step_budget"), max_rows=exec_cfg["max_rows"], progress_n=c,
    )
    for batch in batched(stream, 50):
        recs = []
        for r in batch:
            e, g = by_qid[r["key"]], golds[r["key"]]
            tied = golds.get(f"{e.qid}#tied")
            correct = r["ok"] and g["ok"] and hashes_match(r["result_hash"], g["result_hash"])
            recs.append({
                "qid": e.qid, "db_id": e.db_id, "difficulty": e.difficulty,
                "pred_sql": preds.get(e.qid, ""),
                "correct": bool(correct),
                "correct_tied": (bool(r["ok"] and tied["ok"] and hashes_match(r["result_hash"], tied["result_hash"]))
                                 if tied else None),
                "error_type": r["error_type"], "error_msg": r["error_msg"],
                "n_rows": r["n_rows"], "vm_steps": r["vm_steps"], "gold_vm_steps": g["vm_steps"],
                "gold_status": g["gold_status"], "gold_elapsed_ms": g["elapsed_ms"],
                "rel_cost": rel_cost(r["vm_steps"], g["vm_steps"], c) if correct else None,
            })
        append_jsonl(out_path, recs)


def timing_pass(
    preset: str, examples: list[Example], eval_recs: dict[str, dict[str, Any]], out_path: Path,
    paths: Paths, split: str, timing_cfg: dict[str, Any],
) -> None:
    """Sequential timing of every question for one repeat."""
    done = done_ids(out_path)
    todo = [e for e in examples if e.qid not in done]
    if not todo:
        return
    cfg = timing_cfg[preset]
    if preset == "official":
        # The official script computes correctness itself; a prediction we already found wrong or
        # failing gets reward 0 there too, so only correct ones and EX timeouts (the official pair
        # budget is meta_time_out * iterate_num) are executed.
        run = [e for e in todo if eval_recs[e.qid]["correct"] or eval_recs[e.qid]["error_type"] == "timeout"]
        est = sum(2 * cfg["iterate_num"] * _gold_seconds(eval_recs[e.qid]) for e in run)
        log.info("official timing: %d questions to time, rough upper-bound ETA %.1f min", len(run), est / 60)
    t_start = time.time()
    conns: dict[str, Any] = {}
    try:
        for k, e in enumerate(sorted(todo, key=lambda x: x.db_id), 1):
            rec = eval_recs[e.qid]
            if preset == "quick":
                out: dict[str, Any] = {"qid": e.qid}
                if rec["correct"]:
                    if e.db_id not in conns:
                        conns[e.db_id] = connect_readonly(paths.local_db_path(split, e.db_id), paths)
                    out.update(time_pair_quick(
                        conns[e.db_id], e.gold_sql, rec["pred_sql"], warmup=cfg["warmup"], runs=cfg["runs"],
                        timeout_s=cfg["timeout_s"], cv_threshold=cfg["cv_threshold"],
                    ))
            else:
                if rec["correct"] or rec["error_type"] == "timeout":
                    res = iterated_execute_sql(
                        rec["pred_sql"], e.gold_sql, paths.local_db_path(split, e.db_id), paths,
                        iterate_num=cfg["iterate_num"], meta_time_out=cfg["meta_time_out"],
                    )
                    out = {"qid": e.qid, **res.record()}
                else:
                    out = {"qid": e.qid, "correct": False, "time_ratio": 0.0, "reward": 0,
                           "skipped": "incorrect in correctness pass"}
            append_jsonl(out_path, [out])
            if k % 25 == 0:
                log.info("%s timing: %d/%d (%.0f s)", preset, k, len(todo), time.time() - t_start)
    finally:
        for c in conns.values():
            c.close()


def _gold_seconds(rec: dict[str, Any]) -> float:
    """Per-execution gold time from the step-counting pass. The counter's callbacks slow the query,
    so this overestimates, which is the safe side for an ETA."""
    return (rec.get("gold_elapsed_ms") or 0.0) / 1000.0


# Ledger ----------------------------------------------------------------------------------------

def append_ledger(path: Path, row: dict[str, Any]) -> None:
    """Append one row; if the existing header differs (columns added later), rewrite with the union."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    header = list(LEDGER_COLUMNS)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header += [c for c in (reader.fieldnames or []) if c not in header]
    header += [c for c in row if c not in header]
    rows.append(row)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


# Entry point -----------------------------------------------------------------------------------

def evaluate(
    pred: str,
    split: str,
    run: str,
    timing: str = "none",
    repeat: int = 1,
    limit: int | None = None,
    workers: int | None = None,
    model: str | None = None,
    method: str | None = None,
    overwrite: bool = False,
    paths: Paths | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in ("dev", "minidev"):
        raise ValueError("evaluate is for dev and minidev only")
    if timing not in TIMING_PRESETS:
        raise ValueError(f"--timing must be one of {TIMING_PRESETS}")
    paths = paths or load_paths()
    cfg = cfg or load_config("evaluate.yaml")
    exec_cfg = cfg["exec"]
    workers = workers or exec_cfg.get("workers")

    examples = load_examples(split, paths)[:limit] if limit else load_examples(split, paths)
    missing_dbs = sorted({e.db_id for e in examples if not paths.local_db_path(split, e.db_id).exists()})
    if missing_dbs:
        raise RuntimeError(f"Databases not staged: {missing_dbs}. Run `effsql stage --split {split}`.")
    preds = load_predictions(pred, examples)
    n_missing = sum(1 for e in examples if e.qid not in preds)
    if n_missing:
        log.warning("%d questions have no prediction; they count as parse_fail", n_missing)

    run_dir = paths.run_dir(run)
    eval_path = run_dir / f"eval_{split}.jsonl"
    if overwrite:
        for p in [eval_path, *run_dir.glob(f"timing_*_{split}_r*.jsonl")]:
            p.unlink(missing_ok=True)
    save_config({**cfg, "args": {"pred": pred, "split": split, "timing": timing, "repeat": repeat,
                                 "limit": limit, "model": model, "method": method}},
                run_dir / f"evaluate_{split}_config.yaml")
    meta = write_run_meta(run_dir, data_files=[paths.split_file(split, "questions"), paths.split_file(split, "gold")],
                          extra={"bird_dev_release": paths.bird_dev_release, "split": split})

    tied = load_tied_gold(paths) if split == "dev" else None
    golds = ensure_gold_costs(split, golds_for(examples, tied), paths, exec_cfg, workers)
    correctness_pass(examples, preds, golds, eval_path, paths, split, exec_cfg, workers)
    eval_recs = {r["qid"]: r for r in iter_jsonl(eval_path)}
    recs = [eval_recs[e.qid] for e in examples]

    timing_runs: list[dict[str, dict[str, Any]]] = []
    if timing != "none":
        for i in range(repeat):
            tpath = run_dir / f"timing_{timing}_{split}_r{i}.jsonl"
            timing_pass(timing, examples, eval_recs, tpath, paths, split, cfg["timing"])
            timing_runs.append({r["qid"]: r for r in iter_jsonl(tpath)})

    metrics = compute_metrics(recs, timing_runs, timing, cfg.get("metrics", {}))
    metrics["n_gold_not_ok"] = sum(1 for r in recs if r["gold_status"] != "ok")
    write_json(run_dir / f"metrics_{split}_{timing}.json", metrics)

    o = metrics["overall"]
    append_ledger(paths.results_csv, {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run, "git_commit": meta["git"]["commit"], "git_dirty": meta["git"]["dirty"],
        "model": model, "method": method, "split": split, "n": o["n"], "timing": timing,
        "repeats": repeat if timing != "none" else 0,
        "ex": o["ex"], "ex_tied": o.get("ex_tied"),
        "rves_mean": o.get("rves_mean"), "rves_std": o.get("rves_std"),
        "ves_mean": o.get("ves_mean"), "ves_std": o.get("ves_std"),
        "quick_ratio_median": o.get("quick_ratio_median_mean"),
        "rel_cost_median": o["rel_cost_median"], "rel_cost_geomean": o["rel_cost_geomean"],
        "faster_than_gold_rate": o["faster_than_gold_rate"], "eff_acc@1.5": o.get("eff_acc@1.5"),
        "cpu_model": meta["machine"]["cpu_model"], "cpu_count": meta["machine"]["cpu_count"], "pred": pred,
    })
    return metrics
