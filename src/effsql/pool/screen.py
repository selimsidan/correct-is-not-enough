"""`effsql screen`: execute every unique candidate (correctness, cost, plan).

Uses the two-pass engine: correctness at native speed under the official 30 s limit, then
vm_steps for correct candidates only. Counting stops at candidate_step_cap_factor x
max(gold_steps, 1000); such candidates are recorded as censored (steps_lower_bound) instead of
being counted to the end. Candidates of questions whose gold is not ok are not executed.
Resumable by cand_id (screen_version).
"""

from __future__ import annotations

import logging
from typing import Any

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.exec.engine import batched, execute_two_pass
from effsql.io import append_jsonl, done_ids, latest_by
from effsql.measure.correctness import hashes_match
from effsql.measure.gold import ensure_gold_costs
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
SCREEN_VERSION = 1
CHUNK = 1000


def is_current_screen(rec: dict[str, Any]) -> bool:
    return rec.get("screen_version") == SCREEN_VERSION


def load_candidates(paths: Paths, run: str, split: str) -> dict[str, dict[str, Any]]:
    path = paths.run_dir(run) / f"candidates_{split}.jsonl"
    if not path.exists():
        raise RuntimeError(f"{path} not found. Run `effsql parse` first.")
    return latest_by(path, key="cand_id")


def screen(
    split: str, run: str, *, paths: Paths | None = None, exec_cfg: dict[str, Any] | None = None,
    pool_cfg: dict[str, Any] | None = None, workers: int | None = None,
) -> dict[str, Any]:
    paths = paths or load_paths()
    exec_cfg = exec_cfg or load_config("evaluate.yaml")["exec"]
    pool_cfg = pool_cfg or load_config("pool.yaml")
    workers = workers or exec_cfg.get("workers")
    cands = load_candidates(paths, run, split)
    examples = {e.qid: e for e in load_examples(split, paths)}
    qids = sorted({c["qid"] for c in cands.values()})
    missing_dbs = sorted({examples[q].db_id for q in qids if not paths.local_db_path(split, examples[q].db_id).exists()})
    if missing_dbs:
        raise RuntimeError(f"Databases not staged: {missing_dbs}. Run `effsql stage --split {split}`.")

    golds = ensure_gold_costs(split, {q: (examples[q].db_id, examples[q].gold_sql) for q in qids}, paths, exec_cfg, workers)
    out = paths.run_dir(run) / f"screen_{split}.jsonl"
    done = done_ids(out, key="cand_id", where=is_current_screen)
    cap_factor = pool_cfg.get("candidate_step_cap_factor", 50)

    skipped, jobs = [], []
    for c in cands.values():
        if c["cand_id"] in done:
            continue
        g = golds[c["qid"]]
        base = {"cand_id": c["cand_id"], "qid": c["qid"], "gold_status": g["gold_status"],
                "steps_capped": False, "steps_lower_bound": None, "gold_vm_steps": g.get("vm_steps"),
                "screen_version": SCREEN_VERSION}
        if c["sql_original"] is None:
            skipped.append({**base, "correct": False, "error_type": "parse_fail", "vm_steps": None})
        elif g["gold_status"] != "ok":
            skipped.append({**base, "correct": False, "error_type": "gold_not_ok", "vm_steps": None})
        else:
            cap = int(cap_factor * max(g.get("vm_steps") or 0, 1000))
            jobs.append({"key": c["cand_id"], "db_path": paths.local_db_path(split, c["db_id"]),
                         "sql": c["sql_original"], "features": True, "count_exec": {"step_budget": cap},
                         "qid": c["qid"], "cap": cap})
    if skipped:
        append_jsonl(out, skipped)
    log.info("screen %s: %d candidates to execute, %d not executable, %d already done",
             split, len(jobs), len(skipped), len(done))

    def is_correct(r: dict[str, Any]) -> bool:
        g = golds[cands[r["key"]]["qid"]]
        return bool(r["ok"] and hashes_match(r["result_hash"], g["result_hash"]))

    for chunk in batched(jobs, CHUNK):
        res = execute_two_pass(chunk, paths, exec_cfg, count_if=is_correct, workers=workers)
        recs = []
        for j in chunk:
            r = res[j["key"]]
            g = golds[j["qid"]]
            recs.append({
                "cand_id": j["key"], "qid": j["qid"], "gold_status": g["gold_status"],
                "correct": is_correct(r), "error_type": r["error_type"], "error_msg": r["error_msg"],
                "n_rows": r["n_rows"], "elapsed_ms": r["elapsed_ms"],
                "vm_steps": r.get("vm_steps"), "steps_error": r.get("steps_error"),
                "steps_capped": r.get("steps_error") == "step_budget",
                "steps_lower_bound": r.get("steps_lower_bound"), "step_cap": j["cap"],
                "gold_vm_steps": g.get("vm_steps"),
                "plan": r.get("plan"), "text": r.get("text"), "screen_version": SCREEN_VERSION,
            })
        append_jsonl(out, recs)
        log.info("screen %s: %d candidates done", split, len(recs))

    recs = latest_by(out, key="cand_id", where=is_current_screen)
    current = [recs[c] for c in cands if c in recs]
    return {"n_candidates": len(current), "n_correct": sum(r["correct"] for r in current),
            "n_capped": sum(bool(r.get("steps_capped")) for r in current)}
