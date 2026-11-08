"""Gold cost cache: gold result hash, vm_steps and plan per question, computed once per split.

File: <out_root>/cache/gold_costs/<split>.jsonl, one record per qid. A record is reused while the
gold SQL and db_id are unchanged (`gold_sha`); otherwise it is recomputed and appended (the last
record for a qid wins). Alternative tied golds are stored under the key `<qid>#tied`.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from effsql.data.bird import Example
from effsql.exec.engine import batched, execute_two_pass
from effsql.io import append_jsonl, iter_jsonl
from effsql.paths import Paths

log = logging.getLogger(__name__)
# 2: correctness and step counting in separate passes (2026-11-08); older records are recomputed.
CACHE_VERSION = 2
CHUNK = 500  # jobs per two-pass round (bounds work lost on a disconnect)


def gold_sha(db_id: str, sql: str) -> str:
    return hashlib.sha256(f"{db_id}\t{sql}".encode()).hexdigest()[:16]


def gold_status(rec: dict[str, Any]) -> str:
    """ok, empty (runs but returns no rows: any empty prediction would match), timeout or error."""
    if rec["ok"]:
        return "ok" if rec["n_rows"] > 0 else "empty"
    return "timeout" if rec["error_type"] == "timeout" else "error"


def gold_costs_path(paths: Paths, split: str) -> Path:
    return paths.cache_dir / "gold_costs" / f"{split}.jsonl"


def ensure_gold_costs(
    split: str,
    golds: dict[str, tuple[str, str]],
    paths: Paths,
    exec_cfg: dict[str, Any],
    workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """golds: key -> (db_id, sql). Returns key -> record, computing only what is missing or stale."""
    path = gold_costs_path(paths, split)
    cached = {r["key"]: r for r in iter_jsonl(path)}
    todo = [
        {"key": k, "db_path": paths.local_db_path(split, db), "sql": sql, "features": True,
         "db_id": db, "gold_sha": gold_sha(db, sql)}
        for k, (db, sql) in golds.items()
        if k not in cached
        or cached[k].get("gold_sha") != gold_sha(db, sql)
        or cached[k].get("progress_n") != exec_cfg["progress_n"]  # steps depend on N
        or cached[k].get("cache_version") != CACHE_VERSION
    ]
    if todo:
        log.info("gold costs %s: computing %d of %d", split, len(todo), len(golds))
        # Gold steps are never budget-limited: every ok gold needs a cost.
        cfg = {**exec_cfg, "step_budget": None}
        for chunk in batched(todo, CHUNK):
            results = execute_two_pass(chunk, paths, cfg, count_if=lambda r: r["ok"], workers=workers)
            recs = []
            for j in chunk:
                r = results[j["key"]]
                r.update(db_id=j["db_id"], gold_sha=j["gold_sha"], progress_n=exec_cfg["progress_n"],
                         cache_version=CACHE_VERSION)
                r["gold_status"] = gold_status(r)
                recs.append(r)
                cached[r["key"]] = r
            append_jsonl(path, recs)
        n_no_steps = sum(1 for j in todo if cached[j["key"]]["ok"] and cached[j["key"]]["vm_steps"] is None)
        if n_no_steps:
            log.warning("gold costs %s: %d ok golds have no vm_steps (counting pass failed)", split, n_no_steps)
    return {k: cached[k] for k in golds}


def golds_for(examples: list[Example], tied: dict[int, Any] | None = None) -> dict[str, tuple[str, str]]:
    """Gold map for examples, plus `<qid>#tied` entries for available tied golds."""
    out = {e.qid: (e.db_id, e.gold_sql) for e in examples}
    for e in examples:
        if tied and e.bird_question_id in tied:
            out[f"{e.qid}#tied"] = (e.db_id, tied[e.bird_question_id].sql)
    return out
