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
from effsql.exec.engine import batched, execute_many
from effsql.io import append_jsonl, iter_jsonl
from effsql.paths import Paths

log = logging.getLogger(__name__)


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
    ]
    if todo:
        log.info("gold costs %s: computing %d of %d", split, len(todo), len(golds))
        meta = {j["key"]: (j["db_id"], j["gold_sha"]) for j in todo}
        stream = execute_many(
            todo, paths, workers=workers,
            timeout_s=exec_cfg["timeout_s"], step_budget=None,
            max_rows=exec_cfg["max_rows"], progress_n=exec_cfg["progress_n"],
        )
        for batch in batched(stream, 50):
            recs = []
            for r in batch:
                db_id, sha = meta[r["key"]]
                r.update(db_id=db_id, gold_sha=sha, progress_n=exec_cfg["progress_n"])
                r["gold_status"] = gold_status(r)
                recs.append(r)
                cached[r["key"]] = r
            append_jsonl(path, recs)
    return {k: cached[k] for k in golds}


def golds_for(examples: list[Example], tied: dict[int, Any] | None = None) -> dict[str, tuple[str, str]]:
    """Gold map for examples, plus `<qid>#tied` entries for available tied golds."""
    out = {e.qid: (e.db_id, e.gold_sql) for e in examples}
    for e in examples:
        if tied and e.bird_question_id in tied:
            out[f"{e.qid}#tied"] = (e.db_id, tied[e.bird_question_id].sql)
    return out
