"""JSONL helpers used by every long-running command so that it can resume after a disconnect."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger(__name__)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records and flush to disk, so a crash loses at most the record being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"  # previous write was cut off mid-line
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records; a truncated last line (from an interrupted write) is skipped with a warning."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping unreadable line %d in %s (interrupted write?)", lineno, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def done_ids(path: Path, key: str = "qid", where: Callable[[dict[str, Any]], bool] | None = None) -> set[Any]:
    """IDs already present in an output file; commands skip these unless --overwrite is given.
    `where` restricts to records that still count as done (e.g. the current measurement version)."""
    return {rec[key] for rec in iter_jsonl(path) if key in rec and (where is None or where(rec))}


def latest_by(path: Path, key: str = "qid", where: Callable[[dict[str, Any]], bool] | None = None) -> dict[Any, dict[str, Any]]:
    """Last record per key among those matching `where` (later appends supersede earlier ones)."""
    out: dict[Any, dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        if key in rec and (where is None or where(rec)):
            out[rec[key]] = rec
    return out


def write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (temp file then rename) so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
