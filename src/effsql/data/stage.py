"""Stage BIRD databases from the zip on Drive to the local Colab disk.

Members are extracted one by one straight from the zip on the Drive mount (no full copy of the
zip), so a train subset never touches the other 9 GB. Staging is idempotent: a database whose
files already exist with the right sizes is skipped, so every notebook can call it at start.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effsql.config import load_config
from effsql.data.bird import read_questions
from effsql.data.zipindex import DbMembers, index_zip
from effsql.exec.engine import connect_readonly
from effsql.io import read_json, write_json
from effsql.paths import Paths, check_db_id, load_paths

log = logging.getLogger(__name__)
GB = 1024**3


def parse_shard(shard: str) -> tuple[int, int]:
    """Parse `i/n` (0-based i)."""
    try:
        i, n = (int(x) for x in shard.split("/"))
    except ValueError as e:
        raise ValueError(f"--shard must look like i/n, got {shard!r}") from e
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"--shard {shard!r}: need 0 <= i < n")
    return i, n


def select_db_ids(
    available: list[str],
    db_ids: list[str] | None = None,
    shard: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Pick databases to work on. Shards are round-robin over the sorted ids, so every command
    given the same `--shard i/n` selects the same databases."""
    ids = sorted(set(available))
    if db_ids:
        missing = sorted(set(db_ids) - set(ids))
        if missing:
            raise ValueError(f"Unknown db_ids: {missing}")
        ids = sorted(set(db_ids))
    if shard:
        i, n = parse_shard(shard)
        ids = ids[i::n]
    if limit is not None:
        ids = ids[:limit]
    return ids


def question_db_ids(split: str, paths: Paths) -> list[str]:
    return sorted({r["db_id"] for r in read_questions(paths.split_file(split, "questions"))})


def _desc_dest(db_dir: Path, info: zipfile.ZipInfo) -> Path:
    return db_dir / "database_description" / Path(info.filename).name


def _is_staged(db_dir: Path, db_id: str, members: DbMembers) -> bool:
    sqlite = db_dir / f"{db_id}.sqlite"
    if not sqlite.exists() or sqlite.stat().st_size != members.sqlite.file_size:
        return False
    for info in members.descriptions:
        dest = _desc_dest(db_dir, info)
        if not dest.exists() or dest.stat().st_size != info.file_size:
            return False
    return True


def _extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path, bufsize: int) -> None:
    """Copy one member to dest via a temp file, so an interrupted copy never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with zf.open(info) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, bufsize)
    os.replace(tmp, dest)


def _validate(db_path: Path, paths: Paths) -> int:
    """Open read-only and touch sqlite_master; returns the number of tables."""
    conn = connect_readonly(db_path, paths)
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    finally:
        conn.close()


def _free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def stage(
    split: str,
    db_ids: list[str] | None = None,
    shard: str | None = None,
    limit: int | None = None,
    paths: Paths | None = None,
    data_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage databases for `split` and return the updated manifest."""
    paths = paths or load_paths()
    data_cfg = data_cfg or load_config("data.yaml")
    stage_cfg = data_cfg.get("stage", {})
    bufsize = int(stage_cfg.get("copy_buffer_mb", 16)) * 1024 * 1024
    reserve = float(stage_cfg.get("disk_reserve_gb", 5)) * GB

    db_split = paths.db_split(split)
    zip_path = paths.db_zip(split)
    out_dir = paths.local_split_dir(split)
    log.info("Indexing %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        index = index_zip(zf)
        if not index.dbs and index.nested_zips:
            raise RuntimeError(f"{zip_path} holds nested zips {index.nested_zips[:3]}; not supported yet")
        incomplete = sorted(d for d, m in index.dbs.items() if m.sqlite is None)
        available = sorted(d for d, m in index.dbs.items() if m.sqlite is not None)

        if split != db_split:
            # e.g. minidev: stage exactly the databases its questions use, all of which must be in
            # the zip of the split it runs on.
            needed = question_db_ids(split, paths)
            missing = sorted(set(needed) - set(available))
            if missing:
                raise RuntimeError(f"{split} uses db_ids not in {zip_path.name}: {missing}")
            available = needed
        selected = select_db_ids(available, db_ids, shard, limit)
        if incomplete:
            log.warning("Zip has description folders without a .sqlite for: %s", incomplete)

        todo = [d for d in selected if not _is_staged(out_dir / d, d, index.dbs[d])]
        need = sum(index.dbs[d].uncompressed_bytes for d in todo)
        free = _free_bytes(out_dir)
        log.info(
            "%s: %d selected, %d already staged, %d to extract (%.2f GB); %.2f GB free locally",
            split, len(selected), len(selected) - len(todo), len(todo), need / GB, free / GB,
        )
        if need + reserve > free:
            raise RuntimeError(
                f"Not enough local disk: need {need / GB:.2f} GB plus {reserve / GB:.1f} GB reserve, "
                f"only {free / GB:.2f} GB free. Stage fewer databases with --db-ids or --shard."
            )

        for k, db_id in enumerate(todo, 1):
            check_db_id(db_id)
            members = index.dbs[db_id]
            t0 = time.time()
            db_dir = out_dir / db_id
            _extract(zf, members.sqlite, db_dir / f"{db_id}.sqlite", bufsize)
            for info in members.descriptions:
                _extract(zf, info, _desc_dest(db_dir, info), bufsize)
            log.info(
                "[%d/%d] staged %s (%.2f GB) in %.0f s",
                k, len(todo), db_id, members.uncompressed_bytes / GB, time.time() - t0,
            )

    manifest_path = paths.manifest_path(split)
    manifest = read_json(manifest_path) if manifest_path.exists() else {"split": db_split, "dbs": {}}
    failures: dict[str, str] = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for db_id in selected:
        db_path = out_dir / db_id / f"{db_id}.sqlite"
        try:
            n_tables = _validate(db_path, paths)
        except Exception as e:  # noqa: BLE001 (report every failing database, then raise)
            failures[db_id] = f"{type(e).__name__}: {e}"
            manifest["dbs"].pop(db_id, None)
            continue
        prev = manifest["dbs"].get(db_id, {})
        manifest["dbs"][db_id] = {
            "db_id": db_id,
            "path": str(db_path),
            "size_bytes": db_path.stat().st_size,
            "n_tables": n_tables,
            "n_description_files": len(index.dbs[db_id].descriptions),
            "source_zip": str(zip_path),
            "staged_at": now if db_id in todo else prev.get("staged_at", now),
        }
    write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"Validation failed for {len(failures)} database(s): {failures}")
    log.info("Manifest: %s (%d databases)", manifest_path, len(manifest["dbs"]))
    return manifest


def staged_db_ids(split: str, paths: Paths | None = None) -> list[str]:
    """Databases recorded in the local manifest for the split's database folder."""
    paths = paths or load_paths()
    mp = paths.manifest_path(split)
    return sorted(read_json(mp)["dbs"]) if mp.exists() else []
