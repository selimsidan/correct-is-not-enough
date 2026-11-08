"""Per-database schema cache used to build prompts.

One JSON per db_id under <out_root>/cache/schemas/<db_split>/ holding: CREATE TABLE text, column
info, row counts, a few truncated sample rows, BIRD column descriptions from
database_description/*.csv, and primary/foreign keys from both PRAGMA and *_tables.json.
Built on CPU once; GPU generation reads only this cache and never needs staged databases.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effsql.config import load_config
from effsql.data.bird import read_questions
from effsql.data.stage import select_db_ids, staged_db_ids
from effsql.exec.engine import connect_readonly
from effsql.io import read_json, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 never fails, so it is the last resort


def quote_ident(name: str) -> str:
    """Double-quote an identifier (BIRD has spaces, parentheses and keywords in names)."""
    return '"' + name.replace('"', '""') + '"'


def truncate_value(v: Any, max_chars: int) -> Any:
    if isinstance(v, bytes):
        return f"<blob {len(v)} bytes>"
    if isinstance(v, str) and len(v) > max_chars:
        return v[:max_chars] + "..."
    return v


def read_description_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read one BIRD description CSV, trying encodings in order. Header keys are normalized
    (stripped, lowercased, spaces to underscores); empty rows are dropped."""
    raw = path.read_bytes()
    for enc in CSV_ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        clean = {
            (k or "").strip().lower().replace(" ", "_"): (v or "").strip()
            for k, v in row.items()
            if k is not None
        }
        if any(clean.values()):
            rows.append(clean)
    return rows, enc


def tables_json_keys(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Primary and foreign keys from a Spider-style *_tables.json entry, as `table.column` strings."""
    if not entry:
        return {"primary_keys": [], "foreign_keys": []}
    tables = entry["table_names_original"]
    cols = entry["column_names_original"]

    def name(ci: int) -> str:
        ti, cname = cols[ci]
        return f"{tables[ti]}.{cname}"

    pks = []
    for pk in entry.get("primary_keys", []):
        pks.append([name(c) for c in pk] if isinstance(pk, list) else name(pk))
    fks = [{"from": name(a), "to": name(b)} for a, b in entry.get("foreign_keys", [])]
    return {"primary_keys": pks, "foreign_keys": fks}


def build_schema(
    db_path: Path,
    paths: Paths,
    tables_entry: dict[str, Any] | None = None,
    n_sample_rows: int = 3,
    max_value_chars: int = 80,
) -> dict[str, Any]:
    desc_dir = db_path.parent / "database_description"
    desc_files = {p.stem.lower(): p for p in desc_dir.glob("*.csv")} if desc_dir.exists() else {}
    used_desc: set[str] = set()

    conn = connect_readonly(db_path, paths)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY rowid"
        ).fetchall()
        tables = []
        pragma_fks = []
        for tname, ddl in rows:
            q = quote_ident(tname)
            columns = [
                {"name": c[1], "type": c[2], "notnull": bool(c[3]), "default": c[4], "pk": c[5]}
                for c in conn.execute(f"PRAGMA table_info({q})")
            ]
            for fk in conn.execute(f"PRAGMA foreign_key_list({q})"):
                pragma_fks.append({"from": f"{tname}.{fk[3]}", "to": f"{fk[2]}.{fk[4]}"})
            row_count = conn.execute(f"SELECT count(*) FROM {q}").fetchone()[0]
            cur = conn.execute(f"SELECT * FROM {q} LIMIT ?", (n_sample_rows,))
            sample_cols = [d[0] for d in cur.description]
            sample = [[truncate_value(v, max_value_chars) for v in r] for r in cur.fetchall()]

            table: dict[str, Any] = {
                "name": tname,
                "ddl": ddl,
                "columns": columns,
                "row_count": row_count,
                "sample_columns": sample_cols,
                "sample_rows": sample,
                "column_descriptions": [],
                "description_file": None,
                "description_encoding": None,
            }
            desc = desc_files.get(tname.lower())
            if desc is not None:
                table["column_descriptions"], table["description_encoding"] = read_description_csv(desc)
                table["description_file"] = desc.name
                used_desc.add(tname.lower())
            tables.append(table)
    finally:
        conn.close()

    return {
        "db_id": db_path.stem,
        "sqlite_size_bytes": db_path.stat().st_size,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sample_rows": n_sample_rows,
        "max_value_chars": max_value_chars,
        "tables": tables,
        "foreign_keys_pragma": pragma_fks,
        "tables_json": tables_json_keys(tables_entry),
        "unmatched_description_files": sorted(
            p.name for k, p in desc_files.items() if k not in used_desc
        ),
    }


def build_schema_cache(
    split: str,
    db_ids: list[str] | None = None,
    shard: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    paths: Paths | None = None,
    data_cfg: dict[str, Any] | None = None,
) -> list[Path]:
    """Write schema JSONs for staged databases; existing files are skipped unless overwrite."""
    paths = paths or load_paths()
    cfg = (data_cfg or load_config("data.yaml")).get("schema", {})
    db_split = paths.db_split(split)

    staged = staged_db_ids(split, paths)
    if split != db_split:
        needed = sorted({r["db_id"] for r in read_questions(paths.split_file(split, "questions"))})
        missing = sorted(set(needed) - set(staged))
        if missing:
            raise RuntimeError(f"Not staged yet: {missing}. Run `effsql stage --split {split}`.")
        staged = needed
    if not staged:
        raise RuntimeError(f"No staged databases for {split}. Run `effsql stage --split {split}` first.")
    selected = select_db_ids(staged, db_ids, shard, limit)

    # Column descriptions come from the staged dev zip copy for both dev and minidev (decision
    # 2026-11-08), and keys from the db split's tables.json, so both splits share identical prompts.
    tables_by_db = {t["db_id"]: t for t in read_json(paths.split_file(db_split, "tables"))}
    written = []
    for db_id in selected:
        out = paths.schema_cache_path(split, db_id)
        if out.exists() and not overwrite:
            log.info("schema cache exists, skipping %s", db_id)
            continue
        if db_id not in tables_by_db:
            log.warning("%s missing from %s tables.json; keys come from PRAGMA only", db_id, db_split)
        schema = build_schema(
            paths.local_db_path(split, db_id),
            paths,
            tables_by_db.get(db_id),
            n_sample_rows=int(cfg.get("n_sample_rows", 3)),
            max_value_chars=int(cfg.get("max_value_chars", 80)),
        )
        write_json(out, schema)
        written.append(out)
        log.info("schema cache %s: %d tables", db_id, len(schema["tables"]))
    return written


def load_schema(split: str, db_id: str, paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    return read_json(paths.schema_cache_path(split, db_id))
