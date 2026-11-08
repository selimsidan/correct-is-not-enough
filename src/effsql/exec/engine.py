"""SQLite access. Stage 1 only needs read-only connections; execution and cost come in stage 2."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from effsql.paths import Paths, load_paths


def connect_readonly(db_path: Path, paths: Paths | None = None) -> sqlite3.Connection:
    """Open a database read-only, refusing anything outside the local staging root.

    SQLite over the Drive mount is extremely slow and would corrupt timings, so the guard is
    enforced here, at the single place where connections are created.
    """
    paths = paths or load_paths()
    resolved = paths.assert_local(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved} (run `effsql stage` first)")
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
