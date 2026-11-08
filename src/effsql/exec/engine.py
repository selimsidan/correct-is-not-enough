"""SQLite execution with deterministic cost counting.

Cost: a progress handler is called every `progress_n` VM instructions; vm_steps = calls x N.
The same handler enforces the wall-clock deadline and the optional step budget by returning
nonzero, which aborts the query. Counting adds Python callback overhead, so elapsed_ms here is
NOT a timing measurement; timing lives in measure/timing.py and measure/bird_official.py.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from effsql.exec.results import result_hash
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)

ERROR_TYPES = ("parse_fail", "syntax", "runtime", "timeout", "step_budget", "too_many_rows")
# Messages raised when a statement cannot be compiled (bad syntax or unknown names). Reported as
# `syntax` to separate "not a valid query for this schema" from failures while running.
_COMPILE_MARKERS = (
    "syntax error", "incomplete input", "unrecognized token", "no such table", "no such column",
    "no such function", "ambiguous column", "misuse of", "wrong number of arguments",
    "only execute one statement", "near \"", "no tables specified", "is not a function",
)


def connect_readonly(db_path: Path, paths: Paths | None = None) -> sqlite3.Connection:
    """Open a database read-only, refusing anything outside the local staging root.

    SQLite over the Drive mount is extremely slow and would corrupt timings, so the guard is
    enforced here, at the single place where connections are created. Read-only mode also
    applies to ATTACHed databases, so generated SQL cannot write anywhere.
    """
    paths = paths or load_paths()
    resolved = paths.assert_local(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved} (run `effsql stage` first)")
    # cached_statements=0: a reused prepared statement keeps its VM step counter, which shifts
    # where the every-N progress callback fires and makes vm_steps vary between identical runs.
    conn = sqlite3.connect(
        f"{resolved.as_uri()}?mode=ro", uri=True, check_same_thread=False, cached_statements=0
    )
    # Load the schema now: SQLite parses it lazily on the first statement, running extra VM
    # instructions that would otherwise be counted against whichever query happens to run first.
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    return conn


@dataclass
class ExecResult:
    ok: bool
    n_rows: int = 0
    result_hash: str | None = None
    error_type: str | None = None
    error_msg: str | None = None
    vm_steps: int | None = None
    elapsed_ms: float = 0.0
    rows: list[tuple] | None = None

    def record(self) -> dict[str, Any]:
        """JSON-able form without the rows."""
        d = asdict(self)
        d.pop("rows")
        return d


def classify_error(exc: BaseException) -> str:
    msg = str(exc).lower()
    if isinstance(exc, sqlite3.ProgrammingError) or any(m in msg for m in _COMPILE_MARKERS):
        return "syntax"
    return "runtime"


def execute(
    conn: sqlite3.Connection,
    sql: str | None,
    *,
    timeout_s: float = 30.0,
    step_budget: int | None = None,
    max_rows: int = 1_000_000,
    progress_n: int = 10,
    keep_rows: bool = False,
) -> ExecResult:
    """Run one query, counting VM steps and enforcing the deadline and step budget."""
    if sql is None or not sql.strip():
        return ExecResult(ok=False, error_type="parse_fail", error_msg="empty SQL")
    state = {"calls": 0, "reason": None}
    deadline = time.perf_counter() + timeout_s

    def on_progress() -> int:
        state["calls"] += 1
        if step_budget is not None and state["calls"] * progress_n > step_budget:
            state["reason"] = "step_budget"
            return 1
        if time.perf_counter() > deadline:
            state["reason"] = "timeout"
            return 1
        return 0

    conn.set_progress_handler(on_progress, progress_n)
    t0 = time.perf_counter()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cur.close()
        elapsed = (time.perf_counter() - t0) * 1000
        steps = state["calls"] * progress_n
        if len(rows) > max_rows:
            return ExecResult(ok=False, error_type="too_many_rows", error_msg=f"more than {max_rows} rows",
                              vm_steps=steps, elapsed_ms=elapsed)
        return ExecResult(ok=True, n_rows=len(rows), result_hash=result_hash(rows), vm_steps=steps,
                          elapsed_ms=elapsed, rows=rows if keep_rows else None)
    except Exception as e:  # noqa: BLE001 (any failure of generated SQL is a result, not a crash)
        elapsed = (time.perf_counter() - t0) * 1000
        etype = state["reason"] or classify_error(e)
        return ExecResult(ok=False, error_type=etype, error_msg=f"{type(e).__name__}: {e}"[:500],
                          vm_steps=state["calls"] * progress_n, elapsed_ms=elapsed)
    finally:
        conn.set_progress_handler(None, 0)


# Parallel execution (correctness and step counting only; never timing) -------------------------

_worker_conns: dict[str, sqlite3.Connection] = {}
_worker_paths: Paths | None = None
_worker_kwargs: dict[str, Any] = {}


def _init_worker(paths: Paths, kwargs: dict[str, Any]) -> None:
    global _worker_paths, _worker_kwargs
    _worker_paths, _worker_kwargs = paths, kwargs


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    db_path = str(job["db_path"])
    conn = _worker_conns.get(db_path)
    if conn is None:
        conn = _worker_conns[db_path] = connect_readonly(Path(db_path), _worker_paths)
    res = execute(conn, job["sql"], **_worker_kwargs)
    out = {"key": job["key"], **res.record()}
    if job.get("features") and res.ok:
        from effsql.measure.cost import plan_features, text_features

        out["plan"] = plan_features(conn, job["sql"])
        out["text"] = text_features(job["sql"])
    return out


def execute_many(
    jobs: list[dict[str, Any]],
    paths: Paths,
    *,
    workers: int | None = None,
    **exec_kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Run jobs ({key, db_path, sql, features?}) in a process pool; yields records as they finish.
    With `features: True` a successful job also gets EXPLAIN QUERY PLAN and text features.

    Jobs are sorted by database so each worker reuses its connection; one connection per
    database per worker.
    """
    jobs = sorted(jobs, key=lambda j: str(j["db_path"]))
    workers = workers or os.cpu_count() or 1
    if workers == 1 or len(jobs) <= 1:
        _init_worker(paths, exec_kwargs)
        try:
            for job in jobs:
                yield _run_job(job)
        finally:
            for c in _worker_conns.values():
                c.close()
            _worker_conns.clear()
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(paths, exec_kwargs)) as pool:
        yield from pool.imap_unordered(_run_job, jobs, chunksize=1)


def batched(it: Iterable[Any], n: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
