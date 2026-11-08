"""SQLite execution with deterministic cost counting.

Cost: a progress handler is called every `progress_n` VM instructions; vm_steps = calls x N.
The same handler enforces the wall-clock deadline and the optional step budget by returning
nonzero, which aborts the query.

Counting every 10 instructions slows queries 2 to 4x (Python callback overhead; measured on
Colab 2026-11-08), which would make slow-but-valid queries hit the official 30 s limit. So work is
split in two passes (`execute_two_pass`): correctness at native speed with a deadline-only handler
every 1000 instructions (about 1% overhead), then step counting only where needed, with its own
generous limit. elapsed_ms of the counting pass is never used as a timing measurement.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from effsql.exec.results import result_hash
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)

ERROR_TYPES = ("parse_fail", "syntax", "runtime", "timeout", "step_budget", "too_many_rows")
DEADLINE_CHECK_N = 1000  # handler period when only the deadline is checked
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
    count_steps: bool = True,
    keep_rows: bool = False,
) -> ExecResult:
    """Run one query. With count_steps, vm_steps is counted every progress_n instructions and the
    step budget applies; without, only the deadline is checked (near native speed, vm_steps None)."""
    if sql is None or not sql.strip():
        return ExecResult(ok=False, error_type="parse_fail", error_msg="empty SQL")
    n = progress_n if count_steps else DEADLINE_CHECK_N
    state = {"calls": 0, "reason": None}
    deadline = time.perf_counter() + timeout_s

    def on_progress() -> int:
        state["calls"] += 1
        if count_steps and step_budget is not None and state["calls"] * n > step_budget:
            state["reason"] = "step_budget"
            return 1
        if time.perf_counter() > deadline:
            state["reason"] = "timeout"
            return 1
        return 0

    def steps() -> int | None:
        return state["calls"] * n if count_steps else None

    conn.set_progress_handler(on_progress, n)
    t0 = time.perf_counter()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cur.close()
        elapsed = (time.perf_counter() - t0) * 1000
        if len(rows) > max_rows:
            return ExecResult(ok=False, error_type="too_many_rows", error_msg=f"more than {max_rows} rows",
                              vm_steps=steps(), elapsed_ms=elapsed)
        return ExecResult(ok=True, n_rows=len(rows), result_hash=result_hash(rows), vm_steps=steps(),
                          elapsed_ms=elapsed, rows=rows if keep_rows else None)
    except Exception as e:  # noqa: BLE001 (any failure of generated SQL is a result, not a crash)
        elapsed = (time.perf_counter() - t0) * 1000
        etype = state["reason"] or classify_error(e)
        return ExecResult(ok=False, error_type=etype, error_msg=f"{type(e).__name__}: {e}"[:500],
                          vm_steps=steps(), elapsed_ms=elapsed)
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


def execute_two_pass(
    jobs: list[dict[str, Any]],
    paths: Paths,
    exec_cfg: dict[str, Any],
    count_if: Callable[[dict[str, Any]], bool],
    workers: int | None = None,
) -> dict[Any, dict[str, Any]]:
    """Correctness at native speed, then vm_steps for the records where count_if(record) is true.

    Pass 1: official timeout (exec_cfg.timeout_s), deadline check only; gives ok, error_type,
            result_hash, n_rows, elapsed_ms (near native) and, for jobs with `features`, plan
            and text features.
    Pass 2: counted every progress_n instructions with exec_cfg.count_timeout_s; fills vm_steps,
            or leaves it None with `steps_error` when counting fails (e.g. the longer limit).
    """
    first = {
        r["key"]: r
        for r in execute_many(jobs, paths, workers=workers, timeout_s=exec_cfg["timeout_s"],
                              max_rows=exec_cfg["max_rows"], count_steps=False)
    }
    second_jobs = [{k: v for k, v in j.items() if k != "features"} for j in jobs if count_if(first[j["key"]])]
    for r in execute_many(second_jobs, paths, workers=workers, timeout_s=exec_cfg["count_timeout_s"],
                          step_budget=exec_cfg.get("step_budget"), max_rows=exec_cfg["max_rows"],
                          progress_n=exec_cfg["progress_n"], count_steps=True):
        rec = first[r["key"]]
        if not r["ok"]:
            rec["steps_error"] = r["error_type"]
        elif r["result_hash"] != rec["result_hash"]:
            rec["steps_error"] = "result_changed"  # should never happen; recorded rather than trusted
        else:
            rec["vm_steps"] = r["vm_steps"]
    return first


def batched(it: Iterable[Any], n: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
