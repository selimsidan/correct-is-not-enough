"""`quick` wall-clock timing protocol (development and pool verification).

Strictly sequential in one process. Each run times execute + fetchall with time.perf_counter.
`warmup` rounds are discarded, then `runs` rounds interleave the queries (G, P, G, P, ...), so
slow drifts in machine load hit both sides equally. A coarse progress handler (every 1000 VM
instructions, applied to every query alike) enforces the deadline.

The `official` protocol for reported numbers is in measure/bird_official.py.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

_DEADLINE_N = 1000


class QueryTimeout(Exception):
    pass


def timed_run(conn: sqlite3.Connection, sql: str, timeout_s: float) -> float:
    """Seconds for one execution including fetching every row."""
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), _DEADLINE_N)
    try:
        t0 = time.perf_counter()
        conn.execute(sql).fetchall()
        return time.perf_counter() - t0
    except sqlite3.OperationalError as e:
        if time.perf_counter() > deadline:
            raise QueryTimeout(sql[:80]) from e
        raise
    finally:
        conn.set_progress_handler(None, 0)


def trimmed_mean(xs: list[float]) -> float:
    """Mean without the single min and max (when there are at least 5 values)."""
    s = sorted(xs)
    return statistics.fmean(s[1:-1] if len(s) >= 5 else s)


def cv(xs: list[float]) -> float:
    m = statistics.fmean(xs)
    return statistics.pstdev(xs) / m if m > 0 and len(xs) > 1 else 0.0


@dataclass
class Timing:
    times: list[float]
    median: float
    trimmed_mean: float
    cv: float

    @classmethod
    def of(cls, times: list[float]) -> "Timing":
        return cls(times, statistics.median(times), trimmed_mean(times), cv(times))


@dataclass
class InterleavedResult:
    timings: list[Timing] = field(default_factory=list)
    error: str | None = None


def time_interleaved(
    conn: sqlite3.Connection, sqls: list[str], *, warmup: int = 1, runs: int = 5, timeout_s: float = 30.0
) -> InterleavedResult:
    """Time several queries round-robin on one connection."""
    times: list[list[float]] = [[] for _ in sqls]
    try:
        for r in range(warmup + runs):
            for i, sql in enumerate(sqls):
                t = timed_run(conn, sql, timeout_s)
                if r >= warmup:
                    times[i].append(t)
    except Exception as e:  # noqa: BLE001
        return InterleavedResult(error=f"{type(e).__name__}: {e}"[:300])
    return InterleavedResult(timings=[Timing.of(t) for t in times])


def time_pair_quick(
    conn: sqlite3.Connection, gold_sql: str, pred_sql: str, *, warmup: int = 1, runs: int = 5,
    timeout_s: float = 30.0, cv_threshold: float = 0.2,
) -> dict[str, Any]:
    """Quick gold vs prediction timing; ratio = median(gold) / median(pred) (>1: pred faster)."""
    res = time_interleaved(conn, [gold_sql, pred_sql], warmup=warmup, runs=runs, timeout_s=timeout_s)
    if res.error:
        return {"timing_error": res.error}
    g, p = res.timings
    return {
        "gold_times": g.times, "pred_times": p.times,
        "gold_median": g.median, "pred_median": p.median,
        "gold_trimmed_mean": g.trimmed_mean, "pred_trimmed_mean": p.trimmed_mean,
        "gold_cv": g.cv, "pred_cv": p.cv,
        "quick_ratio": g.median / p.median if p.median > 0 else None,
        "unstable": max(g.cv, p.cv) > cv_threshold,
    }
