"""Official BIRD efficiency metrics: R-VES (mini-dev) and VES (original BIRD).

Sources (verified 2026-11-08):
  R-VES: repository bird-bench/mini_dev, commit 1c49a375e76c471e218564da3f4617513d33e371,
         file evaluation/evaluation_ves.py (execute_sql, clean_abnormal, iterated_execute_sql,
         execute_model, compute_ves). The evaluation/ code there has been unchanged since commit
         47f116d (2024-10-25), so it matches the 2024 mini-dev data release.
  VES:   repository AlibabaResearch/DAMO-ConvAI, commit faf754671f5b850e1b4d1ff870db37a6a2480df6,
         file bird/llm/src/evaluation_ves.py (compute_ves: mean of sqrt(time_ratio) * 100).

Ported exactly:
  * timing of one execution: connect (untimed), then time.time() around cursor creation,
    execute, fetchall and close;
  * correctness: set(predicted_res) == set(ground_truth_res), on fresh connections;
  * if correct: iterate_num (100) rounds of prediction then gold, each appending gold/pred;
  * clean_abnormal keeps values strictly inside mean +/- 3 * np.std (population std); the ratio
    is their mean (an empty list raises ZeroDivisionError, which the official code turns into
    reward 0; kept as is);
  * reward buckets: >= 2: 1.25, [1, 2): 1, [0.5, 1): 0.75, [0.25, 0.5): 0.5, else 0.25, and 0
    when time_ratio == 0 (incorrect);
  * any exception or exceeding meta_time_out * iterate_num for the whole pair gives reward 0;
  * score = sum(sqrt(reward) * 100) / n over all questions.

Documented deviations:
  * pairs run sequentially (official: multiprocessing pool with num_cpus=16), because parallel
    runs disturb timing;
  * connections are read-only (`mode=ro`); the official code opens read-write (no effect on speed);
  * the pair deadline is enforced by a progress handler called every 1000 VM instructions on both
    queries (official: func_timeout thread);
  * VES here applies the original VES formula to the mini-dev timing loop (with fetchall)
    instead of the original loop, which timed only cursor.execute (time to the first row).
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from effsql.paths import Paths

_DEADLINE_N = 1000


class PairTimeout(Exception):
    pass


def clean_abnormal(input):  # noqa: A002 (name kept from the official code)
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def reward_from_ratio(time_ratio: float) -> float:
    if time_ratio == 0:
        reward = 0
    elif time_ratio >= 2:
        reward = 1.25
    elif time_ratio >= 1 and time_ratio < 2:
        reward = 1
    elif time_ratio >= 0.5 and time_ratio < 1:
        reward = 0.75
    elif time_ratio >= 0.25 and time_ratio < 0.5:
        reward = 0.5
    else:
        reward = 0.25
    return reward


def rves_score(rewards: list[float]) -> float:
    """compute_ves from mini_dev evaluation_ves.py."""
    if not rewards:
        return float("nan")
    return sum(math.sqrt(r) * 100 for r in rewards) / len(rewards)


def ves_score(time_ratios: list[float]) -> float:
    """compute_ves from DAMO-ConvAI bird/llm/src/evaluation_ves.py (uncapped)."""
    if not time_ratios:
        return float("nan")
    return sum(math.sqrt(r) * 100 for r in time_ratios) / len(time_ratios)


def _execute_sql(sql: str, uri: str, deadline: float, return_time: bool = False):
    """execute_sql from mini_dev evaluation_ves.py, plus the pair deadline."""
    conn = sqlite3.connect(uri, uri=True)
    conn.set_progress_handler(lambda: int(time.time() > deadline), _DEADLINE_N)
    try:
        start_time = time.time()
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        conn.close()
        exec_time = time.time() - start_time
    except Exception as e:
        conn.close()
        if isinstance(e, sqlite3.OperationalError) and time.time() > deadline:
            raise PairTimeout() from e
        raise
    if return_time:
        return exec_time
    return res


@dataclass
class OfficialPairResult:
    correct: bool
    time_ratio: float
    reward: float
    n_kept: int = 0
    error: str | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


def iterated_execute_sql(
    predicted_sql: str, ground_truth: str, db_path: Path, paths: Paths,
    iterate_num: int = 100, meta_time_out: float = 30.0,
) -> OfficialPairResult:
    """iterated_execute_sql + execute_model from mini_dev evaluation_ves.py."""
    uri = f"{paths.assert_local(db_path).as_uri()}?mode=ro"
    deadline = time.time() + meta_time_out * iterate_num
    correct = False
    try:
        diff_list = []
        predicted_res = _execute_sql(predicted_sql, uri, deadline)
        ground_truth_res = _execute_sql(ground_truth, uri, deadline)
        time_ratio = 0
        correct = set(predicted_res) == set(ground_truth_res)
        n_kept = 0
        if correct:
            for _ in range(iterate_num):
                predicted_time = _execute_sql(predicted_sql, uri, deadline, return_time=True)
                ground_truth_time = _execute_sql(ground_truth, uri, deadline, return_time=True)
                diff_list.append(ground_truth_time / predicted_time)
            processed_diff_list = clean_abnormal(diff_list)
            n_kept = len(processed_diff_list)
            time_ratio = sum(processed_diff_list) / len(processed_diff_list)
        return OfficialPairResult(correct, float(time_ratio), reward_from_ratio(time_ratio), n_kept)
    except PairTimeout:
        return OfficialPairResult(correct, 0.0, 0, error="timeout")
    except Exception as e:  # noqa: BLE001 (official: any exception gives reward 0)
        return OfficialPairResult(correct, 0.0, 0, error=f"{type(e).__name__}: {e}"[:300])
