import math
import sqlite3

import numpy as np
import pytest

from effsql.data.stage import stage
from effsql.exec.engine import connect_readonly
from effsql.measure import bird_official as bo
from effsql.measure.cost import eff_score, plan_features, rel_cost, text_features
from effsql.measure.timing import time_pair_quick


@pytest.mark.parametrize("ratio,reward", [
    (0, 0), (5.0, 1.25), (2.0, 1.25), (1.999, 1), (1.0, 1), (0.999, 0.75), (0.5, 0.75),
    (0.4999, 0.5), (0.25, 0.5), (0.2499, 0.25), (0.01, 0.25),
])
def test_reward_buckets(ratio, reward):
    assert bo.reward_from_ratio(ratio) == reward


def test_clean_abnormal_matches_numpy_rule():
    xs = [1.0] * 30 + [1.1] * 30 + [50.0]
    mean, std = np.mean(xs), np.std(xs)
    expected = [x for x in xs if mean - 3 * std < x < mean + 3 * std]
    assert bo.clean_abnormal(xs) == expected and 50.0 not in expected
    assert bo.clean_abnormal([2.0, 2.0, 2.0]) == []  # std 0: official code then divides by zero


def test_scores():
    assert bo.rves_score([1.25, 1, 0.75, 0]) == pytest.approx(
        (math.sqrt(1.25) + 1 + math.sqrt(0.75) + 0) * 100 / 4)
    assert bo.ves_score([4.0, 1.0, 0.0]) == pytest.approx((2 + 1 + 0) * 100 / 3)


def test_smoothed_cost_formulas():
    assert rel_cost(0, 0, 10) == 1.0 and eff_score(0, 0, 10) == 0.0
    assert rel_cost(10, 30, 10) == 0.5 and eff_score(10, 30, 10) == pytest.approx(1.0)
    assert eff_score(90, 10, 10) == pytest.approx(-math.log2(5))


@pytest.fixture
def staged(fake_bird):
    stage("dev", paths=fake_bird, data_cfg={"stage": {"disk_reserve_gb": 0}})
    return fake_bird


def test_official_pair_gold_vs_gold(staged):
    db = staged.local_db_path("dev", "shop")
    r = bo.iterated_execute_sql("SELECT count(*) FROM orders", "SELECT count(*) FROM orders", db, staged,
                                iterate_num=5)
    assert r.correct and r.time_ratio > 0 and r.reward in (0.25, 0.5, 0.75, 1, 1.25) and r.n_kept >= 1


def test_official_pair_incorrect_and_error(staged):
    db = staged.local_db_path("dev", "shop")
    wrong = bo.iterated_execute_sql("SELECT 1", "SELECT count(*) FROM orders", db, staged, iterate_num=3)
    assert not wrong.correct and wrong.reward == 0 and wrong.time_ratio == 0
    err = bo.iterated_execute_sql("SELEC", "SELECT 1", db, staged, iterate_num=3)
    assert err.reward == 0 and err.error


def test_official_identical_ratios_give_zero_like_official(staged, monkeypatch):
    real = bo._execute_sql

    def fake(sql, uri, deadline, return_time=False):
        return 0.5 if return_time else real(sql, uri, deadline)

    monkeypatch.setattr(bo, "_execute_sql", fake)
    r = bo.iterated_execute_sql("SELECT 1", "SELECT 1", staged.local_db_path("dev", "shop"), staged, iterate_num=4)
    assert r.correct and r.reward == 0 and "ZeroDivisionError" in r.error


def test_official_pair_timeout(staged):
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    r = bo.iterated_execute_sql(runaway, "SELECT 1", staged.local_db_path("dev", "shop"), staged,
                                iterate_num=1, meta_time_out=0.2)
    assert r.reward == 0 and r.error == "timeout"


def test_quick_timing(staged):
    conn = connect_readonly(staged.local_db_path("dev", "shop"), staged)
    t = time_pair_quick(conn, "SELECT count(*) FROM orders", "SELECT count(*) FROM orders", runs=5)
    assert len(t["gold_times"]) == 5 and len(t["pred_times"]) == 5 and t["quick_ratio"] > 0


def test_plan_signature_ignores_alias_spelling():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE a(id INTEGER PRIMARY KEY, x INT); CREATE TABLE b(id INTEGER PRIMARY KEY, a_id INT);")
    p1 = plan_features(c, "SELECT * FROM a AS T1 JOIN b AS T2 ON T1.id = T2.a_id ORDER BY T1.x")
    p2 = plan_features(c, "SELECT * FROM a x1 JOIN b x2 ON x1.id = x2.a_id ORDER BY x1.x")
    assert p1["plan_signature"] == p2["plan_signature"]
    assert p1["n_temp_btree"] == 1 and p1["n_scan"] >= 1
    p3 = plan_features(c, "SELECT x, (SELECT count(*) FROM b WHERE b.a_id = a.id) FROM a")
    assert p3["n_correlated"] == 1 and p3["plan_signature"] != p1["plan_signature"]


def test_text_features():
    f = text_features("WITH t AS (SELECT id FROM a) SELECT * FROM t JOIN b ON t.id = b.a_id "
                      "WHERE b.id IN (SELECT id FROM b)")
    assert f["text_ok"] and f["n_joins"] == 1 and f["n_ctes"] == 1 and f["nesting_depth"] >= 1
    assert text_features("SELEC (((")["text_ok"] is False
