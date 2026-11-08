import pytest

from effsql.data.stage import stage
from effsql.exec.engine import connect_readonly, execute, execute_many, execute_two_pass
from effsql.exec.results import result_hash
from effsql.measure.correctness import calculate_ex

RUNAWAY = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"


@pytest.fixture
def shop(fake_bird):
    stage("dev", paths=fake_bird, data_cfg={"stage": {"disk_reserve_gb": 0}})
    conn = connect_readonly(fake_bird.local_db_path("dev", "shop"), fake_bird)
    yield conn
    conn.close()


def test_ok_result(shop):
    r = execute(shop, "SELECT count(*) FROM customers", keep_rows=True)
    # count(*) is a single counting instruction, so it can cost 0 steps at N=10 (hence smoothing)
    assert r.ok and r.rows == [(10,)] and r.n_rows == 1 and r.vm_steps >= 0
    assert execute(shop, "SELECT name FROM customers WHERE id > 2").vm_steps > 0


def test_read_only_enforced(shop):
    r = execute(shop, "DELETE FROM customers")
    assert not r.ok and r.error_type == "runtime" and "readonly" in r.error_msg
    assert execute(shop, "SELECT count(*) FROM customers", keep_rows=True).rows == [(10,)]


def test_timeout_aborts_runaway(shop):
    r = execute(shop, RUNAWAY, timeout_s=0.2)
    assert not r.ok and r.error_type == "timeout" and r.elapsed_ms < 2000


def test_step_budget_aborts(shop):
    r = execute(shop, RUNAWAY, timeout_s=10, step_budget=5000)
    assert not r.ok and r.error_type == "step_budget" and r.vm_steps <= 5010


def test_steps_deterministic_and_heavier_query_costs_more(shop):
    light = "SELECT count(*) FROM orders"
    heavy = "SELECT count(*) FROM orders a, orders b WHERE a.amount < b.amount"
    s1 = [execute(shop, light).vm_steps for _ in range(3)]
    s2 = [execute(shop, heavy).vm_steps for _ in range(3)]
    assert len(set(s1)) == 1 and len(set(s2)) == 1
    assert s2[0] > 10 * max(s1[0], 10)


def test_steps_identical_on_fresh_connections(fake_bird, shop):
    sql = "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE amount > 20)"
    counts = []
    for _ in range(3):
        c = connect_readonly(fake_bird.local_db_path("dev", "shop"), fake_bird)
        counts.append(execute(c, sql).vm_steps)
        counts.append(execute(c, sql).vm_steps)
        c.close()
    assert len(set(counts)) == 1 and counts[0] > 0


@pytest.mark.parametrize("sql,etype", [
    ("SELEC x FROM customers", "syntax"),
    ("SELECT nope FROM customers", "syntax"),
    ("SELECT 1; SELECT 2", "syntax"),
    ("", "parse_fail"),
    (None, "parse_fail"),
])
def test_error_types(shop, sql, etype):
    assert execute(shop, sql).error_type == etype


def test_too_many_rows(shop):
    r = execute(shop, "SELECT * FROM orders", max_rows=10)
    assert r.error_type == "too_many_rows"


@pytest.mark.parametrize("a,b", [
    ([(1, "a"), (2, "b")], [(2, "b"), (1, "a")]),      # order
    ([(1,), (1,), (2,)], [(2,), (1,)]),                 # duplicates
    ([(1,)], [(1.0,)]),                                 # 1 == 1.0 in Python sets
    ([(None,)], [(None,)]),
    ([("1",)], [(1,)]),                                 # str vs int differ
    ([(0.5,)], [(0.5000001,)]),
    ([], []),
    ([(1, 2)], [(2, 1)]),
])
def test_hash_agrees_with_set_equality(a, b):
    assert (result_hash(a) == result_hash(b)) == bool(calculate_ex(a, b))


def test_execute_many_parallel_matches_serial(fake_bird, shop):
    jobs = [{"key": i, "db_path": fake_bird.local_db_path("dev", db), "sql": sql, "features": True}
            for i, (db, sql) in enumerate([("shop", "SELECT count(*) FROM orders"),
                                           ("school", "SELECT count(*) FROM students"),
                                           ("shop", "SELEC broken")])]
    serial = {r["key"]: r for r in execute_many(jobs, fake_bird, workers=1)}
    parallel = {r["key"]: r for r in execute_many(jobs, fake_bird, workers=2)}
    for k in serial:
        assert serial[k]["result_hash"] == parallel[k]["result_hash"]
        assert serial[k]["vm_steps"] == parallel[k]["vm_steps"]
    assert serial[0]["plan"]["plan_ok"] and serial[0]["text"]["text_ok"]
    assert serial[2]["error_type"] == "syntax" and "plan" not in serial[2]


def test_uncounted_execution_has_no_steps_but_same_result(shop):
    sql = "SELECT name FROM customers WHERE id > 2"
    fast, counted = execute(shop, sql, count_steps=False), execute(shop, sql)
    assert fast.vm_steps is None and counted.vm_steps > 0
    assert fast.result_hash == counted.result_hash
    assert execute(shop, RUNAWAY, timeout_s=0.2, count_steps=False).error_type == "timeout"


def test_two_pass_separates_correctness_from_counting(fake_bird, shop):
    cfg = {"timeout_s": 30, "count_timeout_s": 30, "max_rows": 1000, "progress_n": 10, "step_budget": None}
    jobs = [{"key": k, "db_path": fake_bird.local_db_path("dev", "shop"), "sql": sql} for k, sql in
            [("a", "SELECT name FROM customers WHERE id > 2"), ("b", "SELECT amount FROM orders"), ("c", "SELEC")]]
    res = execute_two_pass(jobs, fake_bird, cfg, count_if=lambda r: r["key"] != "b", workers=1)
    assert res["a"]["ok"] and res["a"]["vm_steps"] > 0
    assert res["b"]["ok"] and res["b"]["vm_steps"] is None          # not selected for counting
    assert res["c"]["error_type"] == "syntax" and res["c"].get("vm_steps") is None
    # a counting failure (here: a step budget) leaves correctness intact and records steps_error
    res = execute_two_pass(jobs[:1], fake_bird, {**cfg, "step_budget": 5}, count_if=lambda r: True, workers=1)
    assert res["a"]["ok"] and res["a"]["vm_steps"] is None and res["a"]["steps_error"] == "step_budget"
