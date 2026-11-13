"""Quality check: flagging arbitrary row picks, verdicts, and the effect on pool and training export."""

import pytest
from fakes import FakeBackend

from effsql.config import load_config
from effsql.data.schema import build_schema_cache
from effsql.data.stage import stage
from effsql.gen.parse import parse_run
from effsql.gen.sample import generate
from effsql.io import append_jsonl, read_json, read_jsonl
from effsql.pool.build import build_pool
from effsql.pool.diversity import diversity
from effsql.pool.quality import analyze, quality, verdict
from effsql.pool.screen import screen
from effsql.train.data import read_export, train_export


@pytest.mark.parametrize("sql", [
    "SELECT a FROM t",
    "SELECT a FROM t WHERE x = (SELECT MAX(x) FROM t)",                       # one row by construction
    "SELECT a FROM t WHERE x = (SELECT COUNT(*) FROM u WHERE u.k = t.k)",      # correlated aggregate
    "SELECT a FROM t WHERE x = (SELECT x FROM u ORDER BY y DESC LIMIT 1)",     # ordered pick (BIRD convention)
    "SELECT a FROM t ORDER BY b DESC LIMIT 1",                                 # top-k
    "SELECT a FROM t WHERE x IN (SELECT x FROM u)",
])
def test_not_flagged(sql):
    a = analyze(sql)
    assert a["flags"] == [] and a["variant"] is None and a["roundtrip"]


def test_scalar_equality_becomes_in():
    a = analyze("SELECT Path FROM Solution WHERE Id = (SELECT SolutionId FROM Method WHERE Name = 'x')")
    assert a["flags"] == ["scalar_subquery"] and not a["untransformable"]
    assert "IN (SELECT SolutionId FROM Method" in a["variant"] and "= (SELECT" not in a["variant"]


def test_scalar_inequality_and_left_side():
    a = analyze("SELECT a FROM t WHERE (SELECT b FROM u) <> a")
    assert a["flags"] == ["scalar_subquery"] and "NOT a IN (SELECT b FROM u)" in a["variant"]


def test_nested_and_multiple_scalars_all_rewritten():
    a = analyze("SELECT a FROM t WHERE x = (SELECT x FROM u WHERE y = (SELECT y FROM v)) AND z = (SELECT z FROM w)")
    assert a["variant"].count(" IN (SELECT") == 3 and "= (SELECT" not in a["variant"]


def test_unordered_limit_removed_and_group_by_is_multi_row():
    a = analyze("SELECT a FROM t WHERE x = (SELECT MAX(x) FROM u GROUP BY k) LIMIT 2")
    assert a["flags"] == ["scalar_subquery", "unordered_limit"]
    assert "LIMIT" not in a["variant"] and "IN (SELECT MAX(x) FROM u GROUP BY k)" in a["variant"]
    b = analyze("SELECT a FROM t WHERE x = (SELECT x FROM u LIMIT 1)")  # LIMIT without ORDER BY: arbitrary
    assert b["flags"] == ["scalar_subquery", "unordered_limit"] and "LIMIT" not in b["variant"]


def test_untransformable_and_parse_error():
    a = analyze("SELECT a FROM t WHERE x > (SELECT x FROM u)")
    assert a["flags"] == ["scalar_subquery"] and a["variant"] is None and "GT" in a["untransformable"]
    assert verdict(a, "h", None, None)[0] == "unverifiable"
    p = analyze("SELEC broken (((")
    assert p["flags"] == ["parse_error"] and verdict(p, "h", None, None)[0] == "unverifiable"


def test_verdicts():
    a = analyze("SELECT a FROM t WHERE x = (SELECT x FROM u)")
    ok = {"ok": True, "result_hash": "g", "n_rows": 1, "error_type": None}
    diff = {"ok": True, "result_hash": "other", "n_rows": 3, "error_type": None}
    assert verdict(analyze("SELECT 1"), "g", None, None) == ("ok", None)
    assert verdict(a, "g", ok, ok)[0] == "verified"
    assert verdict(a, "g", ok, diff)[0] == "fragile"
    assert verdict(a, "g", ok, {"ok": False, "error_type": "too_many_rows"})[0] == "fragile"
    assert verdict(a, "g", ok, {"ok": False, "error_type": "timeout"})[0] == "unverifiable"
    assert verdict(a, "g", diff, ok)[0] == "unverifiable"  # sqlglot re-rendering changed the result


FRAGILE = "SELECT id FROM movies WHERE id = (SELECT id FROM movies ORDER BY id DESC)"   # first row happens to be 19
SAFE = "SELECT id FROM movies WHERE id = (SELECT id FROM movies WHERE title = 't19')"   # one row anyway


@pytest.fixture
def pool_run(fake_bird):
    p, run = fake_bird, "q"
    cfg = {"stage": {"disk_reserve_gb": 0}}
    stage("train", paths=p, data_cfg=cfg)
    build_schema_cache("train", paths=p, data_cfg=cfg)
    generate("train", run, FakeBackend(), paths=p)
    gens = p.run_dir(run) / "generations_train.jsonl"
    rec = next(r for r in read_jsonl(gens) if r["qid"] == "train_2" and r["source"] == "sample")  # "Max id?"
    extra = dict(rec, key="train_2|base|extra", completions=[
        {"text": f"```sql\n{FRAGILE}\n```", "finish_reason": "stop"},
        {"text": f"```sql\n{SAFE}\n```", "finish_reason": "stop"}])
    append_jsonl(gens, [extra])
    parse_run("train", run, paths=p)
    exec_cfg = load_config("evaluate.yaml")["exec"]
    exec_cfg["workers"] = 1
    screen("train", run, paths=p, exec_cfg=exec_cfg)
    diversity("train", run, 0, paths=p)
    return p, run, exec_cfg


def test_quality_end_to_end(pool_run):
    p, run, exec_cfg = pool_run
    build_pool("train", run, paths=p)
    with pytest.raises(RuntimeError, match="no quality verdict"):
        train_export(run, "tr0", paths=p)  # the export refuses an unchecked pool

    stats = quality("train", run, paths=p, exec_cfg=exec_cfg)
    recs = {r["cand_id"]: r for r in read_jsonl(p.run_dir(run) / "quality_train.jsonl")}
    pool = build_pool("train", run, paths=p)
    by_sql = pool.set_index("sql_original")
    assert by_sql.loc[FRAGILE, "correct"] and by_sql.loc[FRAGILE, "quality"] == "fragile"
    assert by_sql.loc[SAFE, "correct"] and by_sql.loc[SAFE, "quality"] == "verified"
    assert (pool[pool["is_gold"]]["quality"] == "gold").all()
    assert pool[~pool["correct"]]["quality"].isna().all()
    assert stats["verdicts"]["fragile"] == 1 and stats["verdicts"]["verified"] == 1
    assert stats["n_checked"] == stats["n_correct_model_candidates"]

    n = len(recs)
    assert quality("train", run, paths=p, exec_cfg=exec_cfg)["verdicts"] == stats["verdicts"]  # resume: no rework
    assert len(read_jsonl(p.run_dir(run) / "quality_train.jsonl")) == n

    meta = train_export(run, "tr", paths=p)
    assert meta["quality_filter"]["n_dropped"] == 1
    fragile_id = by_sql.loc[FRAGILE, "cand_id"]
    rd = p.run_dir("tr")
    for arm in ("plain", "weighted", "tagged"):
        for part in ("train", "val"):
            assert all(r["cand_id"] != fragile_id for r in read_export(rd, arm, part))
    for part in ("train", "val"):
        assert all(fragile_id not in (r["chosen_cand"], r["rejected_cand"]) for r in read_export(rd, "dpo", part))
    assert read_json(rd / "export_meta.json")["quality_filter"]["dropped_by_verdict"] == {"fragile": 1}
