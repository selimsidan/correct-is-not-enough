import json

import pytest
from fakes import FakeBackend

from effsql.config import load_config
from effsql.data.bird import load_examples, select_questions
from effsql.data.schema import build_schema_cache, load_schema
from effsql.data.stage import stage
from effsql.gen.parse import canonicalize, extract_sql
from effsql.gen.prompts import SHRINK_LADDER, build_question_prompt, fill, load_templates, render_schema
from effsql.gen.rewrite import styles_for
from effsql.gen.sample import generate
from effsql.io import read_jsonl

CFG = {"stage": {"disk_reserve_gb": 0}}


@pytest.fixture
def dev_schema(fake_bird):
    stage("dev", paths=fake_bird, data_cfg=CFG)
    build_schema_cache("dev", paths=fake_bird, data_cfg=CFG)
    return fake_bird


@pytest.mark.parametrize("text,sql,how", [
    ("a\n```sql\nSELECT 1;\n```\nb\n```sql\nSELECT 2;\n```", "SELECT 2", "sql_block"),
    ("```SQL\nSELECT 1\n```", "SELECT 1", "sql_block"),
    ("```\nSELECT 3\n```", "SELECT 3", "code_block"),
    ("```sql\nSELECT 4 FROM t WHERE", "SELECT 4 FROM t WHERE", "unclosed_block"),
    ("  select 5 ;", "select 5", "raw"),
    ("WITH a AS (SELECT 1) SELECT * FROM a", "WITH a AS (SELECT 1) SELECT * FROM a", "raw"),
    ("No idea.", None, "parse_fail"),
    ("", None, "parse_fail"),
    ("```sql\n\n```", None, "parse_fail"),
])
def test_extract_sql(text, sql, how):
    assert extract_sql(text) == (sql, how)


def test_canonical_form_collapses_case_and_whitespace():
    a, ok_a = canonicalize("select T1.Name from Customers as T1 where T1.id=1")
    b, ok_b = canonicalize("SELECT  t1.name\nFROM customers AS t1 WHERE t1.id = 1")
    assert ok_a and ok_b and a == b
    q, ok = canonicalize('SELECT "Free Meal" FROM frpm WHERE x = \'Alameda\'')
    assert ok and "'Alameda'" in q  # string literals keep their case
    assert canonicalize("SELEC (((") == ("SELEC (((", False)


def test_fill_is_single_pass():
    assert fill("{a} and {b}", {"a": "{b}", "b": "x"}) == "{b} and x"
    assert fill("{missing}", {}) == "{missing}"


def test_render_schema_and_shrink(dev_schema):
    schema = load_schema("dev", "shop", dev_schema)
    full = render_schema(schema, SHRINK_LADDER[0])
    assert "CREATE TABLE customers" in full and "-- 10 rows" in full and "Example rows (3)" in full
    assert "customer name (caf\xe9)" in full
    bare = render_schema(schema, SHRINK_LADDER[-1])
    assert "Example rows" not in bare and "Column descriptions" not in bare and len(bare) < len(full)
    ex = load_examples("dev", dev_schema)[0]
    tpl = load_templates()
    msgs, level, n = build_question_prompt(tpl, "base", ex, schema, 10_000, lambda m: 1)
    assert level == "full" and msgs[0]["role"] == "system" and ex.question in msgs[1]["content"]
    _, level, _ = build_question_prompt(tpl, "base", ex, schema, 1, lambda m: len(m[1]["content"]))
    assert level == "no_desc"  # could not fit: last level returned
    eff, _, _ = build_question_prompt(tpl, "efficient", ex, schema, 10_000, lambda m: 1)
    assert "fastest" in eff[0]["content"] and "fastest" not in msgs[0]["content"]


def test_select_questions_round_robin(fake_bird):
    ex = load_examples("dev", fake_bird)  # shop, shop, school
    assert [e.db_id for e in select_questions(ex, 2)] == ["school", "shop"]
    assert select_questions(ex, None) == ex


def test_styles_rotation():
    cfg = load_config("rewrite.yaml")
    picks = [styles_for(f"train_{i}", cfg) for i in range(60)]
    assert all(p[0] == "optimized" and len(p) == cfg["k"] and len(set(p)) == len(p) for p in picks)
    assert {s for p in picks for s in p} == set(cfg["styles"])
    assert styles_for("train_7", cfg) == styles_for("train_7", cfg)


def test_generate_refuses_eval_splits(dev_schema):
    with pytest.raises(ValueError, match="analysis"):
        generate("dev", "g", FakeBackend(), paths=dev_schema)


def test_generate_resume_and_seeds(fake_bird):
    stage("train", paths=fake_bird, data_cfg=CFG)
    build_schema_cache("train", paths=fake_bird, data_cfg=CFG)
    cfg = load_config("generate.yaml")
    cfg["chunk_size"] = 4
    fb = FakeBackend()
    stats = generate("train", "g", fb, cfg=cfg, paths=fake_bird)
    recs = read_jsonl(fake_bird.run_dir("g") / "generations_train.jsonl")
    assert stats["n_requests_run"] == len(recs) == 3 * 2 * 2  # questions x variants x (greedy, sample)
    assert fb.calls == [4, 4, 4]
    samples = [r for r in recs if r["source"] == "sample"]
    assert all(len(r["completions"]) == 4 and r["params"]["seed"] is not None for r in samples)
    assert len({r["params"]["seed"] for r in samples}) == len(samples)
    assert all(r["params"]["seed"] is None for r in recs if r["source"] == "greedy")
    assert all(r["model_revision"] == "deadbeef" and r["gen_version"] == 1 for r in recs)
    fb2 = FakeBackend()
    assert generate("train", "g", fb2, cfg=cfg, paths=fake_bird)["n_requests_run"] == 0 and fb2.calls == []
    seeds_again = {r["key"]: r["params"]["seed"] for r in read_jsonl(fake_bird.run_dir("g") / "generations_train.jsonl")}
    assert seeds_again == {r["key"]: r["params"]["seed"] for r in recs}
