import pytest

from effsql.data.inspect import inspect_all
from effsql.data.schema import build_schema_cache, load_schema
from effsql.data.stage import stage

CFG = {"stage": {"disk_reserve_gb": 0}, "schema": {"n_sample_rows": 2, "max_value_chars": 10}}


def test_schema_cache(fake_bird):
    stage("dev", paths=fake_bird, data_cfg=CFG)
    written = build_schema_cache("dev", paths=fake_bird, data_cfg=CFG)
    assert len(written) == 2
    s = load_schema("dev", "shop", fake_bird)
    tables = {t["name"]: t for t in s["tables"]}
    assert set(tables) == {"customers", "orders"}
    assert tables["orders"]["row_count"] == 50
    assert len(tables["orders"]["sample_rows"]) == 2
    note = tables["orders"]["sample_rows"][0][tables["orders"]["sample_columns"].index("note")]
    assert note == "x" * 10 + "..."
    assert "CREATE TABLE customers" in tables["customers"]["ddl"]
    descs = tables["customers"]["column_descriptions"]
    assert descs[1]["column_description"] == "customer name (caf\xe9)"
    assert tables["customers"]["description_encoding"] == "cp1252"
    assert {"from": "orders.customer_id", "to": "customers.id"} in s["foreign_keys_pragma"]
    assert s["tables_json"]["foreign_keys"] == [{"from": "orders.customer_id", "to": "customers.id"}]
    blob = load_schema("dev", "school", fake_bird)["tables"][0]["sample_rows"][0][2]
    assert blob == "<blob 2 bytes>"


def test_schema_cache_skips_existing_and_minidev_shares_dev(fake_bird):
    stage("dev", paths=fake_bird, data_cfg=CFG)
    assert len(build_schema_cache("dev", paths=fake_bird, data_cfg=CFG)) == 2
    assert build_schema_cache("dev", paths=fake_bird, data_cfg=CFG) == []
    assert build_schema_cache("minidev", paths=fake_bird, data_cfg=CFG) == []
    assert fake_bird.schema_cache_path("minidev", "shop") == fake_bird.schema_cache_path("dev", "shop")


def test_schema_cache_requires_staging(fake_bird):
    with pytest.raises(RuntimeError, match="stage"):
        build_schema_cache("train", paths=fake_bird, data_cfg=CFG)


def test_inspect_all(fake_bird):
    r = inspect_all(fake_bird)
    errors = {k: v for k, v in r.items() if isinstance(v, dict) and "error" in v}
    assert not errors, errors
    assert r["dev.zip"]["db_ids"] == ["school", "shop"]
    # folder entry + 2 sqlite + 2 csv; the 5 junk entries are not counted
    assert r["dev.zip"]["top_level"] == {"dev_databases": 5}
    assert r["dev.zip"]["n_junk_entries"] == 5
    assert r["train.questions"]["first_record_keys"] == ["db_id", "question", "evidence", "SQL"]
    assert r["dev.gold"]["tabs_per_line"] == {1: 3}
    assert r["dev.gold"]["json_sql_vs_gold"]["whitespace_only_diff"] == 0
    assert r["dev.gold"]["json_sql_vs_gold"]["different"] == 1
    assert r["minidev.coverage"]["missing_from_zip"] == []
    assert r["minidev_vs_dev"]["n_found_in_dev"] == 3
    assert r["minidev_vs_dev"]["n_question_and_evidence_identical"] == 2
    mq = r["minidev.questions"]
    assert mq["question_id_unique"] is False
    assert mq["duplicated_question_ids"] == [{
        "question_id": 0, "file_positions": [1, 2], "records_identical": False,
        "differing_fields": ["question"], "questions": ["How many customers?", "Count the customers."],
    }]
    d = r["descriptions_dev_zip_vs_minidev"]
    assert d["all_identical"] is False
    shop = d["per_db"]["shop"]
    assert shop["different"] == ["orders.csv"] and shop["identical"] == 1
    assert shop["only_in_zip"] == [] and shop["only_in_minidev"] == []
    assert shop["file_name_case_differences"] == 1
    assert r["dev_tied_append"]["n_sql_differs_from_dev_gold"] == 2
    assert r["dev_tied_append"]["reworded_question_ids (excluded from EX-tied)"] == [1]


def test_shard_universe_is_shared_by_stage_and_schema_cache(fake_bird):
    # dev universe is [school, shop]; shard 0/2 is [school] for both commands
    manifest = stage("dev", shard="0/2", paths=fake_bird, data_cfg=CFG)
    assert list(manifest["dbs"]) == ["school"]
    written = build_schema_cache("dev", shard="0/2", paths=fake_bird, data_cfg=CFG)
    assert [p.stem for p in written] == ["school"]
    with pytest.raises(RuntimeError, match="same"):
        build_schema_cache("dev", shard="1/2", paths=fake_bird, data_cfg=CFG)
