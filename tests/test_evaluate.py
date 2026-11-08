import csv
import json
import sys

import pytest

from effsql.config import load_config
from effsql.data.stage import stage
from effsql.evaluate import evaluate, load_predictions
from effsql.data.bird import load_examples
from effsql.export import export_official
from effsql.io import read_jsonl


@pytest.fixture
def cfg():
    c = load_config("evaluate.yaml")
    c["exec"]["workers"] = 1
    c["timing"]["official"]["iterate_num"] = 3
    c["timing"]["quick"]["runs"] = 3
    return c


@pytest.fixture
def staged(fake_bird):
    stage("dev", paths=fake_bird, data_cfg={"stage": {"disk_reserve_gb": 0}})
    return fake_bird


def test_gold_vs_gold_dev(staged, cfg):
    m = evaluate("gold", "dev", "sanity", timing="none", paths=staged, cfg=cfg)
    o = m["overall"]
    assert o["ex"] == 100.0 and o["rel_cost_median"] == 1.0 and o["rel_cost_geomean"] == pytest.approx(1.0)
    assert o["faster_than_gold_rate"] == 0.0 and o["eff_acc@1.0"] == 100.0
    assert set(m["by_difficulty"]) == {"simple", "moderate", "challenging"}
    with open(staged.results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["run"] == "sanity" and float(rows[0]["ex"]) == 100.0
    assert (staged.run_dir("sanity") / "run_meta.json").exists()


def test_quick_and_official_timing_with_repeats(staged, cfg):
    m = evaluate("gold", "minidev", "t", timing="quick", paths=staged, cfg=cfg)
    assert m["overall"]["quick_ratio_median_mean"] > 0
    m = evaluate("gold", "minidev", "t2", timing="official", repeat=2, paths=staged, cfg=cfg)
    o = m["overall"]
    assert o["n_repeats"] == 2 and 50 <= o["rves_mean"] <= 112 and o["rves_std"] is not None
    assert len(read_jsonl(staged.run_dir("t2") / "timing_official_minidev_r1.jsonl")) == 3


def test_predictions_errors_and_tied(staged, cfg, tmp_path):
    # tied gold for question 0 that returns a different result than the official gold
    tied = staged.split_file("dev", "tied_gold")
    recs = json.loads(tied.read_text())
    recs[0]["SQL"] = "SELECT count(*) FROM orders"
    tied.write_text(json.dumps(recs))
    pred = tmp_path / "p.jsonl"
    pred.write_text("\n".join(json.dumps(r) for r in [
        {"qid": "dev_0", "sql": "SELECT 50"},             # matches the tied gold only
        {"qid": "dev_1", "sql": "SELEC broken"},          # syntax error
    ]) + "\n")                                           # dev_2 missing -> parse_fail
    m = evaluate(str(pred), "dev", "p", paths=staged, cfg=cfg)
    o = m["overall"]
    assert o["ex"] == 0.0 and o["ex_tied"] == pytest.approx(100 / 3)
    assert o["outcomes"] == {"wrong_result": 1, "syntax": 1, "parse_fail": 1}


def test_official_prediction_format_alignment(staged, tmp_path):
    ex = load_examples("dev", staged)
    off = tmp_path / "predict_dev.json"
    off.write_text(json.dumps({str(i): f"{e.gold_sql}\t----- bird -----\t{e.db_id}" for i, e in enumerate(ex)}))
    assert load_predictions(str(off), ex) == {e.qid: e.gold_sql for e in ex}
    bad = {str(i): f"SELECT 1\t----- bird -----\twrong_db" for i in range(len(ex))}
    off.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="db_id"):
        load_predictions(str(off), ex)


def test_evaluate_resumes(staged, cfg, monkeypatch):
    evaluate("gold", "dev", "r", paths=staged, cfg=cfg)
    import effsql.evaluate as ev

    def boom(*a, **k):
        raise AssertionError("should not execute again")

    monkeypatch.setattr(ev, "execute_two_pass", boom)
    m = evaluate("gold", "dev", "r", paths=staged, cfg=cfg)  # gold costs cached, eval file complete
    assert m["overall"]["ex"] == 100.0
    assert len(read_jsonl(staged.run_dir("r") / "eval_dev.jsonl")) == 3


def test_export_official(staged, tmp_path):
    info = export_official("gold", "minidev", tmp_path / "off", paths=staged)
    pred = json.loads((tmp_path / "off" / "predict.json").read_text())
    gold = (tmp_path / "off" / "gold.sql").read_text().splitlines()
    diff = [json.loads(l) for l in (tmp_path / "off" / "diff.jsonl").read_text().splitlines()]
    assert info["n"] == len(pred) == len(gold) == len(diff) == 3
    assert pred["0"].split("\t----- bird -----\t")[0] == gold[0].split("\t")[0]
    assert diff[0]["difficulty"] == "challenging"
    assert "--db_root_path" in info["ves_command"] and info["ves_command"].count("/dev/") == 1


def test_evaluate_never_imports_gpu_stack(staged, cfg):
    evaluate("gold", "dev", "g", paths=staged, cfg=cfg)
    assert "vllm" not in sys.modules and "torch" not in sys.modules
