"""End to end on the fixture: generate -> parse -> screen -> diversity -> rewrite -> ... -> summary."""

import pandas as pd
import pytest
from fakes import FakeBackend

from effsql.config import load_config
from effsql.data.schema import build_schema_cache
from effsql.data.stage import stage
from effsql.evaluate import evaluate
from effsql.gen.parse import parse_run
from effsql.gen.predict import predict
from effsql.gen.rewrite import rewrite
from effsql.gen.sample import generate
from effsql.io import read_json, read_jsonl
from effsql.pool.build import build_pool
from effsql.pool.diversity import assign_levels, diversity
from effsql.pool.screen import screen
from effsql.pool.summary import dpo_pair_counts, summarize
from effsql.pool.verify import agreement_stats, verify_pool

CFG = {"stage": {"disk_reserve_gb": 0}}


@pytest.fixture
def ready(fake_bird):
    for split in ("train", "dev"):
        stage(split, paths=fake_bird, data_cfg=CFG)
        build_schema_cache(split, paths=fake_bird, data_cfg=CFG)
    return fake_bird


@pytest.fixture
def exec_cfg():
    c = load_config("evaluate.yaml")["exec"]
    c["workers"] = 1
    return c


def test_assign_levels():
    # c=10: 5 is not a new level (ratio ok but gap < 50); 100, 160 and 1000 each start one
    assert assign_levels([1000, 0, 160, 5, 100], 10, 1.5, 50) == [3, 0, 2, 0, 1]
    assert assign_levels([], 10, 1.5, 50) == []


def test_full_pipeline(ready, exec_cfg):
    p, run, fb = ready, "pipe", FakeBackend()
    generate("train", run, fb, paths=p)
    stats = parse_run("train", run, paths=p)
    assert stats["n_completions"] == 30 and stats["extraction"]["parse_fail"] == 2  # "I am not sure." x 2 variants
    assert stats["n_gold_reproduced_by_model"] == 3

    s = screen("train", run, paths=p, exec_cfg=exec_cfg)
    d0 = diversity("train", run, 0, paths=p)
    recs0 = {r["qid"]: r for r in read_jsonl(p.run_dir(run) / "diversity_train_r0.jsonl")}
    assert recs0["train_0"]["diverse"]                          # count(*) vs self-join: two levels
    assert recs0["train_1"]["reason"] == "one_correct"         # books: model only reproduced gold
    assert d0["n_diverse"] == 1

    rw = rewrite(run, fb, paths=p)
    assert rw["n_flagged_questions"] == 2 and rw["n_requests_run"] == 2 * 4
    parse_run("train", run, paths=p)
    screen("train", run, paths=p, exec_cfg=exec_cfg)
    d1 = diversity("train", run, 1, paths=p)
    recs1 = {r["qid"]: r for r in read_jsonl(p.run_dir(run) / "diversity_train_r1.jsonl")}
    assert recs1["train_1"]["diverse"]                          # the cte rewrite adds a pricier level
    assert d1["n_diverse"] >= 2
    assert [s_["round"] for s_ in read_json(p.run_dir(run) / "diversity_summary_train.json")] == [0, 1]

    pool = build_pool("train", run, paths=p)
    assert set(pool["qid"]) == {"train_0", "train_1", "train_2"}
    gold = pool[pool["is_gold"]]
    assert len(gold) == 3 and gold["correct"].all()
    assert (gold["eff_score"].abs() < 1e-9).all() and (gold["rel_cost"] == 1.0).all()
    rw_rows = pool[pool["source"].str.startswith("rewrite:")]
    assert set(rw_rows["source"]) >= {"rewrite:optimized", "rewrite:cte"}
    wrong_rw = rw_rows[rw_rows["sql_original"] == "SELECT 1"]
    assert len(wrong_rw) and not wrong_rw["correct"].any()     # wrong rewrites are never trusted
    q0 = pool[(pool["qid"] == "train_0") & pool["correct"]].sort_values("eff_rank")
    assert q0["eff_rank"].tolist() == [1, 2] and q0["eff_percentile"].tolist() == [1.0, 0.0]
    assert q0["eff_score"].iloc[1] < 0                          # self-join is pricier than gold
    assert (p.run_dir(run) / "pool_train.parquet").exists() and (p.run_dir(run) / "pool_train.jsonl").exists()
    assert pool[pool["error_type"] == "parse_fail"]["correct"].eq(False).all()

    v = verify_pool("train", run, sample=5, paths=p,
                    timing_cfg={"quick": {"warmup": 0, "runs": 2, "timeout_s": 10, "cv_threshold": 0.2}})
    assert v["n_questions"] >= 1 and v["pairwise_agreement"]["all"]["n_pairs"] >= 1

    for split in ("dev",):
        for prompt in ("base", "efficient"):
            predict(split, prompt, run, fb, paths=p, limit=2)
    pred = p.run_dir(run) / "predictions_dev_base.jsonl"
    assert len(read_jsonl(pred)) == 2
    cfg = load_config("evaluate.yaml")
    cfg["exec"]["workers"] = 1
    m = evaluate(str(pred), "dev", f"{run}_dev_base", only_predicted=True, paths=p, cfg=cfg)
    assert m["overall"]["n"] == 2 and m["overall"]["outcomes"].get("syntax") == 1  # "studentz" table

    out = summarize("train", run, paths=p)
    assert out["pool"]["potential_pairs"]["efficiency_pairs"] >= 2
    assert out["diversity_rounds"][1]["n_diverse"] == d1["n_diverse"]
    assert f"{run}_dev_base/metrics_dev_none" in out["evaluations"]
    import sys

    assert "vllm" not in sys.modules and "torch" not in sys.modules  # the whole CPU path stays light


def test_screen_censors_runaway_candidates(ready, exec_cfg):
    p, run = ready, "cap"
    generate("train", run, FakeBackend(), paths=p)
    parse_run("train", run, paths=p)
    pool_cfg = load_config("pool.yaml")
    pool_cfg["candidate_step_cap_factor"] = 0.01  # cap = 10 steps: the self-join is censored
    screen("train", run, paths=p, exec_cfg=exec_cfg, pool_cfg=pool_cfg)
    recs = read_jsonl(p.run_dir(run) / "screen_train.jsonl")
    capped = [r for r in recs if r["steps_capped"]]
    assert capped and all(r["correct"] and r["vm_steps"] is None and r["steps_lower_bound"] for r in capped)


def test_dpo_pair_counts():
    df = pd.DataFrame({
        "qid": ["a"] * 4, "correct": [True, True, True, False], "vm_steps": [0, 10, 500, None],
        "error_type": [None, None, None, None],
    })
    # (0, 500) and (10, 500) are at different levels; (0, 10) is not (gap < 50)
    assert dpo_pair_counts(df, 1.5, 50, 10) == {"efficiency_pairs": 2, "correct_vs_incorrect_pairs": 3}


def test_agreement_stats():
    qs = [{"candidates": [{"vm_steps": 10, "median_s": 0.001}, {"vm_steps": 100, "median_s": 0.01},
                          {"vm_steps": 1000, "median_s": 0.02}]},
          {"candidates": [{"vm_steps": 50, "median_s": 0.03}, {"vm_steps": 60, "median_s": 0.02}]}]
    s = agreement_stats(qs, 1.5, 0.005)
    a = s["pairwise_agreement"]
    assert a["all"] == {"agree": 3, "n_pairs": 4, "rate": 0.75}
    assert a["step_ratio_ge_level"]["n_pairs"] == 3 and a["step_ratio_ge_level"]["rate"] == 1.0
    assert a["both_slower_than_min_time"]["n_pairs"] == 2
    assert s["spearman_per_question_mean"] == pytest.approx(1.0)
