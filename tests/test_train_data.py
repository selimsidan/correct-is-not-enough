"""Training data export: arm builders on a hand-made pool, prompts, and one end-to-end export."""

import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fakes import FakeBackend

from effsql.config import load_config
from effsql.data.schema import build_schema_cache
from effsql.data.stage import stage
from effsql.gen.parse import parse_run
from effsql.gen.prompts import build_question_prompt, estimate_tokens, load_templates
from effsql.gen.sample import generate
from effsql.io import read_json
from effsql.pool.build import build_pool
from effsql.pool.diversity import diversity
from effsql.pool.quality import quality
from effsql.pool.screen import screen
from effsql.train.data import (build_dpo, build_plain, build_tagged, build_weighted, check_train_only,
                               format_tag, holdout_dbs, inference_tag, question_pairs, read_export, sft_targets,
                               tag_value, target_completion, train_export, train_messages, weighted_probs)


def cand(qid, cid, *, gold=False, correct=True, eff=0.0, level=0, capped=False, error=None, db="db1",
         sql=None, split="train", gold_steps=1000):
    # steps consistent with eff = log2((gold + 10) / (steps + 10)); censored rows carry a lower bound
    steps = (gold_steps + 10) / 2 ** eff - 10 if correct else None
    return {"qid": qid, "split": split, "db_id": db, "cand_id": cid, "source": "gold" if gold else "sample",
            "is_gold": gold, "sql_original": sql or f"SELECT {cid}", "correct": correct, "steps_capped": capped,
            "eff_score": eff if correct and not capped else (eff if capped else None),
            "vm_steps": None if capped else steps, "steps_lower_bound": steps if capped else None,
            "efficiency_level": level if correct else None, "error_type": error}


@pytest.fixture
def pool():
    rows = [
        # q1: gold, a cheaper candidate (level 0), a pricier one (level 2), a censored one, two wrong ones
        cand("train_1", "a", gold=True, eff=0.0, level=1),
        cand("train_1", "b", eff=1.37, level=0),
        cand("train_1", "c", eff=-2.0, level=2),
        cand("train_1", "d", eff=-6.0, level=3, capped=True),
        cand("train_1", "e", correct=False, error="wrong_result"),
        cand("train_1", "f", correct=False, error="parse_fail", sql=""),
        cand("train_1", "g", correct=False, error="syntax"),
        # q2: gold only
        cand("train_2", "h", gold=True, db="db2"),
        # q3: gold plus an equally cheap candidate and one wrong one
        cand("train_3", "i", gold=True, db="db3"),
        cand("train_3", "j", eff=0.04, level=0, db="db3"),
        cand("train_3", "k", correct=False, error="runtime", db="db3"),
    ]
    return pd.DataFrame(rows)


QIDS = ["train_1", "train_2", "train_3"]


def test_weighted_probs():
    eff = np.array([0.0, 1.0, -1.0])
    assert weighted_probs(eff, 0.0) == pytest.approx([1 / 3] * 3)
    p = weighted_probs(eff, 1.0)  # proportional to 2^eff: 1, 2, 0.5
    assert p.sum() == pytest.approx(1.0) and p == pytest.approx(np.array([1, 2, 0.5]) / 3.5)
    assert weighted_probs(np.array([3.0, -3.0]), 1e4) == pytest.approx([1.0, 0.0])  # no overflow, best only


def test_targets_exclude_incorrect_and_censored(pool):
    t = sft_targets(pool)
    assert set(t["cand_id"]) == {"a", "b", "c", "h", "i", "j"}


def test_plain_is_gold_one_per_question_per_epoch(pool):
    recs = build_plain(sft_targets(pool), QIDS, "train", 2)
    assert len(recs) == 6 and all(r["source"] == "gold" for r in recs)
    assert Counter((r["epoch"], r["qid"]) for r in recs) == {(e, q): 1 for e in (0, 1) for q in QIDS}


def test_weighted_draws(pool):
    t = sft_targets(pool)
    recs = build_weighted(t, QIDS, "train", 3, 1.0, seed=0)
    assert Counter((r["epoch"], r["qid"]) for r in recs) == {(e, q): 1 for e in range(3) for q in QIDS}
    assert all(r["cand_id"] in set(t["cand_id"]) for r in recs)
    # frequencies follow p: on q1 the cheapest candidate b (eff 1.37) dominates, c (eff -2) is rare
    many = build_weighted(t, ["train_1"], "train", 4000, 1.0, seed=0)
    freq = Counter(r["cand_id"] for r in many)
    p = dict(zip(["a", "b", "c"], weighted_probs(np.array([0.0, 1.37, -2.0]), 1.0)))
    for k in "abc":
        assert freq[k] / 4000 == pytest.approx(p[k], abs=0.03)
    # a question's draws do not depend on which other questions are exported
    alone = build_weighted(t, ["train_3"], "train", 3, 1.0, seed=0)
    assert [r["cand_id"] for r in alone] == [r["cand_id"] for r in recs if r["qid"] == "train_3"]


def test_tagged_tags_and_no_repeats(pool):
    t = sft_targets(pool)
    recs = build_tagged(t, ["train_1"], "train", 3, clip=1.0, decimals=1, seed=0)
    assert sorted(r["cand_id"] for r in recs) == ["a", "b", "c"]  # 3 epochs, 3 candidates, no repeat
    tags = {r["cand_id"]: (r["tag"], r["tag_text"]) for r in recs}
    assert tags == {"a": (0.0, "+0.0"), "b": (1.0, "+1.0"), "c": (-1.0, "-1.0")}  # clipped to [-1, 1]
    assert format_tag(tag_value(-0.01, 3, 1), 1) == "+0.0" and format_tag(tag_value(1.37, 3, 1), 1) == "+1.4"
    cycled = build_tagged(t, ["train_2"], "train", 2, clip=3, decimals=1, seed=0)
    assert [r["cand_id"] for r in cycled] == ["h", "h"]  # fewer candidates than epochs: cycle


def test_dpo_pairs(pool):
    eff, corr = question_pairs(pool[pool["qid"] == "train_1"].sort_values("cand_id").reset_index(drop=True), 2.0, 50)
    g = pool[pool["qid"] == "train_1"].sort_values("cand_id").reset_index(drop=True)
    ids = g["cand_id"].tolist()
    eff_ids = {(ids[i], ids[j]) for i, j in eff}
    # chosen never censored; rejected may be censored (on its lower bound); at least 2x and 50 steps apart
    assert eff_ids == {("b", "a"), ("b", "c"), ("b", "d"), ("a", "c"), ("a", "d"), ("c", "d")}
    # correctness: chosen = counted correct not clearly pricier than the cheapest (b); rejected = incorrect with SQL
    assert {(ids[i], ids[j]) for i, j in corr} == {("b", "e"), ("b", "g")}

    recs = build_dpo(pool, QIDS, "train", per_question=4, efficiency_share=0.5, seed=0)
    q1 = [r for r in recs if r["qid"] == "train_1"]
    assert len(q1) == 4 and Counter(r["pair_type"] for r in q1) == {"efficiency": 2, "correctness": 2}
    assert len({r["rejected_cand"] for r in q1}) == 4  # each rejected answer used once
    for r in recs:
        assert r["chosen_cand"] not in {"d", "e", "f", "g", "k"}
        if r["pair_type"] == "efficiency":
            assert r["cost_ratio"] >= 2.0
        else:
            assert r["rejected_error"] not in (None, "parse_fail")
    assert not [r for r in recs if r["qid"] == "train_2"]  # gold only: no pairs
    q3 = [r for r in recs if r["qid"] == "train_3"]
    assert [r["pair_type"] for r in q3] == ["correctness"]  # i and j cost about the same: no efficiency pair
    # efficiency_share 1: only 3 distinct pricier rejected (a, c, d) exist, the 4th slot is topped up
    only_eff = build_dpo(pool, ["train_1"], "train", per_question=4, efficiency_share=1.0, seed=0)
    assert Counter(r["pair_type"] for r in only_eff) == {"efficiency": 3, "correctness": 1}


def test_efficiency_pairs_use_the_pair_cost_ratio_not_levels():
    g = pd.DataFrame([
        cand("train_9", "a", gold=True, eff=0.0, level=0),
        cand("train_9", "b", eff=0.8, level=1),        # a different level but only 1.74x cheaper than a
        cand("train_9", "c", eff=1.2, level=1),        # 2.3x cheaper than a
        cand("train_9", "w", correct=False, error="wrong_result"),
    ])
    eff, corr = question_pairs(g, 2.0, 50)
    ids = g["cand_id"].tolist()
    assert {(ids[i], ids[j]) for i, j in eff} == {("c", "a")}
    assert {(ids[i], ids[j]) for i, j in corr} == {("b", "w"), ("c", "w")}  # a is clearly pricier than c
    assert {(ids[i], ids[j]) for i, j in question_pairs(g, 1.5, 50)[0]} == {("b", "a"), ("c", "a")}
    tiny = pd.DataFrame([cand("train_8", "x", gold=True, eff=0.0, gold_steps=30),
                         cand("train_8", "y", eff=1.5, gold_steps=30)])  # 2.8x cheaper but only 26 steps fewer
    assert question_pairs(tiny, 2.0, 50)[0] == []


def test_holdout_and_leakage(pool):
    dbs = [f"db{i}" for i in range(40)]
    h = holdout_dbs(dbs, 0.05, 0)
    assert len(h) == 2 and set(h) <= set(dbs) and h == holdout_dbs(list(reversed(dbs)), 0.05, 0)
    assert holdout_dbs(["only"], 0.05, 0) == []
    assert len(holdout_dbs(["a", "b"], 0.05, 0)) == 1
    with pytest.raises(ValueError, match="not train data"):
        check_train_only(pd.concat([pool, pd.DataFrame([cand("dev_7", "z", gold=True, split="dev")])]))


def test_inference_tag(pool):
    t = sft_targets(pool)
    # best eff per question: 1.37, 0.0, 0.04 -> 0.9 quantile about 1.10
    assert inference_tag(t, QIDS, 0.9, 3.0, 1) == 1.1
    assert inference_tag(t, QIDS, 1.0, 1.0, 1) == 1.0


SCHEMA = {"tables": [{"ddl": "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", "row_count": 3}]}
EXAMPLE = SimpleNamespace(db_id="db1", question="How many rows?", evidence="")


def test_train_prompts_match_predict_and_tag_line():
    tpl = load_templates()
    base_pred = build_question_prompt(tpl, "base", EXAMPLE, SCHEMA, 12000, estimate_tokens)
    for arm in ("plain", "weighted"):
        assert train_messages(tpl, arm, EXAMPLE, SCHEMA, 12000, estimate_tokens) == base_pred
    tagged, _, _ = train_messages(tpl, "tagged", EXAMPLE, SCHEMA, 12000, estimate_tokens, tag="+1.4")
    assert tagged[1] == base_pred[0][1]  # same user message
    b_lines, t_lines = base_pred[0][0]["content"].splitlines(), tagged[0]["content"].splitlines()
    extra = [ln for ln in t_lines if ln not in b_lines]
    assert len(extra) == 1 and extra[0].startswith("- Efficiency target: +1.4.")
    assert [ln for ln in t_lines if ln in b_lines] == b_lines
    with pytest.raises(ValueError, match="needs an efficiency tag"):
        train_messages(tpl, "tagged", EXAMPLE, SCHEMA, 12000, estimate_tokens)
    with pytest.raises(ValueError, match="takes no efficiency tag"):
        build_question_prompt(tpl, "base", EXAMPLE, SCHEMA, 12000, estimate_tokens, tag="+1.0")
    assert target_completion(" SELECT 1 \n") == "```sql\nSELECT 1\n```"


def test_export_end_to_end(fake_bird):
    p, run = fake_bird, "pool"
    cfg_stage = {"stage": {"disk_reserve_gb": 0}}
    stage("train", paths=p, data_cfg=cfg_stage)
    build_schema_cache("train", paths=p, data_cfg=cfg_stage)
    exec_cfg = load_config("evaluate.yaml")["exec"]
    exec_cfg["workers"] = 1
    generate("train", run, FakeBackend(), paths=p)
    parse_run("train", run, paths=p)
    screen("train", run, paths=p, exec_cfg=exec_cfg)
    diversity("train", run, 0, paths=p)
    quality("train", run, paths=p, exec_cfg=exec_cfg)
    build_pool("train", run, paths=p)

    meta = train_export(run, "tr", paths=p)
    rd = p.run_dir("tr")
    assert meta["n_questions"]["train"] + meta["n_questions"]["val"] == 3
    assert len(meta["holdout_db_ids"]) == 1  # two databases: one held out
    held = set(meta["holdout_db_ids"])
    for arm in ("plain", "weighted", "tagged"):
        tr, va = read_export(rd, arm, "train"), read_export(rd, arm, "val")
        assert len(tr) == 2 * meta["n_questions"]["train"] and len(va) == meta["n_questions"]["val"]
        assert all(r["db_id"] not in held for r in tr) and all(r["db_id"] in held for r in va)
    st = meta["stats"]["plain_train"]
    assert st["prompt_tokens_coverage"] == 1.0 and st["total_tokens_estimate"] > 0
    dpo = read_export(rd, "dpo", "train") + read_export(rd, "dpo", "val")
    assert {r["pair_type"] for r in dpo} >= {"efficiency"}  # movies: count(*) vs self-join
    assert read_json(rd / "export_meta.json")["pool_sha256"]
    assert (rd / "train_config.yaml").exists() and (rd / "run_meta.json").exists()

    abl = train_export(run, "tr_b0", paths=p, arms=["weighted"], beta=0.0)
    assert abl["export_cfg"]["weighted"]["beta"] == 0.0 and set(abl["files"]) == {"weighted_train", "weighted_val"}
    assert "torch" not in sys.modules and "transformers" not in sys.modules
