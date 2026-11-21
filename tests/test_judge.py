"""Cost-aware judge: training data, packing (equal to each candidate alone), the selection rule, and the
whole path on CPU with a tiny random model: train (and resume), candidates, labels, scores, selection."""

import importlib.util
import json

import numpy as np
import pytest
from fakes import FakeBackend

from effsql.config import load_config
from effsql.data.schema import build_schema_cache
from effsql.data.stage import stage
from effsql.io import append_jsonl, read_json, read_jsonl, write_json
from effsql.judge.candidates import (
    generate_candidates,
    label_candidates,
    read_candidates,
)
from effsql.judge.data import build_judge_data, cost_labels, norm_sql
from effsql.judge.model import JQuestion, auc, judge_metrics, pack, plan_steps
from effsql.judge.select import choose_delta, judge_index, vote_index

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))
torch_only = [pytest.mark.torch, pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")]


def judge_cfg(**data):
    c = load_config("judge.yaml")
    c["data"].update(own_run="own", pool_run="pool", export_run="exp", **data)
    c["candidates"].update(generator="gen", samples={"base": 3, "efficient": 2})
    c["train"].update(questions_per_step=1, questions_per_seq=1, save_every=1, eval_every=100, init_from=None)
    return c


def fake_runs(p):
    """Export (books held out), own answers and a pool for the fixture's three train questions."""
    write_json(p.run_dir("exp") / "export_meta.json", {"holdout_db_ids": ["books"], "val_qids": ["train_1"]})
    append_jsonl(p.run_dir("own") / "iters" / "000" / "rewards.jsonl", [
        {"qid": "train_0", "sql": ["SELECT count(*) FROM movies", "SELECT count(*) FROM movies",
                                   "SELECT count(*) FROM movies WHERE id > 3", None],
         "status": ["correct", "correct", "wrong_result", "no_sql"], "steps": [100, 100, None, None]},
        {"qid": "train_1", "sql": ["SELECT count(*) FROM books"], "status": ["correct"], "steps": [50]},
        {"qid": "train_2", "sql": ["SELECT max(id) FROM movies"], "status": ["foreign_constant"], "steps": [None]},
    ])
    import pandas as pd

    rows = [  # qid, sql, correct, quality, steps, is_gold
        ("train_0", "SELECT  count(*) FROM movies;", True, "ok", 100.0, False),   # same as an own answer
        ("train_0", "SELECT count(*) FROM movies a JOIN movies b ON a.id = b.id", True, "ok", 300.0, False),
        ("train_0", "SELECT count(*) FROM movies", True, "gold", 100.0, True),     # gold: never a candidate
        ("train_2", "SELECT max(id) FROM movies LIMIT 1", True, "fragile", 40.0, False),
        ("train_2", "SELECT max(title) FROM movies", False, None, None, False),
    ]
    df = pd.DataFrame([{"qid": q, "split": "train", "is_gold": g, "sql_original": s, "correct": c, "error_type": None,
                        "quality": ql, "vm_steps": st, "steps_capped": False} for q, s, c, ql, st, g in rows])
    df.to_parquet(p.run_dir("pool") / "pool_train.parquet")


def test_judge_data(fake_bird):
    p = fake_bird
    fake_runs(p)
    meta = build_judge_data("jd", paths=p, cfg=judge_cfg())
    train = {r["qid"]: r for r in read_jsonl(p.run_dir("jd") / "judge_train.jsonl")}
    val = read_jsonl(p.run_dir("jd") / "judge_val.jsonl")
    assert set(train) == {"train_0", "train_2"} and [r["qid"] for r in val] == ["train_1"]
    c0 = {norm_sql(c["sql"]): c for c in train["train_0"]["cands"]}
    assert len(c0) == 3  # duplicate own answer and the pool copy merged, gold left out
    assert c0["SELECT count(*) FROM movies"]["source"] == "own"
    assert c0["SELECT count(*) FROM movies WHERE id > 3"]["label"] == 0
    # two correct with steps 100 and 300: log2(200) and log2(400) around their mean
    assert c0["SELECT count(*) FROM movies"]["cost"] == pytest.approx(-0.5)
    assert c0["SELECT count(*) FROM movies a JOIN movies b ON a.id = b.id"]["cost"] == pytest.approx(0.5)
    c2 = {c["sql"]: c for c in train["train_2"]["cands"]}
    assert c2["SELECT max(id) FROM movies"]["label"] == 0  # foreign constant: not correct
    assert c2["SELECT max(id) FROM movies LIMIT 1"]["status"] == "lucky_match"
    assert all(c["cost"] is None for c in c2.values())
    assert meta["stats"]["train"]["n_questions"] == 2 and meta["stats"]["val"]["n_yes"] == 1
    assert cost_labels([{"sql": "a", "status": "correct", "steps": 5}], 100) == {}  # one correct: no cost label


def test_plan_and_pack_structure():
    P, Q1, Q2 = [1, 2, 3, 4], [5, 6], [7]
    q1 = JQuestion("a", "k", [P + Q1 + [9, 10], P + Q1 + [11]])
    q2 = JQuestion("b", "k", [P + Q2 + [12]])
    ids, pos, sq, sc, read = pack([q1, q2])
    assert ids == P + Q1 + [9, 10, 11] + Q2 + [12]
    assert pos == [0, 1, 2, 3, 4, 5, 6, 7, 6, 4, 5]
    assert read == [7, 8, 10] and sq[:4] == [0] * 4 and sc[6:9] == [1, 1, 2]
    one = JQuestion("c", "k", [P + [8]])  # a single candidate keeps its last token as its own
    assert pack([one])[0] == P + [8] and pack([one])[4] == [4]
    qs = [JQuestion(str(i), f"db{i % 2}", [P + [i, 20]]) for i in range(10)]
    steps = plan_steps(qs, per_seq=2, per_step=4, max_tokens=100, seed=0)
    assert sorted(i for s in steps for g in s for i in g) == list(range(10))
    assert all(len({qs[i].key for i in g}) == 1 for s in steps for g in s)


def test_selection_rule_and_metrics():
    cands = [{"sql": "a", "n": 2, "sources": ["greedy", "base"]}, {"sql": "b", "n": 5, "sources": ["base"]},
             {"sql": "c", "n": 1, "sources": ["efficient"]}]
    score = {"p": [0.90, 0.86, 0.40], "c": [0.2, -0.6, -1.8]}
    assert judge_index(cands, score, 0.0) == 0
    assert judge_index(cands, score, 0.05) == 1  # near the top p and cheaper
    assert judge_index(cands, score, 1.0) == 2   # everything allowed: the cheapest
    assert judge_index(cands, None, 0.05) == 0   # not scored: the greedy answer
    assert vote_index(cands) == 1
    sweep = {0.0: {"ex": 80.0, "cost_geomean": 0.95, "correct": [True] * 8 + [False] * 2},
             0.1: {"ex": 70.0, "cost_geomean": 0.90, "correct": [True] * 7 + [False] * 3},
             1.0: {"ex": 10.0, "cost_geomean": 0.50, "correct": [True] + [False] * 9}}
    assert choose_delta(sweep, 0.05) == 0.1  # not significantly below the best, cheaper
    assert auc(np.array([3.0, 1.0, 2.0]), np.array([1, 0, 0])) == 1.0
    m = judge_metrics([{"lp": [2.0, -1.0], "c": [-1.0, 1.0], "labels": [1, 0], "costs": [-0.6, None]},
                       {"lp": [0.5, 1.0], "c": [0.3, -0.3], "labels": [1, 1], "costs": [0.5, -0.5]}])
    assert m["top1"] == 1.0 and m["cost_pair_acc"] == 1.0 and m["accuracy"] == 1.0


@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")
def test_packed_forward_equals_each_candidate_alone(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from test_sft import make_tiny_model
    from transformers import AutoModelForCausalLM

    from effsql.judge.model import forward, new_cost_head

    path = make_tiny_model(tmp_path / "tiny")
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(str(path)),
                           LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.eval()
    head = new_cost_head(model.config.hidden_size, "cpu")
    torch.nn.init.normal_(head.weight)
    rng = np.random.default_rng(0)
    r = lambda n: [int(x) for x in rng.integers(0, 250, n)]
    P = r(30)
    qa, qb = P + r(6), P + r(4)
    group = [JQuestion("a", "k", [qa + r(5), qa + r(3), qa + r(7)]), JQuestion("b", "k", [qb + r(4), qb + r(2)])]
    with torch.no_grad():
        lp, c = forward(model, head, group, "cpu", (3, 4))
        alone = [forward(model, head, [JQuestion("x", "k", [ids])], "cpu", (3, 4)) for q in group for ids in q.ids]
    assert torch.allclose(lp, torch.cat([a[0] for a in alone]), atol=1e-4)
    assert torch.allclose(c, torch.cat([a[1] for a in alone]), atol=1e-4)


@pytest.fixture
def judge_setup(fake_bird, tmp_path):
    p = fake_bird
    c = {"stage": {"disk_reserve_gb": 0}}
    for split in ("train", "dev"):
        stage(split, paths=p, data_cfg=c)
        build_schema_cache(split, paths=p, data_cfg=c)
    fake_runs(p)
    jcfg = judge_cfg()
    build_judge_data("jd", paths=p, cfg=jcfg)
    # the generator's greedy answers (as `python -m effsql.train predict` writes them)
    append_jsonl(p.run_dir("gen") / "predictions_val.jsonl", [{"qid": "train_1", "sql": "SELECT count(*) FROM books"}])
    append_jsonl(p.run_dir("gen") / "predictions_dev.jsonl", [
        {"qid": "dev_0", "sql": "SELECT count(*) FROM customers"}, {"qid": "dev_1", "sql": "SELECT sum(amount) FROM orders"},
        {"qid": "dev_2", "sql": "SELECT count(*) FROM studentz"}])
    return p, jcfg


def test_candidates_and_labels(judge_setup):
    p, jcfg = judge_setup
    st = generate_candidates("cand", "dev", FakeBackend(), paths=p, jcfg=jcfg)
    assert st["n_questions"] == 3 and st["n_greedy_missing"] == 0
    c = {r["qid"]: r for r in read_candidates(p, "cand", "dev")}
    assert c["dev_0"]["cands"][0]["sources"][0] == "greedy" and c["dev_0"]["cands"][0]["n"] == 6  # 1 + 3 + 2
    generate_candidates("cand", "val", FakeBackend(), paths=p, jcfg=jcfg)
    v = read_candidates(p, "cand", "val")
    assert [r["qid"] for r in v] == ["train_1"]
    exec_cfg = load_config("evaluate.yaml")["exec"]
    exec_cfg["workers"] = 1
    label_candidates("cand", "dev", paths=p, jcfg=jcfg, exec_cfg=exec_cfg)
    label_candidates("cand", "dev", paths=p, jcfg=jcfg, exec_cfg=exec_cfg)  # resumed: nothing left
    labs = {r["qid"]: r for r in read_jsonl(p.run_dir("cand") / "labels_dev.jsonl")}
    assert len(labs) == 3
    assert labs["dev_0"]["labels"][0]["correct"] and labs["dev_0"]["labels"][0]["rel_cost"] == pytest.approx(1.0)
    assert not labs["dev_2"]["labels"][0]["correct"] and labs["dev_2"]["labels"][0]["status"] != "correct"


@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")
def test_train_resume_score_and_select(judge_setup, tmp_path):
    import torch
    from test_sft import make_tiny_model

    from effsql.judge.score import read_scores, score_split
    from effsql.judge.select import select
    from effsql.judge.train import load_judge, train_judge

    p, jcfg = judge_setup
    cfg = load_config("train.yaml")
    cfg["sft"].update(base_revision=None, lora={**cfg["sft"]["lora"], "r": 4, "alpha": 8})
    gen_cfg = load_config("generate.yaml")
    gen_cfg["model"] = {"id": str(make_tiny_model(tmp_path / "tiny")), "revision": None}
    kw = {"paths": p, "jcfg": jcfg, "cfg": cfg, "gen_cfg": gen_cfg, "device": "cpu"}
    full = train_judge("J", "jd", **kw)
    assert full["complete"] and full["n_steps"] == 2 and full["val_final"]["n_candidates"] == 1
    log = read_jsonl(p.run_dir("J") / "train_log.jsonl")
    assert len(log) == 2 and all(r["cost_mse"] is None or r["cost_mse"] >= 0 for r in log)
    part = train_judge("J2", "jd", max_steps=1, **kw)
    assert not part["complete"]
    train_judge("J2", "jd", **kw)  # resumes at step 1
    a = torch.load(p.run_dir("J") / "cost_head.pt", weights_only=False)["state"]
    b = torch.load(p.run_dir("J2") / "cost_head.pt", weights_only=False)["state"]
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)

    for split in ("val", "dev"):
        generate_candidates("cand", split, FakeBackend(), paths=p, jcfg=jcfg)
    exec_cfg = load_config("evaluate.yaml")["exec"]
    exec_cfg["workers"] = 1
    for split in ("val", "dev"):
        label_candidates("cand", split, paths=p, jcfg=jcfg, exec_cfg=exec_cfg)
    model, head, tok, yes_no, _ = load_judge(p, "J", "cpu", cfg=cfg, gen_cfg=gen_cfg)
    for split in ("val", "dev"):
        score_split("J", "cand", split, (model, head, tok, yes_no), paths=p, jcfg=jcfg, cfg=cfg, device="cpu")
    s = read_scores(p, "cand", "J", "dev")
    assert set(s) == {"dev_0", "dev_1", "dev_2"} and all(0 <= x <= 1 for r in s.values() for x in r["p"])
    res = select("cand", "J", "J_sel", paths=p, jcfg=jcfg, splits=("val", "dev"))
    assert res["delta"] in [float(d) for d in jcfg["select"]["deltas"]]
    for run in ("J_sel", "J_sel_p", "J_sel_vote"):
        preds = read_jsonl(p.run_dir(run) / "predictions_dev.jsonl")
        assert len(preds) == 3 and read_json(p.run_dir(run) / "adapter_meta.json")["complete"]
    assert "delta chosen on validation" in (p.run_dir("J_sel") / "selection.txt").read_text()
    assert json.loads((p.run_dir("J_sel") / "selection.json").read_text())["splits"]["dev"]["oracle"]["ex"] is not None
