"""Ablation runs: judge data from own answers only (with the results carried over), the untrained judge, and
skipping finished training and scoring."""

import importlib.util

import pytest
from test_judge import judge_setup  # noqa: F401 (fixture)

from effsql.config import load_config
from effsql.io import read_json
from effsql.judge.data import derive_judge_data, read_part
from effsql.judge.vote import exec_judge_data, read_exec

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))


def test_derive_judge_data_keeps_own_answers_and_results(judge_setup):  # noqa: F811
    p, jcfg = judge_setup
    exec_judge_data("jd", paths=p, jcfg=jcfg, workers=1)
    meta = derive_judge_data("jd", "jd_own", ("own",), paths=p)
    assert meta["derived_from"] == "jd" and meta["config"]["sources"] == ["own"]
    for part in ("train", "val"):
        src = {r["qid"]: r for r in read_part(p.run_dir("jd"), part)}
        sx = read_exec(p, "jd", part)
        dx = read_exec(p, "jd_own", part)
        for r in read_part(p.run_dir("jd_own"), part):
            assert r["cands"] and all(c["source"] == "own" for c in r["cands"])
            keep = [i for i, c in enumerate(src[r["qid"]]["cands"]) if c["source"] == "own"]
            assert [c["sql"] for c in r["cands"]] == [src[r["qid"]]["cands"][i]["sql"] for i in keep]
            assert dx[r["qid"]]["results"] == [sx[r["qid"]]["results"][i] for i in keep]
    # train_0 had one correct own answer and a correct pool one: with own only, no relative cost is left
    t0 = {r["qid"]: r for r in read_part(p.run_dir("jd_own"), "train")}["train_0"]
    assert all(c["cost"] is None for c in t0["cands"])
    assert meta["stats"]["train"]["n_own"] == meta["stats"]["train"]["n_candidates"]


@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")
def test_untrained_judge_and_skips(judge_setup, tmp_path):  # noqa: F811
    import torch
    from fakes import FakeBackend
    from test_sft import make_tiny_model

    from effsql.judge.candidates import generate_candidates
    from effsql.judge.score import done_path, read_scores, score_split
    from effsql.judge.train import load_judge, train_judge, untrained_judge
    from effsql.judge.vote import exec_candidates

    p, jcfg = judge_setup
    cfg = load_config("train.yaml")
    cfg["sft"].update(base_revision=None, lora={**cfg["sft"]["lora"], "r": 4, "alpha": 8})
    gen_cfg = load_config("generate.yaml")
    gen_cfg["model"] = {"id": str(make_tiny_model(tmp_path / "tiny")), "revision": None}
    kw = {"paths": p, "jcfg": jcfg, "cfg": cfg, "gen_cfg": gen_cfg, "device": "cpu"}

    m = untrained_judge("U", **kw)
    assert m["untrained"] and m["complete"] and m["init_from"] is None and m["with_results"]
    assert untrained_judge("U", **kw) == m  # already there
    with pytest.raises(RuntimeError):
        untrained_judge("U", with_results=False, **kw)
    model, head, tok, yes_no, _ = load_judge(p, "U", "cpu", cfg=cfg, gen_cfg=gen_cfg)
    lora_b = [v for k, v in model.state_dict().items() if "lora_B" in k]
    assert lora_b and all(torch.count_nonzero(v) == 0 for v in lora_b)  # the adapter changes nothing

    generate_candidates("cand", "dev", FakeBackend(), paths=p, jcfg=jcfg)
    exec_candidates("cand", "dev", paths=p, jcfg=jcfg, workers=1)
    score_split("U", "cand", "dev", (model, head, tok, yes_no), paths=p, jcfg=jcfg, cfg=cfg, device="cpu",
                with_results=True)
    assert done_path(p, "cand", "U", "dev").exists()
    s = read_scores(p, "cand", "U", "dev")
    assert set(s) == {"dev_0", "dev_1", "dev_2"} and all(x == 0.0 for r in s.values() for x in r["c"])

    exec_judge_data("jd", paths=p, jcfg=jcfg, workers=1)
    jc = {**jcfg, "train": {**jcfg["train"], "cost_weight": 0.0}}
    kw0 = {**kw, "jcfg": jc}
    first = train_judge("J0", "jd", with_results=True, **kw0)
    assert first["complete"] and first["train"]["cost_weight"] == 0.0
    head0 = torch.load(p.run_dir("J0") / "cost_head.pt", weights_only=False)["state"]
    assert all(torch.count_nonzero(v) == 0 for v in head0.values())  # no cost loss: the head never moves
    log_before = (p.run_dir("J0") / "train_log.jsonl").read_text()
    assert train_judge("J0", "jd", with_results=True, **kw0) == read_json(p.run_dir("J0") / "judge_meta.json")
    assert (p.run_dir("J0") / "train_log.jsonl").read_text() == log_before  # skipped, not retrained
    with pytest.raises(RuntimeError):
        train_judge("J0", "jd", with_results=True, **kw)  # different settings, same run name
