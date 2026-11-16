"""RL with an execution reward: reward, advantages, clipped loss, and two iterations end to end on the tiny
model (a scripted sampler stands in for vLLM), with resume and the pilot projection."""

import importlib.util

import pytest

pytestmark = [pytest.mark.torch, pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft")),
    reason="training libraries not installed (pip install -e .[train])")]

from test_sft import adapter_weights, setup  # noqa: F401  (setup is a fixture)

from effsql.io import read_json, read_jsonl
from effsql.train.data import target_completion
from effsql.train.rl import (
    advantages,
    answer_chunks,
    focus_questions,
    gen_iter,
    iter_dir,
    ppo_terms,
    reward,
    rl_settings,
    run_rl,
    train_iter,
)
from effsql.train.sft import load_tokenizer, train_sft

RW = {"eff_weight": 0.5, "eff_clip": 2.0, "smooth_steps": 100}

# Scripted answers per question: (sql or plain text, finish reason).
ANSWERS = {
    "How many movies?": [("SELECT count(*) FROM movies", "stop"),
                         ("SELECT count(*) FROM movies a JOIN movies b ON a.id = b.id", "stop"),  # correct, pricier
                         ("SELECT count(*) FROM movies WHERE id > 3", "stop"),                   # wrong
                         ("I do not know.", "stop")],                                            # no SQL
    "How many books?": [("SELECT count(*) FROM books", "stop")] * 4,                              # no signal
    "Max id?": [("SELECT max(id) FROM movies", "stop"),
                ("SELECT max(id) FROM movies LIMIT 1", "stop"),            # LIMIT without ORDER BY: not allowed
                ("SELECT max(id) FROM movies WHERE id >= 0", "stop"),      # a constant found nowhere
                ("SELECT max(id) FROM mov", "length")],                    # cut off
}


def test_reward_and_advantages():
    assert reward("wrong_result", None, 1000, RW) == 0.0
    assert reward("first_row_pick", 10, 1000, RW) == 0.0
    assert reward("correct", 1000, 1000, RW) == pytest.approx(1.0)
    assert reward("correct", 0, 1e9, RW) == pytest.approx(1.5)          # clipped at 4x cheaper
    assert reward("correct", 1e9, 0, RW) == pytest.approx(0.5)          # clipped at 4x costlier
    assert reward("correct", None, 1000, RW) == pytest.approx(0.5)      # no count: the costliest
    assert reward("correct", 5, 30, RW) == pytest.approx(1 + 0.25 * __import__("math").log2(130 / 105))
    assert advantages([1.0, 1.0, 1.0]) is None and advantages([0.7]) is None
    assert advantages([1.0, 0.0, 0.5, 0.5]) == pytest.approx([0.5, -0.5, 0.0, 0.0])
    assert answer_chunks([300, 300, 1500, 100], 1000) == [[0, 1], [2], [3]]


def test_ppo_terms():
    import torch

    new = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss, clipped, kl = ppo_terms(new, new.detach().clone(), 0.5, 0.2)
    loss.backward()
    assert float(loss) == pytest.approx(-1.0) and int(clipped) == 0 and float(kl) == 0.0
    assert torch.allclose(new.grad, torch.tensor([-0.5, -0.5]))  # on-policy: -advantage per token
    far = torch.tensor([0.0], requires_grad=True)                  # ratio e = 2.7 > 1.2 with a positive advantage
    loss, clipped, _ = ppo_terms(far, torch.tensor([-1.0]), 1.0, 0.2)
    loss.backward()
    assert int(clipped) == 1 and float(loss) == pytest.approx(-1.2) and float(far.grad) == 0.0


def make_sampler(tok, calls):
    def sampler(prompts, seeds):
        calls.append(len(prompts))
        out = []
        for ids in prompts:
            q = tok.decode(ids).rsplit("Question: ", 1)[1].split("<|im_end|>", 1)[0].split("\n", 1)[0].strip()
            answers = []
            for text, finish in ANSWERS[q]:
                body = target_completion(text) if text.upper().startswith("SELECT") else text
                if finish == "length":
                    body = body[:12]
                answers.append({"token_ids": tok(body, add_special_tokens=False)["input_ids"], "text": body,
                                "finish_reason": finish, "vllm_logp": None})
            out.append(answers)
        return out
    return sampler


def test_rl_end_to_end(setup):  # noqa: F811 (the fixture from test_sft)
    import torch

    p, cfg, gen_cfg = setup
    train_sft("gold", "exp", "S", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu")
    cfg["rl"].update(iterations=2, questions_per_iter=2, samples=4, questions_per_step=1, lr=1e-3, snapshot_every=1,
                     max_answer_tokens_per_seq=60,  # several sequences per question: exercises the split
                     pilot={"iterations": 1, "questions_per_iter": 2})
    tok = load_tokenizer(gen_cfg["model"])
    calls, steps = [], []

    def runner(step, run, k):
        steps.append((step, run, k))
        if step == "gen":
            gen_iter(run, k, paths=p, cfg=cfg, gen_cfg=gen_cfg, sampler=make_sampler(tok, calls))
        else:
            train_iter(run, k, paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu")

    s = run_rl("R", "S", paths=p, cfg=cfg, gen_cfg=gen_cfg, workers=1, step_runner=runner)
    meta = read_json(p.run_dir("R") / "rl_meta.json")
    assert meta["arm"] == "gold" and meta["holdout_db_ids"] == [] and len(meta["questions"]) == 3
    assert s["iterations_done"] == 2 and [x for x, _, _ in steps] == ["gen", "train", "gen", "train"]

    rows = {r["qid"]: r for k in range(2) for r in read_jsonl(iter_dir(p.run_dir("R"), k) / "rewards.jsonl")}
    assert set(rows) == {"train_0", "train_1", "train_2"}
    m = rows["train_0"]
    assert m["status"] == ["correct", "correct", "wrong_result", "no_sql"]
    assert m["reward"][0] > m["reward"][1] > 0 and m["reward"][2:] == [0.0, 0.0]
    assert rows["train_2"]["status"] == ["correct", "first_row_pick", "foreign_constant", "cut_off"]
    assert rows["train_2"]["reward"][1:] == [0.0, 0.0, 0.0]

    gen = {g["qid"]: g for k in range(2) for g in read_jsonl(iter_dir(p.run_dir("R"), k) / "gen.jsonl")}
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    assert gen["train_0"]["answers"][0]["token_ids"][-1] == eot    # the end of turn the sampler stopped on
    assert gen["train_2"]["answers"][3]["token_ids"][-1] != eot    # a cut-off answer has none

    log = read_jsonl(p.run_dir("R") / "rl_log.jsonl")
    assert [r["iteration"] for r in log] == [0, 1]
    assert sum(r["n_questions_with_signal"] for r in log) == 2      # books: every answer scores the same
    assert all(r["approx_kl"] is not None for r in log if r["n_steps"])
    am = read_json(p.run_dir("R") / "adapter_meta.json")
    assert am["complete"] and am["arm"] == "gold" and am["method"] == "rl-gold" and am["export_run"] == "exp"
    snap = read_json(p.run_dir("R_it1") / "adapter_meta.json")   # a snapshot after iteration 1 of 2
    assert snap["complete"] and snap["iterations"] == 1 and am["iterations"] == 2
    assert not (p.run_dir("R_it2", create=False)).exists()       # the last iteration is the run itself
    a, b = adapter_weights(p.run_dir("R")), adapter_weights(p.run_dir("S"))
    assert any(not torch.allclose(a[k], b[k]) for k in a)            # RL moved the adapter

    n = len(steps)
    run_rl("R", "S", paths=p, cfg=cfg, gen_cfg=gen_cfg, workers=1, step_runner=runner)
    assert len(steps) == n                                           # resume: nothing is redone
    other = {**cfg, "rl": {**cfg["rl"], "lr": 5e-4}}
    with pytest.raises(RuntimeError, match="new run name"):
        run_rl("R", "S", paths=p, cfg=other, gen_cfg=gen_cfg, workers=1, step_runner=runner)

    # Efficiency phase: questions where the model's correct answers differed in cost (movies: count(*)
    # and the self-join), or where the pool knows a query at most 0.9 x gold (none in the fixture).
    f = {"pool_run": "pool", "max_rel_cost": 0.9, "history_runs": ["R"], "min_spread": 1.1}
    assert focus_questions(p, f, {"train_0", "train_1", "train_2"}) == {"train_0"}
    assert focus_questions(p, {**f, "history_runs": []}, {"train_0", "train_1", "train_2"}) == set()

    pilot = run_rl("P", "S", pilot=True, paths=p, cfg=cfg, gen_cfg=gen_cfg, workers=1, step_runner=runner)
    assert pilot["iterations_done"] == 1 and pilot["projection"]["iterations"] == 2
    assert pilot["projection"]["hours"] > 0


def test_settings_override():
    from effsql.config import load_config

    cfg = load_config("train.yaml")
    base, eff = rl_settings(cfg, False), rl_settings(cfg, False, "rl_eff")
    assert base["questions"] == "all" and eff["questions"] == "focus" and eff["samples"] == 16
    assert eff["reward"] == {**base["reward"], "eff_clip": 1.0} and eff["lr"] == base["lr"]
    assert reward("correct", 0, 1e9, eff["reward"]) == pytest.approx(1.9)       # 2x cheaper and more
    assert reward("correct", 1e9, 0, eff["reward"]) == pytest.approx(0.1)       # 2x costlier and more
    assert rl_settings(cfg, True, "rl_eff")["iterations"] == cfg["rl"]["pilot"]["iterations"]
