"""SFT: examples, training, resume, continuation, pilot, prediction with an adapter, validation scoring
and the method selection rule. Uses a tiny random Qwen2 model and an offline tokenizer (no downloads)."""

import importlib.util
import json

import pytest

# torch is imported only inside the tests (never at collection), and these tests run last (conftest):
# other tests check that CPU commands do not import torch.
pytestmark = [pytest.mark.torch, pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft")),
    reason="training libraries not installed (pip install -e .[train])")]

from fakes import FakeBackend  # noqa: E402

from effsql.config import load_config  # noqa: E402
from effsql.data.schema import build_schema_cache  # noqa: E402
from effsql.data.stage import stage  # noqa: E402
from effsql.evaluate import EVAL_VERSION, evaluate  # noqa: E402
from effsql.gen.parse import parse_run  # noqa: E402
from effsql.gen.prompts import load_templates  # noqa: E402
from effsql.gen.sample import generate  # noqa: E402
from effsql.io import append_jsonl, read_json, read_jsonl  # noqa: E402
from effsql.pool.build import build_pool  # noqa: E402
from effsql.pool.diversity import diversity  # noqa: E402
from effsql.pool.quality import quality  # noqa: E402
from effsql.pool.screen import screen  # noqa: E402
from effsql.train.data import read_export, train_export, train_messages  # noqa: E402
from effsql.train.predict_sft import predict_sft  # noqa: E402
from effsql.train.select import mcnemar_p, select_method  # noqa: E402
from effsql.train.sft import (END_OF_TURN, TrainExample, build_examples, group_losses,  # noqa: E402
                              load_tokenizer, make_counter, packed_len, plan_steps, prompt_text,
                              shared_prefix_len, train_sft)

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def make_tiny_model(path):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<|im_start|>", END_OF_TURN, "<|endoftext|>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())  # byte-level: every text round-trips exactly
    vocab = {k: i for i, k in enumerate(alphabet + specials)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token=END_OF_TURN, pad_token="<|endoftext|>",
                                  additional_special_tokens=specials, chat_template=TEMPLATE)
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    return path


@pytest.fixture
def setup(fake_bird, tmp_path):
    p, pool_run = fake_bird, "pool"
    c = {"stage": {"disk_reserve_gb": 0}}
    for split in ("train", "dev"):
        stage(split, paths=p, data_cfg=c)
        build_schema_cache(split, paths=p, data_cfg=c)
    exec_cfg = load_config("evaluate.yaml")["exec"]
    exec_cfg["workers"] = 1
    generate("train", pool_run, FakeBackend(), paths=p)
    parse_run("train", pool_run, paths=p)
    screen("train", pool_run, paths=p, exec_cfg=exec_cfg)
    diversity("train", pool_run, 0, paths=p)
    quality("train", pool_run, paths=p, exec_cfg=exec_cfg)
    build_pool("train", pool_run, paths=p)

    cfg = load_config("train.yaml")
    cfg["export"]["validation"]["fraction"] = 0.0  # all three questions train (val tested separately)
    train_export(pool_run, "exp", paths=p, cfg=cfg)
    cfg["sft"].update(grad_accum=1, save_every=1, eval_every=100, base_revision=None,
                      lora={**cfg["sft"]["lora"], "r": 4, "alpha": 8})
    gen_cfg = load_config("generate.yaml")
    gen_cfg["model"] = {"id": str(make_tiny_model(tmp_path / "tiny")), "revision": None}
    return p, cfg, gen_cfg


def adapter_weights(run_dir):
    from safetensors.torch import load_file

    return load_file(str(run_dir / "adapter" / "adapter_model.safetensors"))


def test_examples_use_the_prediction_prompt_and_end_of_turn(setup):
    p, cfg, gen_cfg = setup
    tok = load_tokenizer(gen_cfg["model"])
    recs = read_export(p.run_dir("exp"), "gold", "train")
    ex, stats = build_examples(recs, "gold", tok, p, 12000)
    assert stats["n_examples"] == len(recs) == 6 and stats["n_skipped_too_long"] == 0
    from effsql.data.bird import load_examples
    from effsql.data.schema import load_schema

    e = {x.qid: x for x in load_examples("train", p)}[ex[0].qid]
    msgs, _, _ = train_messages(load_templates(), "gold", e, load_schema("train", e.db_id, p), 12000, make_counter(tok))
    prompt = tok.decode(ex[0].input_ids[:-ex[0].n_target])
    assert prompt == prompt_text(tok, msgs) and prompt.endswith("<|im_start|>assistant\n")
    target = tok.decode(ex[0].input_ids[-ex[0].n_target:])
    assert target.startswith("```sql\n") and target.endswith("```" + END_OF_TURN)
    _, s2 = build_examples(recs, "gold", tok, p, 50)  # a budget no prompt fits: skipped and counted
    assert s2["n_examples"] == 0 and s2["n_skipped_too_long"] == 6


def test_group_losses_equal_losses_alone(tmp_path):
    """A shared-prefix group gives each question exactly its loss alone, and the same gradients."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    path = make_tiny_model(tmp_path / "tiny")
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32),
                           LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                                      task_type="CAUSAL_LM"))
    for n, w in model.named_parameters():  # LoRA B starts at zero; make the adapter matter
        if "lora_B" in n:
            torch.nn.init.normal_(w, std=0.1)
    g = torch.Generator().manual_seed(0)
    prefix = torch.randint(0, 200, (40,), generator=g).tolist()
    group = [TrainExample(f"q{j}", "c", prefix + torch.randint(0, 200, (n,), generator=g).tolist(), k, "db|full")
             for j, (n, k) in enumerate([(9, 4), (15, 6), (6, 6)])]  # the last one: answer right after the prefix
    group[1].input_ids[40] = group[0].input_ids[40]  # two questions share one token beyond the prefix
    assert shared_prefix_len(group) == 40 and packed_len(group) == 40 + 9 + 15 + 6

    def grads():
        return {n: w.grad.clone() for n, w in model.named_parameters() if w.requires_grad}

    alone = [group_losses(model, [ex], "cpu")[0] for ex in group]
    sum(alone).backward()
    g_alone = grads()
    model.zero_grad()
    packed = group_losses(model, group, "cpu")
    sum(packed).backward()
    g_packed = grads()
    for a, b in zip(alone, packed):
        assert torch.allclose(a, b, atol=1e-5)
    assert all(torch.allclose(g_alone[n], g_packed[n], atol=1e-5) for n in g_alone)


def test_step_plan_two_databases_per_step():
    ex = [TrainExample(f"q{i}", "c", [i], 1, f"db{i % 5}|full") for i in range(203)]  # about 40 per database
    steps = plan_steps(ex, 16, 8, seed=3)
    assert len(steps) == 203 // 16 and all(sum(map(len, s)) == 16 for s in steps)
    used = [i for s in steps for grp in s for i in grp]
    assert len(used) == len(set(used)) == 16 * len(steps)  # every question at most once, 11 dropped
    for s in steps:
        for grp in s:
            assert len({ex[i].key for i in grp}) == 1 and len(grp) <= 8
    full = [s for s in steps if [len(grp) for grp in s] == [8, 8]]
    assert len(full) >= 10 and all(ex[s[0][0]].key != ex[s[1][0]].key for s in full)
    assert plan_steps(ex, 16, 8, seed=3) == steps
    one = plan_steps(ex, 16, 1, seed=3)  # pack_size 1: one question per sequence
    assert all(len(grp) == 1 for s in one for grp in s)


def test_train_resume_gives_the_same_adapter(setup):
    import torch

    p, cfg, gen_cfg = setup
    full = train_sft("gold", "exp", "A", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu")
    assert full["complete"] and full["n_steps"] == 3
    log = read_jsonl(p.run_dir("A") / "train_log.jsonl")
    assert [r["step"] for r in log] == [1, 2, 3] and all(r["loss"] > 0 for r in log)

    part = train_sft("gold", "exp", "B", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu", max_steps=1)
    assert not part["complete"]
    rest = train_sft("gold", "exp", "B", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu")  # resumes at step 1
    assert rest["complete"] and [r["step"] for r in read_jsonl(p.run_dir("B") / "train_log.jsonl")] == [1, 2, 3]
    a, b = adapter_weights(p.run_dir("A")), adapter_weights(p.run_dir("B"))
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k]) for k in a)
    assert (p.run_dir("A") / "adapter" / "adapter_config.json").exists()
    assert len(list((p.run_dir("A") / "checkpoints").glob("step_*.pt"))) == 2  # keep_checkpoints

    with pytest.raises(RuntimeError, match="different training setup"):
        train_sft("best", "exp", "A", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu")

    cont = train_sft("gold", "exp", "A2", epoch=1, init_from="A", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu")
    assert cont["complete"] and cont["epoch"] == 1 and cont["lr"] == pytest.approx(cfg["sft"]["lr"] * 0.5)
    assert any(not torch.allclose(a[k], adapter_weights(p.run_dir("A2"))[k]) for k in a)


def test_pilot_reports_speed_and_projection(setup):
    p, cfg, gen_cfg = setup
    rep = train_sft("tagged", "exp", "exp_pilot", paths=p, cfg=cfg, gen_cfg=gen_cfg, device="cpu",
                    max_steps=3, pilot=True)
    assert rep["gpu"] == "cpu" and rep["tokens_per_s"] > 0 and rep["projected_hours_per_epoch"] > 0
    assert rep["longest_example_tokens"] == rep["train_examples"]["tokens_max"]
    assert not (p.run_dir("exp_pilot") / "checkpoints").exists()


def test_predict_and_validation_scoring(setup):
    p, cfg, gen_cfg = setup
    cfg_v = load_config("train.yaml")
    cfg_v["export"]["validation"] = {"rule": "random", "fraction": 0.05}
    meta = train_export("pool", "expv", paths=p, cfg=cfg_v)
    assert meta["holdout_db_ids"] and meta["val_qids"]
    fb = FakeBackend()
    st = predict_sft("tagged", "val", "R", "expv", fb, paths=p, cfg=cfg_v)
    preds = read_jsonl(p.run_dir("R") / "predictions_val.jsonl")
    assert {r["qid"] for r in preds} == set(meta["val_qids"]) and st["tag"] == preds[0]["tag"].strip()
    assert preds[0]["tag"].startswith(("+", "-"))
    predict_sft("gold", "dev", "R", "expv", fb, paths=p, cfg=cfg_v, limit=2)
    assert len(read_jsonl(p.run_dir("R") / "predictions_dev.jsonl")) == 2

    ecfg = load_config("evaluate.yaml")
    ecfg["exec"]["workers"] = 1
    with pytest.raises(ValueError, match="only-predicted"):
        evaluate(str(p.run_dir("R") / "predictions_val.jsonl"), "train", "R_val", paths=p, cfg=ecfg)
    m = evaluate(str(p.run_dir("R") / "predictions_val.jsonl"), "train", "R_val", only_predicted=True, paths=p, cfg=ecfg)
    assert m["overall"]["n"] == len(meta["val_qids"])
    assert read_jsonl(p.run_dir("R_val") / "eval_train.jsonl")


def test_selection_rule(tmp_path):
    def write(name, correct, cost):
        d = tmp_path / name
        append_jsonl(d / "eval_train.jsonl", [{"qid": f"q{i}", "eval_version": EVAL_VERSION, "correct": c,
                                              "rel_cost": r if c else None} for i, (c, r) in enumerate(zip(correct, cost))])
        return d

    n = 200
    a = write("a", [i < 160 for i in range(n)], [0.4 if i < 20 else 1.0 for i in range(n)])      # EX 80, 10% cheap
    b = write("b", [i < 156 for i in range(n)], [0.4 if i < 60 else 1.0 for i in range(n)])      # EX 78, 30% cheap
    c = write("c", [i < 100 for i in range(n)], [0.3 for i in range(n)])                         # EX 50, 50% cheap
    res = select_method({"a": a, "b": b, "c": c}, out=tmp_path / "sel.json")
    assert res["best_ex_method"] == "a" and res["winner"] == "b"
    assert res["methods"]["b"]["eligible"] and not res["methods"]["c"]["eligible"]
    assert json.loads((tmp_path / "sel.json").read_text())["winner"] == "b"
    assert mcnemar_p([True] * 10, [True] * 10) == 1.0


def test_cli_parses(tmp_path):
    from effsql.train.__main__ import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert read_json  # module imports cleanly
