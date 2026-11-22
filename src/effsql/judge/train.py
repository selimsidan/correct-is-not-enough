"""`python -m effsql.judge train`: train the judge's adapter and cost head (A100, resumable).

The adapter starts as a copy of the generator's (train.init_from) and learns, with the cost head, from
runs/<data_run>/judge_train.jsonl. Loss per optimizer step, averaged over its candidates: cross entropy of
"yes" against "no" (the candidate's label) plus cost_weight x squared error of c against the cost label
(correct candidates with one only). One pass over the training questions.

Resume: a checkpoint (adapter weights, cost head, optimizer, scheduler, RNG, step) every save_every steps;
running the same command again continues from the newest one. At the end: runs/<run>/adapter/ (the
adapter, as vLLM and peft read it), cost_head.pt and judge_meta.json (complete: true). Validation metrics
on the held-out databases go to val_log.jsonl every eval_every steps and at the end.

With results (`--with-results`): every prompt also shows what the candidate returned when run with the
answer-time cap (exec_train.jsonl / exec_val.jsonl of the data run, from `exec-data`). Everything else is as
without, so the two judges differ only in what they read.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

from effsql.config import load_config, save_config
from effsql.io import append_jsonl, read_json, write_json
from effsql.judge.data import read_part
from effsql.judge.model import (
    answer_token_ids,
    build_questions,
    forward,
    judge_metrics,
    load_cost_head,
    make_groups,
    new_cost_head,
    packed_len,
    plan_steps,
)
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
JUDGE_VERSION = 1


def evaluate_questions(model: Any, head: Any, questions: list[Any], groups: list[list[int]], device: str,
                       yes_no: tuple[int, int]) -> list[dict[str, Any]]:
    import torch

    was = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for g in groups:
            group = [questions[i] for i in g]
            lp, c = forward(model, head, group, device, yes_no)
            lp, c = lp.tolist(), c.tolist()
            k = 0
            for q in group:
                n = len(q.ids)
                rows.append({"qid": q.qid, "lp": lp[k:k + n], "c": c[k:k + n], "labels": q.labels, "costs": q.costs})
                k += n
    if was:
        model.train()
    return rows


def save_state(ckpt_dir: Path, step: int, model: Any, head: Any, opt: Any, sched: Any, fp: dict[str, Any],
               keep: int) -> None:
    import torch
    from peft import get_peft_model_state_dict

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {"step": step, "fingerprint": fp, "lora": get_peft_model_state_dict(model), "head": head.state_dict(),
             "opt": opt.state_dict(), "sched": sched.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    final = ckpt_dir / f"step_{step:06d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(final)
    for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
        old.unlink()


def load_state(ckpt_dir: Path, model: Any, head: Any, opt: Any, sched: Any, fp: dict[str, Any]) -> int:
    import torch
    from peft import set_peft_model_state_dict

    ckpts = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        return 0
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    if state["fingerprint"] != fp:
        raise RuntimeError(f"{ckpts[-1]} belongs to a different setup ({state['fingerprint']} vs {fp}); use a new run")
    set_peft_model_state_dict(model, state["lora"])
    head.load_state_dict(state["head"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    log.info("resumed from %s (step %d)", ckpts[-1].name, state["step"])
    return int(state["step"])


def load_judge(paths: Paths, run: str, device: str, cfg: dict[str, Any] | None = None,
               gen_cfg: dict[str, Any] | None = None) -> tuple[Any, Any, Any, tuple[int, int], dict[str, Any]]:
    """(model, cost head, tokenizer, yes/no ids, judge_meta) of a finished judge run, for scoring."""
    from effsql.train.sft import base_model_cfg, load_model, load_tokenizer

    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    run_dir = paths.run_dir(run, create=False)
    meta = read_json(run_dir / "judge_meta.json")
    if not meta.get("complete"):
        raise RuntimeError(f"{run} has not finished training (judge_meta.json: complete is false)")
    model_cfg = base_model_cfg(cfg, gen_cfg)
    tok = load_tokenizer(model_cfg)
    model = load_model(model_cfg, {**cfg["sft"], "gradient_checkpointing": False}, device, run_dir / "adapter")
    model.eval()
    head = load_cost_head(run_dir / "cost_head.pt", device)
    return model, head, tok, (meta["yes_id"], meta["no_id"]), meta


def train_judge(run: str, data_run: str, *, paths: Paths | None = None, jcfg: dict[str, Any] | None = None,
                cfg: dict[str, Any] | None = None, gen_cfg: dict[str, Any] | None = None,
                max_steps: int | None = None, device: str | None = None, with_results: bool = False) -> dict[str, Any]:
    import torch

    from effsql.train.sft import (
        base_model_cfg,
        check_gpu,
        load_model,
        load_tokenizer,
        lr_lambda,
    )

    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    tc, sft = jcfg["train"], cfg["sft"]
    model_cfg = base_model_cfg(cfg, gen_cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = check_gpu(device, sft)
    torch.manual_seed(tc["seed"])
    data_dir = paths.run_dir(data_run, create=False)
    dmeta = read_json(data_dir / "data_meta.json")
    run_dir = paths.run_dir(run)

    tok = load_tokenizer(model_cfg)
    yes_no = answer_token_ids(tok)
    budget = cfg["prompt"]["max_prompt_tokens"]
    max_tokens = int(tc.get("max_seq_tokens", 16000))
    timeout_s = jcfg["vote"]["timeout_s"]
    if with_results:
        from effsql.judge.vote import read_exec

        res = {part: read_exec(paths, data_run, part) for part in ("train", "val")}
    else:
        res = {"train": None, "val": None}
    train_q, tstats = build_questions(tok, read_part(data_dir, "train"), "train", paths, budget, res["train"], timeout_s)
    val_q, vstats = build_questions(tok, read_part(data_dir, "val"), "train", paths, budget, res["val"], timeout_s)
    steps = plan_steps(train_q, tc["questions_per_seq"], tc["questions_per_step"], max_tokens, tc["seed"])
    import numpy as np

    vgroups = make_groups(val_q, list(range(len(val_q))), tc["questions_per_seq"], max_tokens,
                          np.random.default_rng(tc["seed"]))
    full_steps = len(steps)
    n_steps = min(full_steps, max_steps) if max_steps else full_steps
    if n_steps == 0:
        raise RuntimeError("no training questions")

    init_dir = paths.run_dir(tc["init_from"], create=False) / "adapter" if tc.get("init_from") else None
    model = load_model(model_cfg, sft, device, init_dir)
    head = new_cost_head(model.config.hidden_size, device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": tc["lr"]},
                             {"params": list(head.parameters()), "lr": tc["head_lr"]}],
                            weight_decay=0.0, fused=device == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(full_steps, max(1, math.ceil(tc["warmup_ratio"] * full_steps))))
    fp = {"data_run": data_run, "data_stats": dmeta["stats"], "init_from": tc["init_from"], "n_steps": full_steps,
          "train": tc, "judge_version": JUDGE_VERSION, "with_results": with_results}
    ckpt_dir = run_dir / "checkpoints"
    start = load_state(ckpt_dir, model, head, opt, sched, fp)
    save_config(jcfg, run_dir / "judge_config.yaml")
    log.info("judge: %d questions, %d candidates (%d skipped too long), %d steps, starting at %d on %s",
             tstats["n_questions"], tstats["n_candidates"], tstats["n_skipped_too_long"], n_steps, start, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    t0, timed_tokens, t_timed, last_val = time.time(), 0, None, None
    params = lora_params + list(head.parameters())
    for step in range(start, n_steps):
        n_items = sum(len(train_q[i].ids) for g in steps[step] for i in g)
        tot_bce = tot_se = 0.0
        n_cost = n_right = tokens = 0
        for g in steps[step]:
            group = [train_q[i] for i in g]
            lp, c = forward(model, head, group, device, yes_no)
            y = torch.tensor([x for q in group for x in q.labels], dtype=torch.float32, device=device)
            costs = [x for q in group for x in q.costs]
            has = torch.tensor([x is not None for x in costs], device=device)
            t = torch.tensor([0.0 if x is None else x for x in costs], dtype=torch.float32, device=device)
            l_bce = bce(lp, y)
            l_se = (((c - t) ** 2) * has).sum()
            ((l_bce + tc["cost_weight"] * l_se) / n_items).backward()
            tot_bce += float(l_bce.detach())
            tot_se += float(l_se.detach())
            n_cost += int(has.sum())
            n_right += int(((lp.detach() > 0).float() == y).sum())
            tokens += packed_len(group)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"]))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        now = time.time()
        if step > start:
            timed_tokens += tokens
        else:
            t_timed = now
        rec = {"step": step + 1, "n_steps": n_steps, "bce": tot_bce / n_items, "cost_mse": tot_se / n_cost if n_cost else None,
               "accuracy": n_right / n_items, "n_candidates": n_items, "lr": sched.get_last_lr()[0],
               "grad_norm": grad_norm, "tokens": tokens, "elapsed_s": now - t0,
               "tokens_per_s": timed_tokens / (now - t_timed) if timed_tokens and t_timed and now > t_timed else None,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
        if rec["tokens_per_s"]:
            rec["projected_total_min"] = (now - t0) / (step + 1 - start) * (n_steps - start) / 60
        append_jsonl(run_dir / "train_log.jsonl", [rec])
        log.info("step %d/%d bce %.4f cost_mse %s acc %.3f lr %.2e %.0f tok/s", step + 1, n_steps, rec["bce"],
                 f"{rec['cost_mse']:.3f}" if rec["cost_mse"] is not None else "-", rec["accuracy"], rec["lr"],
                 rec["tokens_per_s"] or 0)
        if (step + 1) % tc["save_every"] == 0:
            save_state(ckpt_dir, step + 1, model, head, opt, sched, fp, tc.get("keep_checkpoints", 2))
        if (step + 1) % tc["eval_every"] == 0:
            m = judge_metrics(evaluate_questions(model, head, val_q, vgroups, device, yes_no))
            append_jsonl(run_dir / "val_log.jsonl", [{"step": step + 1, **m}])
            last_val = step + 1
            log.info("validation: %s", m)

    final = judge_metrics(evaluate_questions(model, head, val_q, vgroups, device, yes_no))
    if last_val != n_steps:
        append_jsonl(run_dir / "val_log.jsonl", [{"step": n_steps, **final}])
    if n_steps % tc["save_every"]:
        save_state(ckpt_dir, n_steps, model, head, opt, sched, fp, tc.get("keep_checkpoints", 2))
    adapter, tmp = run_dir / "adapter", run_dir / "adapter.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    if adapter.exists():
        shutil.rmtree(adapter)
    tmp.replace(adapter)
    from effsql.judge.model import save_cost_head

    save_cost_head(head, run_dir / "cost_head.pt")
    meta = {"data_run": data_run, "init_from": tc["init_from"], "base_id": model_cfg["id"],
            "base_revision": model_cfg.get("revision"), "yes_id": yes_no[0], "no_id": yes_no[1],
            "hidden_size": head.in_features, "n_steps": n_steps, "gpu": gpu, "train_questions": tstats,
            "val_questions": vstats, "val_final": final, "complete": n_steps == full_steps,
            "judge_version": JUDGE_VERSION, "with_results": with_results,
            "timeout_s": timeout_s if with_results else None}
    write_json(run_dir / "judge_meta.json", meta)
    log.info("judge written to %s; validation %s", run_dir, final)
    return meta
