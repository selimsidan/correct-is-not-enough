"""SFT with LoRA: one arm of an export, one epoch per invocation, resumable after a disconnect.

Data: runs/<export>/data/<arm>_train.jsonl (the records of the requested epoch) and <arm>_val.jsonl.
Each example is the chat prompt that prediction uses for the same question and arm (same template,
token budget and shrink ladder: train.data.train_messages), followed by the target answer
```sql ...``` and the end-of-turn token. Examples whose prompt exceeds the budget even at the last
shrink level are skipped (and counted).

Loss: mean cross entropy over the answer tokens only. Logits are computed only for the answer
positions (`logits_to_keep`), so a 10k-token schema costs no vocabulary-sized activations.

Shared-schema groups: the questions of one database share the system prompt and the schema, which
are about 95% of every prompt. Up to `pack_size` questions of one database (same prompt shrink level)
go into one sequence: the shared prefix once, then each question's own tokens (evidence, question,
answer). An attention mask lets each question see only the prefix and itself, and its position ids
continue from the prefix as if it were alone, so every question's loss is exactly the loss of
training it alone. The prefix is computed once per group instead of once per question.

Batching: an optimizer step has exactly `grad_accum` questions, each weighing the same (per-question
mean loss, averaged over the step). With grad_accum 16 and pack_size 8 a step is two groups from two
different databases (decided 2026-11-14); the last questions of each database (fewer than pack_size)
fill the remaining steps. Steps come in a seeded random order. A final partial step is dropped.

Resume: a checkpoint (LoRA weights, optimizer, scheduler, RNG, step) is written atomically to
runs/<run>/checkpoints/ every `save_every` optimizer steps; running the same command again continues
from the newest one. The finished adapter goes to runs/<run>/adapter/ (save_pretrained, read by vLLM)
with adapter_meta.json. Validation loss (held-out databases) is logged every `eval_every` steps and
at the end.

Continuing (the winner's second epoch): `init_from` names a finished run whose adapter starts the new
run; the optimizer is fresh and the learning rate is lr x continue_lr_factor.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from effsql.config import load_config, save_config
from effsql.data.bird import load_examples
from effsql.gen.backend import stable_seed
from effsql.gen.prompts import load_templates
from effsql.gen.sample import schemas_for
from effsql.io import append_jsonl, read_json, write_json
from effsql.paths import Paths, load_paths
from effsql.runmeta import write_run_meta
from effsql.train.data import read_export, target_completion, train_messages

log = logging.getLogger(__name__)
END_OF_TURN = "<|im_end|>"  # Qwen2.5 chat end of turn; vLLM stops generation on it
SFT_VERSION = 2  # 2: shared-schema groups


@dataclass
class TrainExample:
    qid: str
    cand_id: str
    input_ids: list[int]
    n_target: int
    key: str = ""  # questions with the same key share their prompt prefix: database and shrink level


# Tokenization ---------------------------------------------------------------------------------
def prompt_text(tok: Any, messages: list[dict[str, str]]) -> str:
    """Exactly what the vLLM backend sends for these messages."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_counter(tok: Any):
    """Token count of the prompt, as the vLLM backend counts it (shrink-ladder decisions match)."""
    return lambda messages: len(tok(prompt_text(tok, messages), add_special_tokens=False)["input_ids"])


def end_token_id(tok: Any) -> int:
    tid = tok.convert_tokens_to_ids(END_OF_TURN)
    if tid is None or tid == getattr(tok, "unk_token_id", None):
        raise ValueError(f"tokenizer has no {END_OF_TURN} token")
    return int(tid)


def build_examples(recs: list[dict[str, Any]], arm: str, tok: Any, paths: Paths, budget: int
                   ) -> tuple[list[TrainExample], dict[str, Any]]:
    examples = {e.qid: e for e in load_examples("train", paths)}
    schemas = schemas_for("train", {examples[r["qid"]].db_id for r in recs}, paths)
    templates = load_templates()
    count, eot = make_counter(tok), end_token_id(tok)
    out, skipped, levels = [], 0, {}
    for r in recs:
        e = examples[r["qid"]]
        msgs, level, n = train_messages(templates, arm, e, schemas[e.db_id], budget, count, tag=r.get("tag_text"))
        if n > budget:
            skipped += 1
            continue
        levels[level] = levels.get(level, 0) + 1
        prompt_ids = tok(prompt_text(tok, msgs), add_special_tokens=False)["input_ids"]
        target_ids = tok(target_completion(r["sql"]), add_special_tokens=False)["input_ids"] + [eot]
        out.append(TrainExample(r["qid"], r["cand_id"], prompt_ids + target_ids, len(target_ids), f"{e.db_id}|{level}"))
    lens = [len(x.input_ids) for x in out]
    stats = {"n_records": len(recs), "n_examples": len(out), "n_skipped_too_long": skipped,
             "prompt_levels": levels, "tokens_total": int(sum(lens)),
             "tokens_mean": float(np.mean(lens)) if lens else 0.0, "tokens_max": int(max(lens)) if lens else 0}
    return out, stats


# Model ----------------------------------------------------------------------------------------
def base_model_cfg(cfg: dict[str, Any], gen_cfg: dict[str, Any]) -> dict[str, Any]:
    """generate.yaml's model with the exact base revision of train.yaml (training and prediction)."""
    model = dict(gen_cfg["model"])
    if "base_revision" in cfg.get("sft", {}):
        model["revision"] = cfg["sft"]["base_revision"]
    return model


def load_tokenizer(model_cfg: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg.get("revision"))


def load_model(model_cfg: dict[str, Any], sft: dict[str, Any], device: str, init_adapter: Path | None = None) -> Any:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], revision=model_cfg.get("revision"), dtype=dtype,
        attn_implementation=sft.get("attn_implementation", "sdpa"),
        device_map=device if device == "cuda" else None,
    )
    model.config.use_cache = False
    if sft.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if init_adapter is not None:
        return PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    lc = sft["lora"]
    return get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                            target_modules=list(lc["target_modules"]), task_type="CAUSAL_LM"))


# Shared-schema groups -------------------------------------------------------------------------
def shared_prefix_len(group: list[TrainExample]) -> int:
    """Longest common token prefix of the group, never reaching into any answer."""
    n = min(len(e.input_ids) - e.n_target for e in group)
    first = np.asarray(group[0].input_ids[:n])
    for e in group[1:]:
        diff = np.nonzero(np.asarray(e.input_ids[:n]) != first)[0]
        if len(diff):
            n = int(diff[0])
            first = first[:n]
    return n


def packed_len(group: list[TrainExample]) -> int:
    p = shared_prefix_len(group)
    return p + sum(len(e.input_ids) - p for e in group)


def make_groups(idx: list[int], examples: list[TrainExample], size: int, rng: np.random.Generator
                ) -> tuple[list[list[int]], list[list[int]]]:
    """Per key: a random order cut into groups of `size`. Returns (full groups, smaller remainders)."""
    by_key: dict[str, list[int]] = {}
    for i in idx:
        by_key.setdefault(examples[i].key, []).append(i)
    full, rest = [], []
    for k in sorted(by_key):
        members = [by_key[k][j] for j in rng.permutation(len(by_key[k]))]
        n_full = len(members) // size * size
        full += [members[j:j + size] for j in range(0, n_full, size)]
        if n_full < len(members):
            rest.append(members[n_full:])
    return full, rest


def plan_steps(examples: list[TrainExample], ga: int, pack_size: int, seed: int) -> list[list[list[int]]]:
    """Optimizer steps, each a list of groups (example indices sharing a prompt prefix) with exactly
    `ga` questions in total. Full groups are combined so a step's groups come from different keys
    when possible; the remainders are shuffled together and cut into steps (adjacent questions of
    one key share a group). The order of the steps is random."""
    rng = np.random.default_rng(seed)
    full, rest = make_groups(list(range(len(examples))), examples, max(1, pack_size), rng)
    per_step = ga // pack_size if pack_size > 1 and ga % pack_size == 0 else 0
    steps: list[list[list[int]]] = []
    if per_step:
        pending = [full[j] for j in rng.permutation(len(full))]
        while len(pending) >= per_step:
            step = [pending.pop(0)]
            while len(step) < per_step:
                keys = {examples[g[0]].key for g in step}
                j = next((j for j, g in enumerate(pending) if examples[g[0]].key not in keys), 0)
                step.append(pending.pop(j))
            steps.append(step)
        rest += pending
    else:
        rest += full
    flat = [i for j in rng.permutation(len(rest)) for i in rest[j]]
    for s in range(0, len(flat) // ga * ga, ga):
        chunk, groups = flat[s:s + ga], []
        for i in chunk:
            if groups and examples[groups[-1][0]].key == examples[i].key and len(groups[-1]) < max(1, pack_size):
                groups[-1].append(i)
            else:
                groups.append([i])
        steps.append(groups)
    return [steps[j] for j in rng.permutation(len(steps))]


def group_losses(model: Any, group: list[TrainExample], device: str) -> list[Any]:
    """Per-question mean cross entropy over the answer tokens, for a group sharing a prompt prefix:
    one forward pass over [prefix, rest of question 1, rest of question 2, ...]. Each question
    attends only to the prefix and to itself and keeps the positions it has when alone, so each loss
    equals the loss of that question in its own sequence."""
    import torch
    import torch.nn.functional as F

    if len(group) == 1:
        ex = group[0]
        ids = torch.tensor([ex.input_ids], device=device)
        k = ex.n_target
        logits = model(input_ids=ids, logits_to_keep=k + 1, use_cache=False).logits[:, :-1, :].float()
        return [F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, -k:].reshape(-1))]

    p = shared_prefix_len(group)
    ids, pos, seg, keep, labels, spans = list(group[0].input_ids[:p]), list(range(p)), [0] * p, [], [], []
    for j, ex in enumerate(group, start=1):
        start, n = len(ids), len(ex.input_ids)
        ids += ex.input_ids[p:]
        pos += range(p, n)
        seg += [j] * (n - p)
        # original position t of an answer token is predicted at t - 1; map it into the packed sequence
        first = len(keep)
        for t in range(n - ex.n_target, n):
            keep.append(t - 1 if t - 1 < p else start + (t - 1 - p))
            labels.append(ex.input_ids[t])
        spans.append((first, len(keep)))
    seg_t = torch.tensor(seg, device=device)
    L = len(ids)
    causal = torch.ones(L, L, dtype=torch.bool, device=device).tril_()
    visible = (seg_t[None, :] == 0) | (seg_t[None, :] == seg_t[:, None])
    mask = (causal & visible)[None, None]
    out = model(input_ids=torch.tensor([ids], device=device), position_ids=torch.tensor([pos], device=device),
                attention_mask=mask, logits_to_keep=torch.tensor(keep, device=device), use_cache=False)
    logits = out.logits[0].float()
    lab = torch.tensor(labels, device=device)
    return [F.cross_entropy(logits[a:b], lab[a:b]) for a, b in spans]


def val_groups(examples: list[TrainExample], pack_size: int, seed: int) -> list[list[int]]:
    full, rest = make_groups(list(range(len(examples))), examples, max(1, pack_size), np.random.default_rng(seed))
    return full + rest


def val_loss(model: Any, examples: list[TrainExample], groups: list[list[int]], device: str) -> float | None:
    import torch

    if not examples:
        return None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        total = sum(float(x) for g in groups for x in group_losses(model, [examples[i] for i in g], device))
    if was_training:
        model.train()
    return total / sum(len(g) for g in groups)


def lr_lambda(n_steps: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, n_steps - warmup)))
    return f


# Checkpoints ----------------------------------------------------------------------------------
def _fingerprint(arm: str, epoch: int, export_sha: str, n_steps: int, init_from: str | None,
                 pack_size: int) -> dict[str, Any]:
    return {"arm": arm, "epoch": epoch, "export_pool_sha256": export_sha, "n_steps": n_steps,
            "init_from": init_from, "pack_size": pack_size, "sft_version": SFT_VERSION}


def save_checkpoint(ckpt_dir: Path, step: int, model: Any, opt: Any, sched: Any, fp: dict[str, Any], keep: int) -> None:
    import torch
    from peft import get_peft_model_state_dict

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {"step": step, "fingerprint": fp, "lora": get_peft_model_state_dict(model),
             "opt": opt.state_dict(), "sched": sched.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    final = ckpt_dir / f"step_{step:06d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(final)  # a disconnect mid-write never leaves a broken newest checkpoint
    for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
        old.unlink()


def load_latest_checkpoint(ckpt_dir: Path, model: Any, opt: Any, sched: Any, fp: dict[str, Any]) -> int:
    import torch
    from peft import set_peft_model_state_dict

    ckpts = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        return 0
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    if state["fingerprint"] != fp:
        raise RuntimeError(f"{ckpts[-1]} belongs to a different training setup ({state['fingerprint']} vs {fp}); "
                           "use a new run name")
    set_peft_model_state_dict(model, state["lora"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    log.info("resumed from %s (step %d)", ckpts[-1].name, state["step"])
    return int(state["step"])


# Training -------------------------------------------------------------------------------------
def check_gpu(device: str, sft: dict[str, Any]) -> str:
    """Stop before any download when the GPU is too small for the longest training sequence."""
    import torch

    if device != "cuda":
        return device
    name = torch.cuda.get_device_name()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    need = sft.get("min_gpu_memory_gb")
    if need and total < need:
        raise RuntimeError(f"this runtime's GPU ({name}, {total:.1f} GiB) is too small: training needs at least "
                           f"{need} GiB (an A100 40 GB). Change the runtime type to A100 and run the notebook again.")
    return f"{name} ({total:.1f} GiB)"


def train_sft(arm: str, export_run: str, run: str, *, epoch: int = 0, init_from: str | None = None,
              paths: Paths | None = None, cfg: dict[str, Any] | None = None, gen_cfg: dict[str, Any] | None = None,
              max_steps: int | None = None, pilot: bool = False, device: str | None = None) -> dict[str, Any]:
    import torch

    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    sft, model_cfg = cfg["sft"], base_model_cfg(cfg, gen_cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = check_gpu(device, sft)
    torch.manual_seed(sft["seed"])
    export_dir = paths.run_dir(export_run, create=False)
    emeta = read_json(export_dir / "export_meta.json")
    run_dir = paths.run_dir(run)

    tok = load_tokenizer(model_cfg)
    budget = cfg["prompt"]["max_prompt_tokens"]
    recs = [r for r in read_export(export_dir, arm, "train") if r["epoch"] == epoch]
    if not recs:
        raise RuntimeError(f"{arm}_train.jsonl has no records for epoch {epoch}")
    train_ex, tstats = build_examples(recs, arm, tok, paths, budget)
    val_ex, vstats = build_examples(read_export(export_dir, arm, "val"), arm, tok, paths, budget)
    ga, pack = sft["grad_accum"], int(sft.get("pack_size", 1))
    steps = plan_steps(train_ex, ga, pack, stable_seed(sft["seed"], "sft", arm, epoch))
    vgroups = val_groups(val_ex, pack, sft["seed"])
    if pilot:  # the step with the longest sequence first: the memory peak shows up at once
        top = max(range(len(steps)), key=lambda s: max(packed_len([train_ex[i] for i in g]) for g in steps[s]))
        steps = [steps[top]] + steps[:top] + steps[top + 1:]
    full_steps = len(steps)  # the schedule always spans the whole epoch, also for --max-steps
    n_steps = min(full_steps, max_steps) if max_steps else full_steps
    if n_steps == 0:
        raise RuntimeError(f"{len(train_ex)} examples are fewer than one batch of {ga}")

    init_dir = paths.run_dir(init_from, create=False) / "adapter" if init_from else None
    model = load_model(model_cfg, sft, device, init_dir)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = sft["lr"] * (sft.get("continue_lr_factor", 1.0) if init_from else 1.0)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=sft.get("weight_decay", 0.0),
                            fused=device == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(full_steps, max(1, math.ceil(sft["warmup_ratio"] * full_steps))))
    fp = _fingerprint(arm, epoch, emeta["pool_sha256"], full_steps, init_from, pack)
    ckpt_dir = run_dir / "checkpoints"
    start = 0 if pilot else load_latest_checkpoint(ckpt_dir, model, opt, sched, fp)
    if not pilot:
        save_config(cfg, run_dir / "train_config.yaml")
        write_run_meta(run_dir, extra={"command": "sft", "gpu": gpu, "arm": arm, "export_run": export_run, "epoch": epoch,
                                       "init_from": init_from, "fingerprint": fp})
    log_path = run_dir / ("pilot_log.jsonl" if pilot else "train_log.jsonl")
    if pilot and log_path.exists():
        log_path.unlink()  # a pilot is a fresh measurement
    log.info("sft %s epoch %d: %d examples (%d skipped too long), %d steps of %d in groups of up to %d, "
             "starting at %d on %s", arm, epoch, len(train_ex), tstats["n_skipped_too_long"], n_steps, ga, pack,
             start, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, tokens, timed_tokens, t_timed = time.time(), 0, 0, None
    last_val_step = None
    for step in range(start, n_steps):
        loss_sum, step_tokens, n_q = 0.0, 0, 0
        for g in steps[step]:
            group = [train_ex[i] for i in g]
            losses = group_losses(model, group, device)
            (sum(losses) / ga).backward()
            loss_sum += sum(x.detach().item() for x in losses)
            step_tokens += packed_len(group)
            n_q += len(group)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, sft["max_grad_norm"]))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        tokens += step_tokens
        now = time.time()
        if step > start:  # the first step includes warm-up costs (kernels, allocator)
            timed_tokens += step_tokens
        else:
            t_timed = now
        rec = {"step": step + 1, "n_steps": n_steps, "loss": loss_sum / n_q, "n_seqs": len(steps[step]), "lr": sched.get_last_lr()[0],
               "grad_norm": grad_norm, "tokens": step_tokens, "elapsed_s": now - t0,
               "tokens_per_s": timed_tokens / (now - t_timed) if timed_tokens and now > t_timed else None,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
        append_jsonl(log_path, [rec])
        log.info("step %d/%d loss %.4f lr %.2e %.0f tok/s", step + 1, n_steps, rec["loss"], rec["lr"],
                 rec["tokens_per_s"] or 0)
        if pilot:
            continue
        if (step + 1) % sft["save_every"] == 0:
            save_checkpoint(ckpt_dir, step + 1, model, opt, sched, fp, sft.get("keep_checkpoints", 2))
        if (step + 1) % sft["eval_every"] == 0:
            append_jsonl(run_dir / "val_log.jsonl", [{"step": step + 1, "val_loss": val_loss(model, val_ex, vgroups, device)}])
            last_val_step = step + 1

    tok_s = timed_tokens / (time.time() - t_timed) if timed_tokens and t_timed else None
    if pilot:
        epoch_tokens = sum(packed_len([train_ex[i] for i in g]) for s in steps for g in s)
        val_tokens = sum(packed_len([val_ex[i] for i in g]) for g in vgroups)
        return pilot_report(run_dir, arm, tstats, vstats, tok_s, device, sft, epoch_tokens, val_tokens, full_steps)

    final_val = val_loss(model, val_ex, vgroups, device) if last_val_step != n_steps else None
    if final_val is not None:
        append_jsonl(run_dir / "val_log.jsonl", [{"step": n_steps, "val_loss": final_val}])
    if n_steps % sft["save_every"]:
        save_checkpoint(ckpt_dir, n_steps, model, opt, sched, fp, sft.get("keep_checkpoints", 2))
    adapter = run_dir / "adapter"
    tmp = run_dir / "adapter.tmp"
    model.save_pretrained(str(tmp))
    if adapter.exists():
        import shutil

        shutil.rmtree(adapter)
    tmp.replace(adapter)
    meta = {"arm": arm, "epoch": epoch, "export_run": export_run, "export_pool_sha256": emeta["pool_sha256"],
            "init_from": init_from, "base_id": model_cfg["id"], "base_revision": model_cfg.get("revision"),
            "lr": lr, "n_steps": n_steps, "grad_accum": ga, "pack_size": pack, "gpu": gpu, "train_examples": tstats, "val_examples": vstats,
            "val_loss_final": final_val, "tokens_this_session": tokens, "tokens_per_s_this_session": tok_s,
            "complete": n_steps == full_steps, "sft_version": SFT_VERSION}
    write_json(run_dir / "adapter_meta.json", meta)
    log.info("adapter written to %s", adapter)
    return meta


def pilot_report(run_dir: Path, arm: str, tstats: dict[str, Any], vstats: dict[str, Any], tok_s: float | None,
                 device: str, sft: dict[str, Any], epoch_tokens: int, val_tokens: int, n_steps: int) -> dict[str, Any]:
    """Measured speed and memory, projected to one epoch of every SFT arm (they cost the same)."""
    import torch

    gpu = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    rate = next((v for k, v in (sft.get("units_per_hour") or {}).items() if k.lower() in gpu.lower()), None)
    # tokens actually processed (shared prefixes counted once); validation is a forward pass, about a
    # third of a training step per token, at every eval_every steps and at the end
    val_cost = val_tokens / 3 * (1 + n_steps // sft["eval_every"])
    hours = (epoch_tokens + val_cost) / tok_s / 3600 if tok_s else None
    rep = {"gpu": gpu, "arm": arm, "tokens_per_s": tok_s,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
           "longest_example_tokens": tstats["tokens_max"], "epoch_tokens": int(epoch_tokens),
           "epoch_tokens_unshared": tstats["tokens_total"], "pack_size": sft.get("pack_size", 1),
           "projected_hours_per_epoch": hours, "units_per_hour": rate,
           "projected_units_per_epoch": hours * rate if hours and rate else None,
           "train_examples": tstats, "val_examples": vstats}
    write_json(run_dir / f"pilot_{gpu.replace(' ', '_')}.json", rep)
    return rep
