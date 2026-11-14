"""DPO with an NLL term on the chosen answer ("RPO"), starting from a finished SFT adapter; resumable.

Data: runs/<export>/data/dpo_{train,val}.jsonl (`export --arms dpo`): for every question with an
efficiency pair, up to 4 pairs of a cheaper correct answer (chosen, at least 2x and 50 steps cheaper)
over a pricier correct one, plus 1 pair of a cheapest correct answer over an incorrect one. The
validation pairs come from the same held-out train databases as the SFT validation (checked).

Prompt: the prompt the starting SFT model predicts with: its arm's template and, for the tagged model,
one efficiency target `tag` on every question (the target chosen on validation). DPO trains exactly the
prompt used at test time.

Loss per pair (Rafailov et al. 2023; the NLL term as in Pang et al. 2024):
  -log sigmoid(beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]) + rpo_alpha * NLL(c)
log pi(x) is the summed log probability of answer x's tokens (```sql ...``` and the end of turn), NLL(c)
the chosen answer's mean token cross entropy. The reference is the starting SFT model: its log
probabilities are computed once before training (eval mode, no dropout) and saved to ref_logps.jsonl,
so only one model is in memory. A step's loss is the mean over its pairs.

Batching: all answers of a question share its prompt, and the questions of one database share the
system prompt and schema. Steps are planned over questions exactly as SFT plans them (sft.plan_steps:
`questions_per_step` questions, groups of `pack_size` questions of one database); a group is cut into
sequences of at most `max_answers_per_seq` answers (whole questions) and computed with
sft.group_losses, so every answer's log probability is exactly its log probability alone.

Resume: checkpoints and the finished adapter work as in SFT (runs/<run>/checkpoints, runs/<run>/adapter).
adapter_meta.json carries the starting model's arm and the tag, so `predict` and `eval-val` treat the
DPO adapter like any SFT adapter.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from effsql.config import load_config, save_config
from effsql.gen.backend import stable_seed
from effsql.io import append_jsonl, iter_jsonl, read_json, write_json
from effsql.paths import Paths, load_paths
from effsql.runmeta import write_run_meta
from effsql.train.data import read_export
from effsql.train.sft import (TrainExample, base_model_cfg, build_examples, check_gpu, group_losses, load_latest_checkpoint,
                              load_model, load_tokenizer, lr_lambda, packed_len, plan_steps, save_checkpoint)

log = logging.getLogger(__name__)
DPO_VERSION = 1


@dataclass
class Unit:
    """One question: its distinct answers (indices into the examples) and its pairs."""
    qid: str
    key: str
    answers: list[int]
    pairs: list[tuple[int, int, str]] = field(default_factory=list)  # (chosen, rejected, pair type)


def dpo_terms(pc: Any, pr: Any, rc: float, rr: float, nll_c: Any, beta: float, rpo_alpha: float) -> tuple[Any, Any]:
    """(loss, margin) of one pair; margin = beta * (chosen log ratio - rejected log ratio)."""
    import torch.nn.functional as F

    margin = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(margin) + rpo_alpha * nll_c, margin


def build_units(recs: list[dict[str, Any]], arm: str, tag: str | None, tok: Any, paths: Paths, budget: int
                ) -> tuple[list[TrainExample], list[Unit], dict[str, Any]]:
    answers: dict[tuple[str, str], dict[str, Any]] = {}
    for r in recs:
        for side in ("chosen", "rejected"):
            answers.setdefault((r["qid"], r[f"{side}_cand"]), {"qid": r["qid"], "cand_id": r[f"{side}_cand"],
                                                                "sql": r[f"{side}_sql"], "tag_text": tag})
    examples, stats = build_examples(list(answers.values()), arm, tok, paths, budget)
    pos = {(e.qid, e.cand_id): i for i, e in enumerate(examples)}
    units: dict[str, Unit] = {}
    for i, e in enumerate(examples):
        units.setdefault(e.qid, Unit(e.qid, e.key, [])).answers.append(i)
    n_dropped = 0
    for r in recs:
        c, j = pos.get((r["qid"], r["chosen_cand"])), pos.get((r["qid"], r["rejected_cand"]))
        if c is None or j is None:
            n_dropped += 1
            continue
        units[r["qid"]].pairs.append((c, j, r["pair_type"]))
    out = [u for u in units.values() if u.pairs]
    stats.update(n_pairs=sum(len(u.pairs) for u in out), n_pairs_dropped_too_long=n_dropped, n_questions=len(out),
                 by_type=dict(Counter(t for u in out for _, _, t in u.pairs)))
    return examples, out, stats


def seqs(group: list[Unit], cap: int) -> list[list[Unit]]:
    """Cut a group into sequences of at most `cap` answers, whole questions only."""
    out: list[list[Unit]] = []
    n = 0
    for u in group:
        if out and n + len(u.answers) <= cap:
            out[-1].append(u)
            n += len(u.answers)
        else:
            out.append([u])
            n = len(u.answers)
    return out


def answer_logps(model: Any, units: list[Unit], examples: list[TrainExample], device: str) -> tuple[dict[int, Any], dict[int, Any]]:
    """Summed log probability and mean token cross entropy of every answer of these units (one sequence)."""
    idx = [i for u in units for i in u.answers]
    ces = group_losses(model, [examples[i] for i in idx], device)
    return {i: -ce * examples[i].n_target for i, ce in zip(idx, ces)}, dict(zip(idx, ces))


def by_key_seqs(units: list[Unit], pack: int, cap: int) -> list[list[Unit]]:
    """Every unit once, in sequences of one key (reference pass and validation)."""
    by: dict[str, list[Unit]] = {}
    for u in units:
        by.setdefault(u.key, []).append(u)
    return [s for k in sorted(by) for j in range(0, len(by[k]), pack) for s in seqs(by[k][j:j + pack], cap)]


def reference_logps(model: Any, parts: dict[str, tuple[list[TrainExample], list[Unit]]], path: Any, fp: str,
                    pack: int, cap: int, device: str) -> dict[str, dict[tuple[str, str], float]]:
    """Log probabilities of the starting model, computed once (resumable) and saved."""
    import torch

    done: dict[str, dict[tuple[str, str], float]] = {p: {} for p in parts}
    for r in (iter_jsonl(path) if path.exists() else []):
        if r["fp"] != fp:
            raise RuntimeError(f"{path} was computed for another starting model or prompt ({r['fp']} vs {fp}); "
                               "use a new run name")
        done[r["part"]][(r["qid"], r["cand_id"])] = r["logp"]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for part, (examples, units) in parts.items():
            todo = [u for u in units if any((u.qid, examples[i].cand_id) not in done[part] for i in u.answers)]
            if todo:
                log.info("reference log probabilities: %d %s questions", len(todo), part)
            for s in by_key_seqs(todo, pack, cap):
                lp, _ = answer_logps(model, s, examples, device)
                recs = [{"part": part, "qid": examples[i].qid, "cand_id": examples[i].cand_id, "logp": float(v), "fp": fp}
                        for i, v in lp.items()]
                append_jsonl(path, recs)
                done[part].update({(r["qid"], r["cand_id"]): r["logp"] for r in recs})
    if was_training:
        model.train()
    return done


def pair_metrics(rows: list[tuple[float, float, float, str]]) -> dict[str, Any]:
    """rows: (loss, margin, nll of chosen, pair type) per pair."""
    if not rows:
        return {}
    a = np.asarray([r[:3] for r in rows], dtype=float)
    out = {"loss": float(a[:, 0].mean()), "reward_acc": float((a[:, 1] > 0).mean()), "margin": float(a[:, 1].mean()),
           "nll_chosen": float(a[:, 2].mean()), "n_pairs": len(rows)}
    for t in sorted({r[3] for r in rows}):
        out[f"reward_acc_{t}"] = float(np.mean([r[1] > 0 for r in rows if r[3] == t]))
    return out


def val_metrics(model: Any, examples: list[TrainExample], units: list[Unit], ref: dict[tuple[str, str], float],
                d: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    rows = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in by_key_seqs(units, d["pack_size"], d["max_answers_per_seq"]):
            lp, ce = answer_logps(model, s, examples, device)
            for u in s:
                for c, r, t in u.pairs:
                    loss, m = dpo_terms(lp[c], lp[r], ref[(u.qid, examples[c].cand_id)], ref[(u.qid, examples[r].cand_id)],
                                        ce[c], d["beta"], d["rpo_alpha"])
                    rows.append((float(loss), float(m), float(ce[c]), t))
    if was_training:
        model.train()
    return pair_metrics(rows)


def train_dpo(export_run: str, run: str, init_from: str, *, tag: str | None = None, paths: Paths | None = None,
              cfg: dict[str, Any] | None = None, gen_cfg: dict[str, Any] | None = None, max_steps: int | None = None,
              device: str | None = None) -> dict[str, Any]:
    import torch

    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    sft, d, model_cfg = cfg["sft"], cfg["dpo"], base_model_cfg(cfg, gen_cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = check_gpu(device, sft)
    torch.manual_seed(d["seed"])

    init_dir = paths.run_dir(init_from, create=False)
    imeta = read_json(init_dir / "adapter_meta.json")
    if not imeta.get("complete"):
        raise RuntimeError(f"{init_from} has not finished training")
    if imeta.get("base_revision") != model_cfg.get("revision"):
        raise RuntimeError(f"{init_from} was trained on base revision {imeta.get('base_revision')!r}, config has "
                           f"{model_cfg.get('revision')!r}")
    arm = imeta["arm"]
    sft_export = read_json(paths.run_dir(imeta["export_run"], create=False) / "export_meta.json")
    export_dir = paths.run_dir(export_run, create=False)
    emeta = read_json(export_dir / "export_meta.json")
    if sorted(emeta["val_qids"]) != sorted(sft_export["val_qids"]) or emeta["pool_sha256"] != sft_export["pool_sha256"]:
        raise RuntimeError(f"{export_run} and {imeta['export_run']} differ in pool or validation databases")
    if arm == "tagged" and tag is None:
        raise ValueError(f"{init_from} is a tagged model: give the efficiency target (--tag)")
    if arm != "tagged":
        tag = None
    run_dir = paths.run_dir(run)

    tok = load_tokenizer(model_cfg)
    budget = cfg["prompt"]["max_prompt_tokens"]
    train_ex, train_units, tstats = build_units(read_export(export_dir, "dpo", "train"), arm, tag, tok, paths, budget)
    val_ex, val_units, vstats = build_units(read_export(export_dir, "dpo", "val"), arm, tag, tok, paths, budget)
    qps, pack, cap = d["questions_per_step"], int(d["pack_size"]), int(d["max_answers_per_seq"])
    steps = [s for e in range(int(d.get("epochs", 1)))
             for s in plan_steps(train_units, qps, pack, stable_seed(d["seed"], "dpo", e))]
    full_steps = len(steps)
    n_steps = min(full_steps, max_steps) if max_steps else full_steps
    if n_steps == 0:
        raise RuntimeError(f"{len(train_units)} questions are fewer than one step of {qps}")

    model = load_model(model_cfg, sft, device, init_dir / "adapter")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=d["lr"], weight_decay=sft.get("weight_decay", 0.0), fused=device == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(full_steps, max(1, math.ceil(d["warmup_ratio"] * full_steps))))
    fp = {"init_from": init_from, "tag": tag, "export_pool_sha256": emeta["pool_sha256"], "n_steps": full_steps,
          "beta": d["beta"], "rpo_alpha": d["rpo_alpha"], "lr": d["lr"], "dpo_version": DPO_VERSION}
    ref_fp = f"{init_from}|{tag}|{emeta['pool_sha256'][:12]}|{budget}"
    ref = reference_logps(model, {"train": (train_ex, train_units), "val": (val_ex, val_units)},
                          run_dir / "ref_logps.jsonl", ref_fp, pack, cap, device)
    ckpt_dir = run_dir / "checkpoints"
    start = load_latest_checkpoint(ckpt_dir, model, opt, sched, fp)
    save_config(cfg, run_dir / "train_config.yaml")
    write_run_meta(run_dir, extra={"command": "dpo", "gpu": gpu, "export_run": export_run, "init_from": init_from,
                                   "tag": tag, "fingerprint": fp})
    val_log = run_dir / "val_log.jsonl"
    if start == 0 and not val_log.exists():  # the starting point: margins are 0, accuracy shows ties
        append_jsonl(val_log, [{"step": 0, **val_metrics(model, val_ex, val_units, ref["val"], d, device)}])
    log.info("dpo from %s (%s, tag %s): %d questions, %d pairs %s; %d steps of %d questions, starting at %d on %s",
             init_from, arm, tag, tstats["n_questions"], tstats["n_pairs"], tstats["by_type"], n_steps, qps, start, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, t_timed, timed_tokens, tokens, last_val, final_val = time.time(), None, 0, 0, None, None
    for step in range(start, n_steps):
        n_pairs = sum(len(train_units[i].pairs) for g in steps[step] for i in g)
        rows, step_tokens = [], 0
        for g in steps[step]:
            for s in seqs([train_units[i] for i in g], cap):
                lp, ce = answer_logps(model, s, train_ex, device)
                losses = []
                for u in s:
                    for c, r, t in u.pairs:
                        loss, m = dpo_terms(lp[c], lp[r], ref["train"][(u.qid, train_ex[c].cand_id)],
                                            ref["train"][(u.qid, train_ex[r].cand_id)], ce[c], d["beta"], d["rpo_alpha"])
                        losses.append(loss)
                        rows.append((loss.item(), m.item(), ce[c].item(), t))
                (sum(losses) / n_pairs).backward()
                step_tokens += packed_len([train_ex[i] for u in s for i in u.answers])
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, sft["max_grad_norm"]))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        tokens += step_tokens
        now = time.time()
        if step > start:
            timed_tokens += step_tokens
        else:
            t_timed = now
        rec = {"step": step + 1, "n_steps": n_steps, **pair_metrics(rows), "lr": sched.get_last_lr()[0],
               "grad_norm": grad_norm, "tokens": step_tokens, "elapsed_s": now - t0,
               "tokens_per_s": timed_tokens / (now - t_timed) if timed_tokens and now > t_timed else None,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
        append_jsonl(run_dir / "train_log.jsonl", [rec])
        log.info("step %d/%d loss %.4f reward acc %.2f margin %.3f lr %.2e %.0f tok/s", step + 1, n_steps, rec["loss"],
                 rec["reward_acc"], rec["margin"], rec["lr"], rec["tokens_per_s"] or 0)
        if (step + 1) % d["save_every"] == 0:
            save_checkpoint(ckpt_dir, step + 1, model, opt, sched, fp, sft.get("keep_checkpoints", 2))
        if (step + 1) % d["eval_every"] == 0:
            final_val = val_metrics(model, val_ex, val_units, ref["val"], d, device)
            append_jsonl(val_log, [{"step": step + 1, **final_val}])
            last_val = step + 1

    if last_val != n_steps:
        final_val = val_metrics(model, val_ex, val_units, ref["val"], d, device)
        append_jsonl(val_log, [{"step": n_steps, **final_val}])
    if n_steps % d["save_every"]:
        save_checkpoint(ckpt_dir, n_steps, model, opt, sched, fp, sft.get("keep_checkpoints", 2))
    adapter, tmp = run_dir / "adapter", run_dir / "adapter.tmp"
    model.save_pretrained(str(tmp))
    if adapter.exists():
        import shutil

        shutil.rmtree(adapter)
    tmp.replace(adapter)
    meta = {"arm": arm, "method": f"dpo-{arm}", "tag": tag, "epoch": imeta["epoch"], "export_run": imeta["export_run"],
            "dpo_export_run": export_run, "init_from": init_from, "export_pool_sha256": emeta["pool_sha256"],
            "base_id": model_cfg["id"], "base_revision": model_cfg.get("revision"), "beta": d["beta"],
            "rpo_alpha": d["rpo_alpha"], "lr": d["lr"], "n_steps": n_steps, "gpu": gpu, "train_pairs": tstats,
            "val_pairs": vstats, "val_final": final_val, "tokens_this_session": tokens,
            "complete": n_steps == full_steps, "dpo_version": DPO_VERSION}
    write_json(run_dir / "adapter_meta.json", meta)
    log.info("adapter written to %s", adapter)
    return meta
