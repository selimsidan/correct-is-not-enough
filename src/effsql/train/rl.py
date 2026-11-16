"""Reinforcement learning with an execution reward (GRPO-style), from a finished SFT adapter; resumable.

Why: DPO learns from a fixed pool of other models' queries, most of which our model would rarely write,
and the pool has a cheaper honest query for few questions. Here the model learns from its own answers:
it writes several answers per train question, every answer is executed, and answers that are correct
and cheaper are made more likely than the question's other answers. The cost-guided search pilot showed
that this 7B model finds cheaper queries when it may try several times; RL trains it to write them in
one shot.

An iteration k (runs/<run>/iters/<k>/):
  1. gen (GPU, own process): `questions_per_iter` new train questions (never the validation databases),
     `samples` answers each, by vLLM with the current adapter at temperature 1 (exactly the model's
     distribution). The prompt is the one prediction uses (train.data.train_messages, same budget).
     -> gen.jsonl with the prompt and answer token ids.
  2. score (CPU): every distinct answer is executed (correct = gold's result; then VM steps, capped at
     step_cap_factor x gold) and rewarded (see `reward`). -> rewards.jsonl, score_stats.json.
  3. train (GPU, own process): advantage = reward - mean reward of the question's answers; questions
     whose answers all score the same are skipped. The log probabilities under the starting weights are
     computed first (no dropout anywhere), then one pass of clipped-ratio policy updates (PPO) in steps of
     `questions_per_step` questions, loss averaged over answer tokens. A question's answers share their
     prompt in one sequence (sft.group_token_logps), so the long schema is computed once per question.
     -> adapter/ (read by the next gen), checkpoint (optimizer state), done.json, a line in rl_log.jsonl.
Generation and training run in separate processes because vLLM and the training model do not fit on a
40 GB A100 together. After the last iteration the adapter is copied to runs/<run>/adapter with
adapter_meta.json, so `predict`, `eval-val`, `select` and `report` treat it like any other adapter.

Resume: running the same command again continues with the first iteration that has no done.json; an
interrupted step is redone from its inputs (at most one iteration of work is lost). The settings are
fixed in rl_meta.json on the first call; other settings need a new run name.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.gen.backend import stable_seed
from effsql.gen.parse import extract_sql
from effsql.gen.prompts import load_templates
from effsql.gen.sample import schemas_for
from effsql.io import append_jsonl, read_json, read_jsonl, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
RL_VERSION = 1
CORRECT = "correct"


# Layout -----------------------------------------------------------------------------------------
def iter_dir(run_dir: Path, k: int) -> Path:
    return run_dir / "iters" / f"{k:03d}"


def adapter_dir(paths: Paths, run: str, meta: dict[str, Any], k: int) -> Path:
    """Adapter that generates iteration k: the starting SFT adapter, then the previous iteration's."""
    if k == 0:
        return paths.run_dir(meta["init_from"], create=False) / "adapter"
    return iter_dir(paths.run_dir(run), k - 1) / "adapter"


def iter_questions(meta: dict[str, Any], k: int) -> list[str]:
    n = meta["fingerprint"]["rl"]["questions_per_iter"]
    return meta["questions"][k * n:(k + 1) * n]


def _write_jsonl_atomic(path: Path, recs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    tmp.replace(path)


# Setup ------------------------------------------------------------------------------------------
def rl_settings(cfg: dict[str, Any], pilot: bool, settings: str | None = None) -> dict[str, Any]:
    """train.yaml `rl`, with the overrides of another section (e.g. `rl_eff`; nested dicts are merged)
    and, for a pilot, `rl.pilot`."""
    rl = {k: v for k, v in cfg["rl"].items() if k != "pilot"}
    for k, v in (cfg[settings] if settings else {}).items():
        rl[k] = {**rl[k], **v} if isinstance(v, dict) and isinstance(rl.get(k), dict) else v
    if pilot:
        rl.update(cfg["rl"]["pilot"])
    return rl


def focus_questions(paths: Paths, focus: dict[str, Any], candidates: set[str]) -> set[str]:
    """Questions where efficiency can be learned:
      pool_run: a correct query of the pool costs at most max_rel_cost x gold (quality ok or verified,
        no first-row pick: the strict rule);
      history_runs: in an earlier RL run, the model's correct answers to the question differed in cost
        by at least min_spread (max / min)."""
    import pandas as pd

    from effsql.pool.quality import analyze

    out: set[str] = set()
    if focus.get("pool_run"):
        df = pd.read_parquet(paths.run_dir(focus["pool_run"], create=False) / "pool_train.parquet",
                             columns=["qid", "is_gold", "correct", "quality", "steps_capped", "rel_cost", "sql_original"])
        df = df[df["qid"].isin(candidates) & df["correct"] & ~df["is_gold"] & df["quality"].isin(["ok", "verified"])
                & ~df["steps_capped"] & df["rel_cost"].notna() & (df["rel_cost"] <= focus["max_rel_cost"])]
        for q, sql in zip(df["qid"], df["sql_original"]):
            if q not in out and not analyze(sql)["flags"]:
                out.add(q)
    for run in focus.get("history_runs") or []:
        for f in sorted((paths.run_dir(run, create=False) / "iters").glob("*/rewards.jsonl")):
            for r in read_jsonl(f):
                rc = [x for x, st in zip(r["rel_cost"], r["status"]) if st == CORRECT and x]
                if len(rc) >= 2 and max(rc) / min(rc) >= focus["min_spread"]:
                    out.add(r["qid"])
    return out & candidates


def setup(run: str, init_from: str, *, paths: Paths, cfg: dict[str, Any], gen_cfg: dict[str, Any],
          exec_cfg: dict[str, Any], pilot: bool = False, workers: int | None = None,
          settings: str | None = None) -> dict[str, Any]:
    """rl_meta.json: the settings (fingerprint) and the ordered train questions, fixed on the first call."""
    from effsql.measure.gold import ensure_gold_costs
    from effsql.train.sft import base_model_cfg

    run_dir = paths.run_dir(run)
    imeta = read_json(paths.run_dir(init_from, create=False) / "adapter_meta.json")
    if not imeta.get("complete"):
        raise RuntimeError(f"{init_from} has not finished training")
    model_cfg = base_model_cfg(cfg, gen_cfg)
    if imeta.get("base_revision") != model_cfg.get("revision"):
        raise RuntimeError(f"{init_from} was trained on base revision {imeta.get('base_revision')!r}, config has "
                           f"{model_cfg.get('revision')!r}")
    if imeta["arm"] == "tagged" and not imeta.get("tag"):
        raise ValueError(f"{init_from} is a tagged model without a fixed efficiency target")
    rl = rl_settings(cfg, pilot, settings)
    fp = {"init_from": init_from, "base_revision": model_cfg.get("revision"), "rl": rl, "settings": settings,
          "max_prompt_tokens": cfg["prompt"]["max_prompt_tokens"], "rl_version": RL_VERSION}
    path = run_dir / "rl_meta.json"
    if path.exists():
        saved = read_json(path)
        if saved["fingerprint"] != fp:
            raise RuntimeError(f"{run} was started with other settings ({saved['fingerprint']} vs {fp}); "
                               "use a new run name")
        return saved
    held = sorted(read_json(paths.run_dir(imeta["export_run"], create=False) / "export_meta.json")["holdout_db_ids"])
    cands = [e for e in load_examples("train", paths) if e.db_id not in set(held)]
    golds = ensure_gold_costs("train", {e.qid: (e.db_id, e.gold_sql) for e in cands}, paths, exec_cfg, workers)
    ok = sorted(e.qid for e in cands if golds[e.qid]["gold_status"] == "ok" and golds[e.qid].get("vm_steps") is not None)
    if rl.get("questions", "all") == "focus":
        ok = sorted(focus_questions(paths, rl["focus"], set(ok)))
    need = rl["iterations"] * rl["questions_per_iter"]
    if len(ok) < need:  # the last iterations get fewer questions
        log.warning("%d questions available, %d iterations of %d would use %d", len(ok), rl["iterations"],
                    rl["questions_per_iter"], need)
    rng = np.random.default_rng(stable_seed(rl["seed"], "rl_questions"))
    order = [ok[i] for i in rng.permutation(len(ok))][:need]
    meta = {"fingerprint": fp, "init_from": init_from, "arm": imeta["arm"], "tag": imeta.get("tag"),
            "epoch": imeta.get("epoch"), "export_run": imeta["export_run"], "holdout_db_ids": held,
            "n_questions_available": len(ok), "questions": order, "pilot": pilot}
    write_json(path, meta)
    return meta


# 1. Generation ----------------------------------------------------------------------------------
Sampler = Callable[[list[list[int]], list[int]], list[list[dict[str, Any]]]]


class VLLMSampler:
    """vLLM with one LoRA adapter; prompts go in as token ids, so they are exactly the training prompts.
    Each answer: token ids, text, finish reason and vLLM's summed log probability (a check against the
    trainer's)."""

    def __init__(self, model_cfg: dict[str, Any], adapter: Path, rl: dict[str, Any], gen_cfg: dict[str, Any]):
        from vllm import LLM  # GPU-only import
        from vllm.lora.request import LoRARequest

        acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        self.llm = LLM(model=model_cfg["id"], revision=model_cfg.get("revision"), dtype=model_cfg.get("dtype", "auto"),
                       max_model_len=model_cfg.get("max_model_len"), gpu_memory_utilization=rl["gpu_memory_utilization"],
                       enable_prefix_caching=True, seed=gen_cfg.get("seed", 0), enable_lora=True, max_loras=1,
                       max_lora_rank=int(acfg["r"]))
        self.lora = LoRARequest("rl", 1, str(adapter))
        self.rl = rl

    def __call__(self, prompts: list[list[int]], seeds: list[int]) -> list[list[dict[str, Any]]]:
        from vllm import SamplingParams

        sps = [SamplingParams(n=self.rl["samples"], temperature=1.0, top_p=1.0, max_tokens=self.rl["max_tokens"],
                              seed=s, logprobs=0) for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps, lora_request=self.lora,
                                 use_tqdm=False)
        res = []
        for out in outs:
            answers = []
            for o in out.outputs:
                lp = None
                if o.logprobs:
                    lp = float(sum(d[t].logprob for t, d in zip(o.token_ids, o.logprobs) if d and t in d))
                answers.append({"token_ids": list(o.token_ids), "text": o.text, "finish_reason": o.finish_reason,
                                "vllm_logp": lp})
            res.append(answers)
        return res


def gen_iter(run: str, k: int, *, paths: Paths | None = None, cfg: dict[str, Any] | None = None,
             gen_cfg: dict[str, Any] | None = None, sampler: Sampler | None = None) -> dict[str, Any]:
    from effsql.train.data import train_messages
    from effsql.train.sft import (
        base_model_cfg,
        end_token_id,
        load_tokenizer,
        make_counter,
        prompt_text,
    )

    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    run_dir = paths.run_dir(run)
    d = iter_dir(run_dir, k)
    if (d / "gen.jsonl").exists():
        return read_json(d / "gen_stats.json")
    meta = read_json(run_dir / "rl_meta.json")
    rl = meta["fingerprint"]["rl"]
    model_cfg = base_model_cfg(cfg, gen_cfg)
    tok = load_tokenizer(model_cfg)
    count, eot = make_counter(tok), end_token_id(tok)
    eos = {eot, tok.eos_token_id}
    budget = meta["fingerprint"]["max_prompt_tokens"]
    examples = {e.qid: e for e in load_examples("train", paths)}
    qids = iter_questions(meta, k)
    schemas = schemas_for("train", {examples[q].db_id for q in qids}, paths)
    templates = load_templates()
    prompts, skipped = {}, []
    for q in sorted(qids, key=lambda q: (examples[q].db_id, q)):  # one database after another: prefix caching
        e = examples[q]
        msgs, level, n = train_messages(templates, meta["arm"], e, schemas[e.db_id], budget, count, tag=meta["tag"])
        if n > budget:
            skipped.append(q)
            continue
        prompts[q] = (tok(prompt_text(tok, msgs), add_special_tokens=False)["input_ids"], level)
    t0 = time.time()
    sampler = sampler or VLLMSampler(model_cfg, adapter_dir(paths, run, meta, k), rl, gen_cfg)
    t_load = time.time() - t0
    t1 = time.time()
    outs = sampler([v[0] for v in prompts.values()], [stable_seed(rl["seed"], "rl", q, k) for q in prompts])
    t_gen = time.time() - t1
    recs, n_tokens = [], 0
    for (q, (ids, level)), answers in zip(prompts.items(), outs):
        for a in answers:
            if a["finish_reason"] == "stop" and (not a["token_ids"] or a["token_ids"][-1] not in eos):
                a["token_ids"] = a["token_ids"] + [eot]  # the end of turn the sampler stopped on
            n_tokens += len(a["token_ids"])
        recs.append({"qid": q, "iteration": k, "prompt_ids": ids, "prompt_level": level, "answers": answers})
    stats = {"iteration": k, "n_questions": len(recs), "n_skipped_too_long": len(skipped), "n_answers": sum(len(r["answers"]) for r in recs),
             "generated_tokens": n_tokens, "load_s": t_load, "gen_s": t_gen,
             "tokens_per_s": n_tokens / t_gen if t_gen > 0 else None}
    _write_jsonl_atomic(d / "gen.jsonl", recs)
    write_json(d / "gen_stats.json", stats)
    log.info("rl iteration %d: %d answers for %d questions, %.0f s (+ %.0f s loading)", k, stats["n_answers"],
             len(recs), t_gen, t_load)
    return stats


# 2. Scoring -------------------------------------------------------------------------------------
def eff_term(steps: float, gold_steps: float, rw: dict[str, Any]) -> float:
    s = rw["smooth_steps"]
    e = math.log2((gold_steps + s) / (steps + s))
    return max(-rw["eff_clip"], min(rw["eff_clip"], e))


def reward(status: str, steps: float | None, gold_steps: float, rw: dict[str, Any]) -> float:
    """0 unless correct; a correct answer gets 1 + eff_weight * e / eff_clip (0.1 to 1.9 with eff_weight 0.9). A
    correct answer without a step count (counting failed or stopped at the cap) counts as the costliest."""
    if status != CORRECT:
        return 0.0
    e = -rw["eff_clip"] if steps is None else eff_term(steps, gold_steps, rw)
    return 1.0 + rw["eff_weight"] * e / rw["eff_clip"]


def score_iter(run: str, k: int, *, paths: Paths | None = None, exec_cfg: dict[str, Any] | None = None,
               workers: int | None = None) -> dict[str, Any]:
    from effsql.exec.engine import execute_two_pass
    from effsql.measure.correctness import hashes_match
    from effsql.measure.cost import rel_cost
    from effsql.measure.gold import ensure_gold_costs
    from effsql.pool.quality import analyze, foreign_literals

    paths = paths or load_paths()
    exec_cfg = exec_cfg or load_config("evaluate.yaml")["exec"]
    workers = workers or exec_cfg.get("workers")
    run_dir = paths.run_dir(run)
    d = iter_dir(run_dir, k)
    if (d / "rewards.jsonl").exists():
        return read_json(d / "score_stats.json")
    meta = read_json(run_dir / "rl_meta.json")
    rl = meta["fingerprint"]["rl"]
    rw = rl["reward"]
    gen = read_jsonl(d / "gen.jsonl")
    examples = {e.qid: e for e in load_examples("train", paths)}
    golds = ensure_gold_costs("train", {g["qid"]: (examples[g["qid"]].db_id, examples[g["qid"]].gold_sql) for g in gen},
                              paths, exec_cfg, workers)
    t0 = time.time()
    sqls: dict[tuple[str, str], dict[str, Any]] = {}
    for g in gen:
        for a in g["answers"]:
            sql, _ = extract_sql(a["text"])
            a["sql"] = sql
            if sql and a["finish_reason"] != "length":
                sqls.setdefault((g["qid"], sql), {})
    jobs = []
    for i, (q, sql) in enumerate(sqls):
        cap = int(rl["step_cap_factor"] * max(golds[q]["vm_steps"], 1000))
        jobs.append({"key": i, "db_path": paths.local_db_path("train", examples[q].db_id), "sql": sql,
                     "count_exec": {"step_budget": cap}, "qid": q})
    keys = list(sqls)

    def is_correct(r: dict[str, Any]) -> bool:
        return bool(r["ok"] and hashes_match(r["result_hash"], golds[keys[r["key"]][0]]["result_hash"]))

    ecfg = {**exec_cfg, "timeout_s": rl["exec_timeout_s"], "count_timeout_s": rl["count_timeout_s"]}
    res = execute_two_pass(jobs, paths, ecfg, count_if=is_correct, workers=workers) if jobs else {}
    gold_flags = {q: set(analyze(examples[q].gold_sql)["flags"]) for q in {g["qid"] for g in gen}}
    for i, key in enumerate(keys):
        q, sql = key
        r = res[i]
        e = examples[q]
        if not is_correct(r):
            status = r["error_type"] or "wrong_result"
        elif set(analyze(sql)["flags"]) - gold_flags[q]:
            status = "first_row_pick"
        elif foreign_literals(sql, e.gold_sql, e.question, e.evidence):
            status = "foreign_constant"
        else:
            status = CORRECT
        steps = r.get("vm_steps") if status == CORRECT else None
        sqls[key] = {"status": status, "steps": steps}

    recs, statuses, rels, rewards = [], [], [], []
    for g in gen:
        q = g["qid"]
        gs = golds[q]["vm_steps"]
        row = {"qid": q, "iteration": k, "gold_steps": gs, "status": [], "steps": [], "rel_cost": [], "reward": []}
        for a in g["answers"]:
            if a["finish_reason"] == "length":
                x = {"status": "cut_off", "steps": None}
            elif not a["sql"]:
                x = {"status": "no_sql", "steps": None}
            else:
                x = sqls[(q, a["sql"])]
            rc = rel_cost(x["steps"], gs, exec_cfg["progress_n"]) if x["steps"] is not None else None
            row["status"].append(x["status"])
            row["steps"].append(x["steps"])
            row["rel_cost"].append(rc)
            row["reward"].append(reward(x["status"], x["steps"], gs, rw))
            statuses.append(x["status"])
            rewards.append(row["reward"][-1])
            if x["status"] == CORRECT and rc is not None:
                rels.append(rc)
        recs.append(row)
    n = len(statuses)
    rels_a = np.array(rels) if rels else np.array([1.0])
    per_q = [np.array(r["reward"]) for r in recs]
    stats = {
        "iteration": k, "n_questions": len(recs), "n_answers": n, "n_distinct_executed": len(jobs),
        "status": {s: statuses.count(s) for s in sorted(set(statuses))},
        "pct_correct": 100 * statuses.count(CORRECT) / max(n, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "correct_geomean_rel_cost": float(np.exp(np.mean(np.log(rels_a)))) if rels else None,
        "correct_pct_2x_cheaper": 100 * float(np.mean(rels_a <= 0.5)) if rels else None,
        "correct_pct_costlier_10pct": 100 * float(np.mean(rels_a > 1.1)) if rels else None,
        "n_questions_with_signal": int(sum(1 for r in per_q if len(r) and r.max() - r.min() > 1e-6)),
        "score_s": time.time() - t0,
    }
    _write_jsonl_atomic(d / "rewards.jsonl", recs)
    write_json(d / "score_stats.json", stats)
    log.info("rl iteration %d: %.1f%% correct, mean reward %.3f, %d of %d questions with signal", k,
             stats["pct_correct"], stats["mean_reward"] or 0, stats["n_questions_with_signal"], len(recs))
    return stats


# 3. Training ------------------------------------------------------------------------------------
def advantages(rewards: list[float]) -> list[float] | None:
    """reward - mean of the question's answers; None when they all score the same (no signal)."""
    r = np.asarray(rewards, dtype=float)
    if len(r) < 2 or r.max() - r.min() <= 1e-6:
        return None
    return (r - r.mean()).tolist()


def ppo_terms(new_lp: Any, old_lp: Any, adv: float, eps: float) -> tuple[Any, Any, Any]:
    """(summed clipped surrogate loss, clipped token count, summed old - new log ratio) of one answer."""
    import torch

    ratio = torch.exp(new_lp - old_lp)
    s1, s2 = ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    loss = -torch.minimum(s1, s2).sum()
    clipped = ((ratio < 1 - eps) | (ratio > 1 + eps)).sum()
    return loss, clipped, (old_lp - new_lp).detach().sum()


def answer_chunks(n_targets: list[int], cap: int) -> list[list[int]]:
    """Indices of a question's answers in sequences of at most `cap` answer tokens (at least one answer
    each): bounds the memory of the vocabulary-sized log probabilities. Exact either way."""
    out: list[list[int]] = []
    n = 0
    for i, t in enumerate(n_targets):
        if out and n + t <= cap:
            out[-1].append(i)
            n += t
        else:
            out.append([i])
            n = t
    return out


def token_logps(model: Any, exs: list[Any], cap: int, device: str) -> list[Any]:
    from effsql.train.sft import group_token_logps

    res: list[Any] = [None] * len(exs)
    for idx in answer_chunks([e.n_target for e in exs], cap):
        for i, lp in zip(idx, group_token_logps(model, [exs[i] for i in idx], device)):
            res[i] = lp
    return res


def train_iter(run: str, k: int, *, paths: Paths | None = None, cfg: dict[str, Any] | None = None,
               gen_cfg: dict[str, Any] | None = None, device: str | None = None) -> dict[str, Any]:
    import torch

    from effsql.train.sft import (
        TrainExample,
        base_model_cfg,
        check_gpu,
        load_model,
    )

    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    run_dir = paths.run_dir(run)
    d = iter_dir(run_dir, k)
    if (d / "done.json").exists():
        return read_json(d / "done.json")
    meta = read_json(run_dir / "rl_meta.json")
    rl, sft = meta["fingerprint"]["rl"], cfg["sft"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = check_gpu(device, sft)
    model_cfg = base_model_cfg(cfg, gen_cfg)
    torch.manual_seed(stable_seed(rl["seed"], "rl_train", k))
    gen = {g["qid"]: g for g in read_jsonl(d / "gen.jsonl")}
    rewards = {r["qid"]: r for r in read_jsonl(d / "rewards.jsonl")}

    units = []  # (qid, examples, advantages)
    for q, g in gen.items():
        adv = advantages(rewards[q]["reward"])
        if adv is None:
            continue
        exs, a_kept = [], []
        for i, (a, x) in enumerate(zip(g["answers"], adv)):
            if a["token_ids"]:
                exs.append(TrainExample(q, str(i), g["prompt_ids"] + a["token_ids"], len(a["token_ids"]), q))
                a_kept.append(x)
        if exs:
            units.append((q, exs, a_kept))

    t0 = time.time()
    model = load_model(model_cfg, sft, device, adapter_dir(paths, run, meta, k))
    for m in model.modules():  # no dropout: the ratio must compare the same function before and after
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rl["lr"], weight_decay=sft.get("weight_decay", 0.0), fused=device == "cuda")
    global_step = 0
    if k > 0:
        state = torch.load(run_dir / "checkpoints" / f"iter_{k - 1:03d}.pt", map_location="cpu", weights_only=False)
        if state["fingerprint"] != meta["fingerprint"]:
            raise RuntimeError("checkpoint of a different setup")
        opt.load_state_dict(state["opt"])
        global_step = state["global_step"]
    t_load = time.time() - t0

    t1 = time.time()
    cap = rl.get("max_answer_tokens_per_seq", 2048)
    model.eval()
    old, vdiff = {}, []
    with torch.no_grad():
        for q, exs, _ in units:
            old[q] = [x.detach() for x in token_logps(model, exs, cap, device)]
            for ex, lp in zip(exs, old[q]):
                v = gen[q]["answers"][int(ex.cand_id)].get("vllm_logp")
                if v is not None:
                    vdiff.append(abs(v - float(lp.sum())) / max(ex.n_target, 1))
    model.train()
    rng = np.random.default_rng(stable_seed(rl["seed"], "rl_order", k))
    order = [units[i] for i in rng.permutation(len(units))]
    qps = rl["questions_per_step"]
    steps = [order[i:i + qps] for i in range(0, len(order), qps)]
    rows = []
    for step in steps:
        n_tok = sum(ex.n_target for _, exs, _ in step for ex in exs)
        loss_v, clipped, kl = 0.0, 0, 0.0
        for q, exs, adv in step:
            new = token_logps(model, exs, cap, device)
            terms = [ppo_terms(n, o.to(n.device), a, rl["clip_eps"]) for n, o, a in zip(new, old[q], adv)]
            loss = sum(t[0] for t in terms) / n_tok
            loss.backward()
            loss_v += float(loss.detach())
            clipped += int(sum(int(t[1]) for t in terms))
            kl += float(sum(float(t[2]) for t in terms))
        lr = rl["lr"] * min(1.0, (global_step + 1) / max(1, rl["warmup_steps"]))
        for pg in opt.param_groups:
            pg["lr"] = lr
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, sft["max_grad_norm"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        global_step += 1
        rows.append({"loss": loss_v, "clip_frac": clipped / max(n_tok, 1), "approx_kl": kl / max(n_tok, 1),
                     "grad_norm": grad_norm, "lr": lr, "tokens": n_tok})
    t_train = time.time() - t1

    out = d / "adapter"
    tmp = d / "adapter.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    ckpt = run_dir / "checkpoints" / f"iter_{k:03d}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"opt": opt.state_dict(), "global_step": global_step, "fingerprint": meta["fingerprint"]},
               ckpt.with_suffix(".pt.tmp"))
    ckpt.with_suffix(".pt.tmp").replace(ckpt)

    # the finished model or a snapshot first: an iteration counts as done only after it
    n_iters = rl["iterations"]
    every = rl.get("snapshot_every")
    if k == n_iters - 1:
        write_model(run_dir, out, meta, model_cfg, k + 1, global_step, gpu)
    elif every and (k + 1) % every == 0:
        write_model(paths.run_dir(f"{run}_it{k + 1}"), out, meta, model_cfg, k + 1, global_step, gpu)
    score = read_json(d / "score_stats.json")
    rec = {"iteration": k, **{f"score_{a}": score.get(a) for a in ("pct_correct", "mean_reward", "correct_geomean_rel_cost",
                                                                    "correct_pct_2x_cheaper", "correct_pct_costlier_10pct")},
           "status": score.get("status"), "n_questions": len(gen), "n_questions_with_signal": len(units),
           "n_steps": len(steps), "global_step": global_step,
           "loss": float(np.mean([r["loss"] for r in rows])) if rows else None,
           "clip_frac": float(np.mean([r["clip_frac"] for r in rows])) if rows else None,
           "approx_kl": float(np.mean([r["approx_kl"] for r in rows])) if rows else None,
           "grad_norm": float(np.mean([r["grad_norm"] for r in rows])) if rows else None,
           "vllm_vs_trainer_logp_per_token": float(np.mean(vdiff)) if vdiff else None,
           "load_s": t_load, "train_s": t_train, "gpu": gpu,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
    append_jsonl(run_dir / "rl_log.jsonl", [rec])
    write_json(d / "done.json", rec)
    log.info("rl iteration %d trained: %d questions with signal, %d steps, clip %.3f, kl %.4f", k, len(units),
             len(steps), rec["clip_frac"] or 0, rec["approx_kl"] or 0)

    for j in range(k - rl.get("keep_iterations", 2) + 1):  # older adapters and optimizer states
        old_d = iter_dir(run_dir, j) / "adapter"
        if old_d.exists():
            shutil.rmtree(old_d)
        old_c = run_dir / "checkpoints" / f"iter_{j:03d}.pt"
        if old_c.exists():
            old_c.unlink()
    return rec


def write_model(dest: Path, adapter: Path, meta: dict[str, Any], model_cfg: dict[str, Any], n_iters: int,
                global_step: int, gpu: str) -> None:
    """The adapter after n_iters iterations as a finished run (adapter/ and adapter_meta.json), so
    `predict`, `eval-val`, `select` and `report` treat it like any other adapter."""
    final = dest / "adapter"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(adapter, final)
    write_json(dest / "adapter_meta.json", {
        "arm": meta["arm"], "method": f"rl-{meta['arm']}", "tag": meta["tag"], "epoch": meta["epoch"],
        "export_run": meta["export_run"], "init_from": meta["init_from"], "base_id": model_cfg["id"],
        "base_revision": model_cfg.get("revision"), "rl": meta["fingerprint"]["rl"], "iterations": n_iters,
        "global_steps": global_step, "gpu": gpu, "complete": True, "rl_version": RL_VERSION})


# Orchestration ----------------------------------------------------------------------------------
def subprocess_step(step: str, run: str, k: int) -> None:
    """A GPU step in its own process, so its GPU memory is fully released when it ends."""
    subprocess.run([sys.executable, "-m", "effsql.train", "rl-step", step, "--run", run, "--iter", str(k)], check=True)


def run_rl(run: str, init_from: str, *, pilot: bool = False, paths: Paths | None = None,
           cfg: dict[str, Any] | None = None, gen_cfg: dict[str, Any] | None = None,
           exec_cfg: dict[str, Any] | None = None, workers: int | None = None,
           step_runner: Callable[[str, str, int], None] | None = None, settings: str | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    exec_cfg = exec_cfg or load_config("evaluate.yaml")["exec"]
    step_runner = step_runner or subprocess_step
    meta = setup(run, init_from, paths=paths, cfg=cfg, gen_cfg=gen_cfg, exec_cfg=exec_cfg, pilot=pilot, workers=workers,
                 settings=settings)
    run_dir = paths.run_dir(run)
    n_iters = meta["fingerprint"]["rl"]["iterations"]
    log.info("rl %s from %s: %d iterations of %d questions (%d available)", run, init_from, n_iters,
             meta["fingerprint"]["rl"]["questions_per_iter"], meta["n_questions_available"])
    for k in range(n_iters):
        d = iter_dir(run_dir, k)
        if (d / "done.json").exists():
            continue
        t = {}
        t0 = time.time()
        if not (d / "gen.jsonl").exists():
            step_runner("gen", run, k)
        t["gen_total_s"] = time.time() - t0
        t0 = time.time()
        score_iter(run, k, paths=paths, exec_cfg=exec_cfg, workers=workers)
        t["score_total_s"] = time.time() - t0
        t0 = time.time()
        step_runner("train", run, k)
        t["train_total_s"] = time.time() - t0
        write_json(d / "times.json", t)
    return summarize(run, paths=paths, cfg=cfg)


def summarize(run: str, *, paths: Paths, cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-iteration log, and for a pilot the projected time and cost of the full run."""
    run_dir = paths.run_dir(run)
    meta = read_json(run_dir / "rl_meta.json")
    rl = meta["fingerprint"]["rl"]
    iters = [read_json(iter_dir(run_dir, k) / "done.json") for k in range(rl["iterations"])
             if (iter_dir(run_dir, k) / "done.json").exists()]
    out: dict[str, Any] = {"run": run, "iterations_done": len(iters), "iterations": iters}
    last = iter_dir(run_dir, len(iters) - 1) if iters else None
    if meta.get("pilot") and last is not None and (last / "times.json").exists():
        g, s, t = read_json(last / "gen_stats.json"), read_json(last / "score_stats.json"), read_json(last / "done.json")
        full = cfg["rl"]
        scale = full["questions_per_iter"] / rl["questions_per_iter"]
        per_iter = g["load_s"] + g["gen_s"] * scale + s["score_s"] * scale + t["load_s"] + t["train_s"] * scale + 60
        hours = per_iter * full["iterations"] / 3600
        rate = next((v for k_, v in (cfg["sft"].get("units_per_hour") or {}).items() if k_.lower() in str(t["gpu"]).lower()), None)
        out["projection"] = {"seconds_per_iteration": per_iter, "iterations": full["iterations"],
                             "questions_per_iteration": full["questions_per_iter"], "hours": hours,
                             "units": hours * rate if rate else None, "gpu": t["gpu"],
                             "basis": "last pilot iteration; model loading counted once per iteration, the rest "
                                      "scaled by questions; plus a minute per iteration for process start-up"}
    write_json(run_dir / "rl_summary.json", out)
    return out


def format_summary(s: dict[str, Any]) -> str:
    cols = ["iteration", "score_pct_correct", "score_mean_reward", "score_correct_geomean_rel_cost",
            "score_correct_pct_2x_cheaper", "score_correct_pct_costlier_10pct", "n_questions_with_signal", "n_steps",
            "clip_frac", "approx_kl", "vllm_vs_trainer_logp_per_token"]
    head = ["iter", "correct%", "reward", "cost(geo)", "2x cheap%", "costlier%", "signal q", "steps", "clip", "kl",
            "logp diff"]
    lines = ["".join(f"{h:>11}" for h in head)]
    for r in s["iterations"]:
        lines.append("".join(f"{'-' if r.get(c) is None else (f'{r[c]:.4f}' if isinstance(r[c], float) else r[c]):>11}"
                             for c in cols))
    if "projection" in s:
        p = s["projection"]
        lines.append(f"projected full run on {p['gpu']}: {p['iterations']} iterations of {p['questions_per_iteration']} "
                     f"questions, {p['hours']:.1f} h" + (f", about {p['units']:.0f} units" if p["units"] else ""))
    return "\n".join(lines)
