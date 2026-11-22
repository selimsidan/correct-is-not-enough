"""`python -m effsql.judge latency` (A100, notebook 84): the whole answer path, timed question by question.

One vLLM engine holds the base model once and both adapters (generator and judge). Per question, one at a
time, as a user would ask:
  1. generate  the generator's greedy answer and its samples per prompt variant (judge.yaml candidates),
               one batched call; distinct candidates as in `candidates`;
  2. run       every distinct candidate compiled and run on the database in warm worker processes with the
               answer-time cap (vote.timeout_s); started right after step 1, so it overlaps step 3;
  3. judge     one prompt per candidate, one generated token with the judge adapter; p = sigmoid of the
               log-probability of "yes" minus that of "no";
  4. choose    every method of vote.methods (vote.py).
A judge trained with results (judge_meta with_results) reads each candidate's result, so step 3 waits for
step 2 and run_wait is the time between the end of generation and the start of judging.
Writes runs/<out>/latency_<split>.jsonl (per question: seconds per stage, total, candidates, the answer
of each method) and latency_<split>.json (median, p90 and max per stage, warm-up questions left out).
The answers are for checking only; accuracy comes from the offline runs (notebooks 81 to 83).
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from typing import Any

import numpy as np

from effsql.config import load_config
from effsql.io import append_jsonl, done_ids, iter_jsonl, read_json, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
STAGES = ("generate", "judge", "run_wait", "choose", "total")


def judge_p(logprobs: dict[int, float], yes_no: tuple[int, int]) -> tuple[float, bool]:
    """p from the first generated token's top log-probabilities; a token missing from the top list gets the
    lowest listed value (an upper bound). Returns (p, whether one was missing)."""
    floor = min(logprobs.values()) if logprobs else 0.0
    y, n = (logprobs.get(t) for t in yes_no)
    x = (y if y is not None else floor) - (n if n is not None else floor)
    p = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
    return p, y is None or n is None


def summarize(rows: list[dict[str, Any]], warmup: int) -> dict[str, Any]:
    rows = rows[warmup:]
    out: dict[str, Any] = {"n_questions": len(rows), "warmup_left_out": warmup}
    for s in STAGES:
        x = np.asarray([r["seconds"][s] for r in rows], dtype=float)
        if len(x):
            out[s] = {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max()),
                      "mean": float(x.mean())}
    out["pct_over_3s"] = 100 * float(np.mean([r["seconds"]["total"] > 3 for r in rows])) if rows else None
    out["candidates_per_question"] = float(np.mean([r["n_candidates"] for r in rows])) if rows else None
    return out


def format_latency(s: dict[str, Any]) -> str:
    lines = [(f"{s['n_questions']} questions answered one at a time ({s['warmup_left_out']} warm-up left out), "
              f"{s['candidates_per_question']:.1f} distinct candidates each"),
             f"  {'stage':<10}{'median':>8}{'p90':>8}{'max':>8}  seconds"]
    for st in STAGES:
        if st in s:
            lines.append(f"  {st:<10}{s[st]['median']:>8.2f}{s[st]['p90']:>8.2f}{s[st]['max']:>8.2f}")
    wait = ("before the judge (it reads the results)" if s.get("judge_reads_results")
            else "after the judge finished")
    lines.append(f"  answers over 3 s: {s['pct_over_3s']:.1f}%   (run_wait: time waiting for the queries {wait})")
    return "\n".join(lines)


def measure_latency(judge_run: str, out: str, split: str = "dev", *, paths: Paths | None = None,
                    jcfg: dict[str, Any] | None = None, n: int | None = None) -> dict[str, Any]:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi") or subprocess.run(["nvidia-smi"], check=False,
                                                        capture_output=True).returncode != 0:
        raise RuntimeError("this runtime has no GPU: Runtime > Change runtime type > A100, then run the notebook "
                           "again (finished questions are kept)")
    from vllm import LLM, SamplingParams  # GPU-only import
    from vllm.lora.request import LoRARequest

    from effsql.data.bird import load_examples
    from effsql.exec.engine import _init_worker, _run_job
    from effsql.gen.backend import stable_seed
    from effsql.gen.parse import extract_sql
    from effsql.gen.prompts import build_question_prompt, load_templates
    from effsql.gen.sample import schemas_for
    from effsql.judge.candidates import add_candidate
    from effsql.judge.model import answer_token_ids, question_prompts, result_text
    from effsql.judge.vote import exec_jobs, method_index
    from effsql.train.data import train_messages
    from effsql.train.sft import base_model_cfg, make_counter

    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    cfg, gen_cfg = load_config("train.yaml"), load_config("generate.yaml")
    cc, vc, lc = jcfg["candidates"], jcfg["vote"], jcfg["latency"]
    n = n or lc["questions"]
    model_cfg = base_model_cfg(cfg, gen_cfg)
    gen_dir = paths.run_dir(cc["generator"], create=False)
    gmeta = read_json(gen_dir / "adapter_meta.json")
    judge_dir = paths.run_dir(judge_run, create=False)
    jmeta = read_json(judge_dir / "judge_meta.json")

    examples = load_examples(split, paths)
    rng = np.random.default_rng(0)
    examples = [examples[i] for i in rng.permutation(len(examples))[:n]]  # random order: databases mixed
    schemas = schemas_for(split, {e.db_id for e in examples}, paths)
    templates = load_templates()
    budget = cfg["prompt"]["max_prompt_tokens"]

    # Warm worker processes first (spawned, so no GPU state is copied), then the engine.
    workers = min(mp.cpu_count(), 8)
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(
        paths, {"timeout_s": vc["timeout_s"], "max_rows": vc["max_rows"], "count_steps": False}))
    llm = LLM(model=model_cfg["id"], revision=model_cfg.get("revision"), dtype=model_cfg.get("dtype", "auto"),
              max_model_len=model_cfg.get("max_model_len"), gpu_memory_utilization=lc["gpu_memory_utilization"],
              enable_prefix_caching=True, seed=0, enable_lora=True, max_loras=2,
              max_lora_rank=max(read_json(d / "adapter" / "adapter_config.json")["r"] for d in (gen_dir, judge_dir)))
    lora_gen = LoRARequest("generator", 1, str(gen_dir / "adapter"))
    lora_judge = LoRARequest("judge", 2, str(judge_dir / "adapter"))
    tok = llm.get_tokenizer()
    yes_no = answer_token_ids(tok)
    if list(yes_no) != [jmeta["yes_id"], jmeta["no_id"]]:
        raise RuntimeError(f"yes/no token ids {yes_no} differ from the judge's {jmeta['yes_id'], jmeta['no_id']}")
    count = make_counter(tok)
    greedy_max = gen_cfg["sampling"]["greedy"]["max_tokens"]
    with_results = bool(jmeta.get("with_results"))

    path = paths.run_dir(out) / f"latency_{split}.jsonl"
    done = done_ids(path)
    try:
        for e in examples:
            if e.qid in done:
                continue
            schema = schemas[e.db_id]
            t0 = time.perf_counter()
            msgs, _, _ = train_messages(templates, gmeta.get("arm", "best"), e, schema, budget, count, tag=gmeta.get("tag"))
            prompts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)]
            params = [SamplingParams(n=1, temperature=0.0, max_tokens=greedy_max)]
            for variant, k in cc["samples"].items():
                if k:
                    m, _, _ = build_question_prompt(templates, variant, e, schema, budget, count)
                    prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
                    params.append(SamplingParams(n=int(k), temperature=cc["temperature"], max_tokens=cc["max_tokens"],
                                                 seed=stable_seed(cc["seed"], "latency", e.qid, variant)))
            outs = llm.generate(prompts, params, use_tqdm=False, lora_request=lora_gen)
            cands: dict[str, dict[str, Any]] = {}
            sources = ["greedy"] + [v for v, k in cc["samples"].items() if k]
            for src, o in zip(sources, outs):
                for comp in o.outputs:
                    if src == "greedy" or comp.finish_reason != "length":
                        add_candidate(cands, extract_sql(comp.text)[0], src)
            cl = list(cands.values())
            t1 = time.perf_counter()

            ran_at: list[float] = []
            jobs = exec_jobs([{"qid": e.qid, "db_id": e.db_id, "cands": cl}], split, paths,
                             vc["preview_rows"] if with_results else 0)
            pending = pool.map_async(_run_job, jobs, callback=lambda _, t=ran_at: t.append(time.perf_counter())) \
                if jobs else None
            results = None
            if with_results:
                results = sorted(pending.get(), key=lambda r: r["key"]) if pending else []
            t_judge = time.perf_counter()

            p: list[float] = []
            n_missing = 0
            if cl:
                texts = [result_text(r, vc["timeout_s"]) for r in results] if with_results else None
                ids, _, _ = question_prompts(tok, templates, e, schema, [c["sql"] for c in cl], budget, texts)
                jout = llm.generate([{"prompt_token_ids": x} for x in ids],
                                    SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=20),
                                    use_tqdm=False, lora_request=lora_judge)
                for o in jout:
                    lp = {t: v.logprob for t, v in (o.outputs[0].logprobs or [{}])[0].items()}
                    pi, miss = judge_p(lp, yes_no)
                    p.append(pi)
                    n_missing += miss
            t2 = time.perf_counter()
            if results is None:
                results = sorted(pending.get(), key=lambda r: r["key"]) if pending else []
            t3 = time.perf_counter()
            choice = {m: method_index(m, cl, p, results) for m in vc["methods"]}
            t4 = time.perf_counter()
            append_jsonl(path, [{
                "qid": e.qid, "db_id": e.db_id, "n_candidates": len(cl), "n_judge_missing": n_missing,
                "seconds": {"generate": t1 - t0, "judge": t2 - t_judge, "run_wait": (t_judge - t1) + (t3 - t2),
                            "choose": t4 - t3,
                            "total": t4 - t0, "run_done_after_generate": (ran_at[0] - t1) if ran_at else 0.0},
                "max_completion_tokens": max((len(c.token_ids) for o in outs for c in o.outputs), default=0),
                "answers": {m: (cl[i]["sql"] if i is not None else "") for m, i in choice.items()}}])
    finally:
        pool.close()
        pool.join()
    rows = list(iter_jsonl(path))
    s = summarize(rows, lc["warmup"])
    s.update(split=split, judge_run=judge_run, generator=cc["generator"], timeout_s=vc["timeout_s"], workers=workers,
             judge_reads_results=with_results)
    write_json(paths.run_dir(out) / f"latency_{split}.json", s)
    (paths.run_dir(out) / f"latency_{split}.txt").write_text(format_latency(s) + "\n", encoding="utf-8")
    return s
