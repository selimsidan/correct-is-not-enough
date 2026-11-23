"""One question through the whole system: generate, run, judge, choose. The demo and the latency measurement
(`python -m effsql.judge latency`) both answer through `System.answer`, so the timed path is the demo's path.

Per question, as a user would ask it:
  1. generate  the generator's greedy answer and its samples per prompt variant (judge.yaml candidates), one
               batched call; distinct candidates as in `python -m effsql.judge candidates`;
  2. run       every distinct candidate compiled and run on the database in warm worker processes with the
               answer-time cap (judge.yaml vote.timeout_s);
  3. judge     one prompt per candidate, one generated token with the judge adapter; p = sigmoid of the
               log-probability of "yes" minus that of "no". A judge that reads results waits for step 2
               and sees each result exactly as in training (the first vote.preview_rows rows);
  4. choose    a method of judge/vote.py (the final system: weighted_vote).

The engine holds the base model once and both LoRA adapters (VLLMEngine). `quantize` loads the base model
in 4 bit (bitsandbytes) for GPUs too small for 16 bit, such as a free Colab T4; answers can then differ
slightly from the reported system.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from effsql.config import load_config
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)


def judge_p(logprobs: dict[int, float], yes_no: tuple[int, int]) -> tuple[float, bool]:
    """p from the first generated token's top log-probabilities; a token missing from the top list gets the
    lowest listed value (an upper bound). Returns (p, whether one was missing)."""
    floor = min(logprobs.values()) if logprobs else 0.0
    y, n = (logprobs.get(t) for t in yes_no)
    x = (y if y is not None else floor) - (n if n is not None else floor)
    p = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
    return p, y is None or n is None


class Engine(Protocol):
    tok: Any

    def generate(self, prompts: list[str], params: list[dict[str, Any]]) -> list[list[tuple[str, str, int]]]:
        """Per prompt, its completions as (text, finish_reason, n_tokens), with the generator adapter."""

    def judge(self, ids: list[list[int]]) -> list[dict[int, float]]:
        """Per prompt (token ids), the top log-probabilities of the first generated token, judge adapter."""


class VLLMEngine:
    """vLLM with the base model once and two LoRA adapters: generator and judge."""

    def __init__(self, model_cfg: dict[str, Any], generator: Path, judge: Path, *, gpu_memory_utilization: float,
                 quantize: bool = False, max_model_len: int | None = None, seed: int = 0):
        import json

        from vllm import LLM  # GPU-only import
        from vllm.lora.request import LoRARequest

        rank = max(json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))["r"] for d in (generator, judge))
        kw: dict[str, Any] = {"quantization": "bitsandbytes", "dtype": "float16"} if quantize else \
            {"dtype": model_cfg.get("dtype", "auto")}
        self.llm = LLM(model=model_cfg["id"], revision=model_cfg.get("revision"),
                       max_model_len=max_model_len or model_cfg.get("max_model_len"),
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=seed,
                       enable_lora=True, max_loras=2, max_lora_rank=rank, **kw)
        self.lora_gen = LoRARequest("generator", 1, str(generator))
        self.lora_judge = LoRARequest("judge", 2, str(judge))
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts: list[str], params: list[dict[str, Any]]) -> list[list[tuple[str, str, int]]]:
        from vllm import SamplingParams

        sp = [SamplingParams(**p) for p in params]
        outs = self.llm.generate(prompts, sp, use_tqdm=False, lora_request=self.lora_gen)
        return [[(c.text, c.finish_reason, len(c.token_ids)) for c in o.outputs] for o in outs]

    def judge(self, ids: list[list[int]]) -> list[dict[int, float]]:
        from vllm import SamplingParams

        if not ids:
            return []
        outs = self.llm.generate([{"prompt_token_ids": x} for x in ids],
                                 SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=20),
                                 use_tqdm=False, lora_request=self.lora_judge)
        return [{t: v.logprob for t, v in (o.outputs[0].logprobs or [{}])[0].items()} for o in outs]


@dataclass
class Answer:
    qid: str
    db_id: str
    candidates: list[dict[str, Any]]      # {sql, n, sources}, greedy first
    results: list[dict[str, Any]]         # execution record per candidate (judge/vote.py EXEC_FIELDS)
    p: list[float]                        # the judge's probability that each candidate is correct
    choice: int | None                    # index of the answer by `method`
    method: str
    answers: dict[str, int | None]        # the index chosen by every method asked for
    seconds: dict[str, float]
    n_judge_missing: int = 0
    max_completion_tokens: int = 0
    result_texts: list[str] = field(default_factory=list)   # what the judge read (empty when it reads SQL only)

    @property
    def sql(self) -> str:
        return self.candidates[self.choice]["sql"] if self.choice is not None else ""


class System:
    """The final system for one question at a time. Worker processes for running queries start before the
    engine (spawned, so no GPU state is copied); `make_engine` builds the engine afterwards."""

    def __init__(self, make_engine: Callable[[], Engine], *, paths: Paths | None = None,
                 jcfg: dict[str, Any] | None = None, cfg: dict[str, Any] | None = None,
                 gen_cfg: dict[str, Any] | None = None, arm: str = "best", tag: str | None = None,
                 reads_results: bool = True, workers: int | None = None, yes_no: tuple[int, int] | None = None):
        from effsql.exec.engine import _init_worker
        from effsql.gen.prompts import load_templates
        from effsql.judge.model import answer_token_ids
        from effsql.train.sft import make_counter

        self.paths = paths or load_paths()
        self.jcfg = jcfg or load_config("judge.yaml")
        self.cfg = cfg or load_config("train.yaml")
        self.gen_cfg = gen_cfg or load_config("generate.yaml")
        self.cc, self.vc = self.jcfg["candidates"], self.jcfg["vote"]
        self.arm, self.tag, self.reads_results = arm, tag, reads_results
        self.templates = load_templates()
        self.budget = self.cfg["prompt"]["max_prompt_tokens"]
        self.exec_kwargs = {"timeout_s": self.vc["timeout_s"], "max_rows": self.vc["max_rows"], "count_steps": False}
        self.workers = workers if workers is not None else min(mp.cpu_count(), 8)
        self.pool = None
        if self.workers > 1:
            self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker,
                                                     initargs=(self.paths, self.exec_kwargs))
        try:
            self.engine = make_engine()
        except BaseException:
            self.close()
            raise
        self.tok = self.engine.tok
        self.count = make_counter(self.tok)
        self.yes_no = answer_token_ids(self.tok)
        if yes_no is not None and list(self.yes_no) != list(yes_no):
            raise RuntimeError(f"yes/no token ids {self.yes_no} differ from the judge's {yes_no}")

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _run(self, jobs: list[dict[str, Any]]) -> Callable[[], list[dict[str, Any]]]:
        """Start the jobs; returns a function that waits for their records, in job order."""
        from effsql.exec.engine import _init_worker, _run_job

        if not jobs:
            return list
        if self.pool is None:
            _init_worker(self.paths, self.exec_kwargs)
            done = [_run_job(j) for j in jobs]
            return lambda: sorted(done, key=lambda r: r["key"])
        pending = self.pool.map_async(_run_job, jobs)
        return lambda: sorted(pending.get(), key=lambda r: r["key"])

    def answer(self, e: Any, schema: dict[str, Any], db_split: str, *, method: str = "weighted_vote",
               methods: list[str] | None = None, display_rows: int = 0) -> Answer:
        """e: an Example (db_id, question, evidence; qid any string). db_split: the split whose staged
        databases hold e.db_id. display_rows: rows kept in each result's preview for showing (the judge
        still reads vote.preview_rows)."""
        from effsql.gen.backend import stable_seed
        from effsql.gen.parse import extract_sql
        from effsql.gen.prompts import build_question_prompt
        from effsql.judge.candidates import add_candidate
        from effsql.judge.model import question_prompts, result_text
        from effsql.judge.vote import exec_jobs, method_index
        from effsql.train.data import train_messages

        cc, vc, tok = self.cc, self.vc, self.tok
        t0 = time.perf_counter()
        msgs, _, _ = train_messages(self.templates, self.arm, e, schema, self.budget, self.count, tag=self.tag)
        prompts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)]
        params = [{"n": 1, "temperature": 0.0, "max_tokens": self.gen_cfg["sampling"]["greedy"]["max_tokens"]}]
        sources = ["greedy"]
        for variant, k in cc["samples"].items():
            if k:
                m, _, _ = build_question_prompt(self.templates, variant, e, schema, self.budget, self.count)
                prompts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
                params.append({"n": int(k), "temperature": cc["temperature"], "max_tokens": cc["max_tokens"],
                               "seed": stable_seed(cc["seed"], "latency", e.qid, variant)})
                sources.append(variant)
        outs = self.engine.generate(prompts, params)
        cands: dict[str, dict[str, Any]] = {}
        for src, comps in zip(sources, outs):
            for text, finish, _ in comps:
                if src == "greedy" or finish != "length":
                    add_candidate(cands, extract_sql(text)[0], src)
        cl = list(cands.values())
        t1 = time.perf_counter()

        keep = vc["preview_rows"] if self.reads_results else 0
        wait = self._run(exec_jobs([{"qid": e.qid, "db_id": e.db_id, "cands": cl}], db_split, self.paths,
                                   max(keep, display_rows)))
        results = wait() if self.reads_results else None
        t_judge = time.perf_counter()

        p: list[float] = []
        n_missing = 0
        texts: list[str] = []
        if cl:
            if self.reads_results:
                texts = [result_text({**r, "preview": (r.get("preview") or [])[:keep]}, vc["timeout_s"])
                         for r in results]
            ids, _, _ = question_prompts(tok, self.templates, e, schema, [c["sql"] for c in cl], self.budget,
                                         texts if self.reads_results else None)
            for lp in self.engine.judge(ids):
                pi, miss = judge_p(lp, self.yes_no)
                p.append(pi)
                n_missing += miss
        t2 = time.perf_counter()
        if results is None:
            results = wait()
        t3 = time.perf_counter()
        answers = {m: method_index(m, cl, p, results) for m in dict.fromkeys([method, *(methods or [])])}
        t4 = time.perf_counter()
        return Answer(
            qid=e.qid, db_id=e.db_id, candidates=cl, results=results, p=p, choice=answers[method], method=method,
            answers=answers, n_judge_missing=n_missing, result_texts=texts,
            max_completion_tokens=max((n for comps in outs for _, _, n in comps), default=0),
            seconds={"generate": t1 - t0, "judge": t2 - t_judge, "run_wait": (t_judge - t1) + (t3 - t2),
                     "choose": t4 - t3, "total": t4 - t0})
