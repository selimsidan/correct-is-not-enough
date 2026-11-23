"""`python -m effsql.judge latency` (A100, notebook 84): the whole answer path, timed question by question.

Every question is answered by effsql.answer.System, the same path the demo uses: one vLLM engine with the
base model once and both adapters (generator and judge); generate, run every distinct candidate with the
answer-time cap, judge, choose by every method of vote.methods. A judge trained with results
(judge_meta with_results) reads each candidate's result, so judging waits for the queries and run_wait is
the time between the end of generation and the start of judging.
Writes runs/<out>/latency_<split>.jsonl (per question: seconds per stage, total, candidates, the answer
of each method) and latency_<split>.json (median, p90 and max per stage, warm-up questions left out).
The answers are for checking only; accuracy comes from the offline runs (notebooks 81 to 83).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from effsql.answer import judge_p  # noqa: F401 (kept importable from here)
from effsql.config import load_config
from effsql.io import append_jsonl, done_ids, iter_jsonl, read_json, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
STAGES = ("generate", "judge", "run_wait", "choose", "total")


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
    from effsql.answer import System, VLLMEngine
    from effsql.data.bird import load_examples
    from effsql.gen.sample import schemas_for
    from effsql.train.sft import base_model_cfg

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
    with_results = bool(jmeta.get("with_results"))
    system = System(lambda: VLLMEngine(model_cfg, gen_dir / "adapter", judge_dir / "adapter",
                                       gpu_memory_utilization=lc["gpu_memory_utilization"]),
                    paths=paths, jcfg=jcfg, cfg=cfg, gen_cfg=gen_cfg, arm=gmeta.get("arm", "best"),
                    tag=gmeta.get("tag"), reads_results=with_results, yes_no=(jmeta["yes_id"], jmeta["no_id"]))
    workers = system.workers

    path = paths.run_dir(out) / f"latency_{split}.jsonl"
    done = done_ids(path)
    try:
        for e in examples:
            if e.qid in done:
                continue
            a = system.answer(e, schemas[e.db_id], split, method=vc["methods"][0], methods=list(vc["methods"]))
            append_jsonl(path, [{
                "qid": e.qid, "db_id": e.db_id, "n_candidates": len(a.candidates),
                "n_judge_missing": a.n_judge_missing, "seconds": a.seconds,
                "max_completion_tokens": a.max_completion_tokens,
                "answers": {m: (a.candidates[i]["sql"] if i is not None else "") for m, i in a.answers.items()}}])
    finally:
        system.close()
    rows = list(iter_jsonl(path))
    s = summarize(rows, lc["warmup"])
    s.update(split=split, judge_run=judge_run, generator=cc["generator"], timeout_s=vc["timeout_s"], workers=workers,
             judge_reads_results=with_results)
    write_json(paths.run_dir(out) / f"latency_{split}.json", s)
    (paths.run_dir(out) / f"latency_{split}.txt").write_text(format_latency(s) + "\n", encoding="utf-8")
    return s
