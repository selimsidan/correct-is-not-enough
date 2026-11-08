"""`effsql generate`: candidate SQL for train questions (and dev/minidev for analysis only).

For every question and prompt variant: one greedy request (n=1, temperature 0) and one sampled
request (n samples). Requests are grouped by database for prefix caching and processed in chunks;
each chunk is appended to runs/<run>/generations_<split>.jsonl, so a disconnect loses at most one
chunk. Resume key: `<qid>|<variant>|<source>`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effsql.config import load_config
from effsql.data.bird import Example, load_examples, select_questions
from effsql.data.schema import load_schema
from effsql.data.stage import select_db_ids
from effsql.exec.engine import batched
from effsql.gen.backend import Backend, GenParams, stable_seed
from effsql.gen.prompts import build_question_prompt, load_templates
from effsql.io import append_jsonl, done_ids, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
GEN_VERSION = 1


@dataclass
class Request:
    key: str
    messages: list[dict[str, str]]
    params: GenParams
    meta: dict[str, Any] = field(default_factory=dict)


def is_current_gen(rec: dict[str, Any]) -> bool:
    return rec.get("gen_version") == GEN_VERSION


def run_requests(
    requests: list[Request], backend: Backend, out_path: Path, chunk_size: int, label: str
) -> dict[str, Any]:
    """Run requests not yet in out_path, append results chunk by chunk, return throughput stats."""
    done = done_ids(out_path, key="key", where=is_current_gen)
    todo = [r for r in requests if r.key not in done]
    log.info("%s: %d requests to run (%d already done)", label, len(todo), len(requests) - len(todo))
    t0 = time.time()
    n_tokens = n_completions = 0
    for k, chunk in enumerate(batched(todo, chunk_size), 1):
        outs = backend.generate([r.messages for r in chunk], [r.params for r in chunk])
        recs = []
        for r, comps in zip(chunk, outs):
            n_tokens += sum(c.n_tokens for c in comps)
            n_completions += len(comps)
            recs.append({
                "key": r.key, **r.meta, "params": r.params.record(),
                "model": backend.model_id, "model_revision": backend.model_revision,
                "completions": [c.record() for c in comps], "gen_version": GEN_VERSION,
            })
        append_jsonl(out_path, recs)
        el = time.time() - t0
        log.info("%s: chunk %d, %d/%d requests, %.0f s, %.0f generated tokens/s",
                 label, k, min(k * chunk_size, len(todo)), len(todo), el, n_tokens / max(el, 1e-9))
    elapsed = time.time() - t0
    return {"label": label, "n_requests_run": len(todo), "n_requests_total": len(requests),
            "n_completions": n_completions, "generated_tokens": n_tokens, "elapsed_s": elapsed,
            "generated_tokens_per_s": n_tokens / elapsed if elapsed > 0 else None}


def schemas_for(split: str, db_ids: set[str], paths: Paths) -> dict[str, dict[str, Any]]:
    missing = [d for d in sorted(db_ids) if not paths.schema_cache_path(split, d).exists()]
    if missing:
        raise RuntimeError(f"No schema cache for {missing}. Run `effsql schema-cache --split {split}` on CPU first.")
    return {d: load_schema(split, d, paths) for d in db_ids}


def pick_examples(split: str, paths: Paths, limit: int | None, db_ids: list[str] | None,
                  shard: str | None) -> list[Example]:
    examples = load_examples(split, paths)
    if db_ids or shard:
        keep = set(select_db_ids([e.db_id for e in examples], db_ids, shard))
        examples = [e for e in examples if e.db_id in keep]
    return select_questions(examples, limit)


def generate(
    split: str, run: str, backend: Backend, *, cfg: dict[str, Any] | None = None, paths: Paths | None = None,
    limit: int | None = None, db_ids: list[str] | None = None, shard: str | None = None, analysis: bool = False,
) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = cfg or load_config("generate.yaml")
    if split != "train" and not analysis:
        raise ValueError(f"generate on {split} is evaluation data; pass --analysis to generate it for analysis "
                         "only (records are marked and never used for training). Use `effsql predict` for scoring.")
    examples = pick_examples(split, paths, limit, db_ids, shard)
    schemas = schemas_for(split, {e.db_id for e in examples}, paths)
    templates = load_templates()
    budget = cfg["prompt"]["max_prompt_tokens"]
    max_len = cfg["model"]["max_model_len"]
    s_cfg = cfg["sampling"]

    requests: list[Request] = []
    n_too_long = 0
    for e in sorted(examples, key=lambda x: x.db_id):  # group by database for prefix caching
        for variant in cfg["prompt"]["variants"]:
            msgs, level, n_tok = build_question_prompt(templates, variant, e, schemas[e.db_id], budget,
                                                       backend.count_tokens)
            meta = {"qid": e.qid, "split": split, "db_id": e.db_id, "variant": variant, "prompt_level": level,
                    "prompt_tokens": n_tok, "analysis_only": split != "train"}
            for source in ("greedy", "sample"):
                sp = s_cfg[source]
                if n_tok + sp["max_tokens"] > max_len:
                    n_too_long += 1
                    continue
                params = GenParams(n=sp["n"], temperature=sp["temperature"], top_p=sp["top_p"],
                                   max_tokens=sp["max_tokens"],
                                   seed=None if source == "greedy" else stable_seed(cfg["seed"], e.qid, variant))
                requests.append(Request(f"{e.qid}|{variant}|{source}", msgs, params, {**meta, "source": source}))
    if n_too_long:
        log.warning("%d requests skipped: prompt longer than max_model_len even after shrinking", n_too_long)

    out = paths.run_dir(run) / f"generations_{split}.jsonl"
    stats = run_requests(requests, backend, out, cfg["chunk_size"], f"generate {split}")
    stats.update(n_questions=len(examples), n_skipped_too_long=n_too_long, model=backend.model_id,
                 model_revision=backend.model_revision, vllm=getattr(backend, "vllm_version", None))
    total = len(load_examples(split, paths))
    if stats["n_requests_run"] and stats["elapsed_s"]:
        frac = stats["n_requests_run"] / max(len(requests), 1)
        stats["projected_hours_full_split"] = stats["elapsed_s"] / frac / len(examples) * total / 3600
    write_json(paths.run_dir(run) / f"gen_stats_{split}.json", stats)
    return stats
