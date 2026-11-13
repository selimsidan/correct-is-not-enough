"""Greedy predictions of a fine-tuned model (base + LoRA adapter) for validation, dev or minidev.

Prompts come from train.data.train_messages, the function training used, with the same token budget:
the model sees at test time exactly the prompt format it was trained on (the tagged arm gets its
test-time tag, by default the export's `inference_tag`). Output files match `effsql predict`
({qid, sql, extraction, ...} per line), so `effsql evaluate` scores them unchanged:
  runs/<run>/predictions_<split>.jsonl, predict_raw_<split>.jsonl, predict_stats_<split>.json
`val` means the export's held-out train databases (scored with `evaluate --split train`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from effsql.config import load_config
from effsql.data.bird import load_examples, select_questions
from effsql.gen.backend import Completion, GenParams
from effsql.gen.parse import extract_sql
from effsql.gen.prompts import load_templates
from effsql.gen.sample import Request, is_current_gen, run_requests, schemas_for
from effsql.io import latest_by, read_json, write_json
from effsql.paths import Paths, load_paths
from effsql.train.data import arm_prompt_variant, format_tag, train_messages

log = logging.getLogger(__name__)


class LoRAVLLMBackend:
    """vLLM engine for the base model with one LoRA adapter applied to every request."""

    def __init__(self, model_cfg: dict[str, Any], adapter_dir: Path, seed: int = 0):
        from vllm import LLM  # GPU-only import
        from vllm.lora.request import LoRARequest

        acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        if acfg.get("base_model_name_or_path") not in (None, model_cfg["id"]):
            raise ValueError(f"adapter trained on {acfg['base_model_name_or_path']!r}, engine uses {model_cfg['id']!r}")
        self.model_id = model_cfg["id"]
        self.model_revision = model_cfg.get("revision")
        self.adapter = str(adapter_dir)
        self.llm = LLM(
            model=self.model_id, revision=self.model_revision, dtype=model_cfg.get("dtype", "auto"),
            max_model_len=model_cfg.get("max_model_len"),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.9),
            enable_prefix_caching=model_cfg.get("enable_prefix_caching", True), seed=seed,
            enable_lora=True, max_loras=1, max_lora_rank=int(acfg["r"]),
        )
        self.lora = LoRARequest("sft", 1, self.adapter)
        self.tok = self.llm.get_tokenizer()

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return len(self.tok.encode(self._prompt(messages), add_special_tokens=False))

    def generate(self, batch: list[list[dict[str, str]]], params: list[GenParams],
                 adapters: list[Any] | None = None) -> list[list[Completion]]:
        from vllm import SamplingParams

        if adapters and any(a is not None for a in adapters):
            raise ValueError("this backend applies its own adapter to every request")
        prompts = [self._prompt(m) for m in batch]
        sps = [SamplingParams(n=p.n, temperature=p.temperature, top_p=p.top_p, max_tokens=p.max_tokens,
                              seed=p.seed) for p in params]
        outs = self.llm.generate(prompts, sps, use_tqdm=False, lora_request=self.lora)
        return [[Completion(o.text, o.finish_reason, len(o.token_ids)) for o in out.outputs] for out in outs]


def val_qids(export_dir: Path) -> list[str]:
    return list(read_json(export_dir / "export_meta.json")["val_qids"])


def predict_sft(arm: str, split: str, run: str, export_run: str, backend: Any, *, paths: Paths | None = None,
                cfg: dict[str, Any] | None = None, gen_cfg: dict[str, Any] | None = None,
                tag: str | None = None, limit: int | None = None) -> dict[str, Any]:
    if split not in ("val", "dev", "minidev"):
        raise ValueError("split must be val (held-out train databases), dev or minidev")
    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    export_dir = paths.run_dir(export_run, create=False)
    emeta = read_json(export_dir / "export_meta.json")
    if arm_prompt_variant(arm) == "tagged" and tag is None:
        tag = format_tag(emeta["inference_tag"], emeta["export_cfg"]["tagged"]["decimals"])
    if arm_prompt_variant(arm) != "tagged":
        tag = None
    data_split = "train" if split == "val" else split
    examples = load_examples(data_split, paths)
    if split == "val":
        keep = set(val_qids(export_dir))
        examples = [e for e in examples if e.qid in keep]
    examples = select_questions(examples, limit)
    schemas = schemas_for(data_split, {e.db_id for e in examples}, paths)
    templates = load_templates()
    g = gen_cfg["sampling"]["greedy"]
    params = GenParams(n=1, temperature=0.0, top_p=1.0, max_tokens=g["max_tokens"])
    budget = cfg["prompt"]["max_prompt_tokens"]
    requests = []
    for e in sorted(examples, key=lambda x: x.db_id):
        msgs, level, n_tok = train_messages(templates, arm, e, schemas[e.db_id], budget, backend.count_tokens, tag=tag)
        requests.append(Request(e.qid, msgs, params, {"qid": e.qid, "split": data_split, "db_id": e.db_id,
                                                      "variant": arm_prompt_variant(arm), "arm": arm, "tag": tag,
                                                      "source": "predict_sft", "prompt_level": level,
                                                      "prompt_tokens": n_tok}))
    run_dir = paths.run_dir(run)
    raw = run_dir / f"predict_raw_{split}.jsonl"
    stats = run_requests(requests, backend, raw, gen_cfg["chunk_size"], f"predict {arm} {split}")
    recs = latest_by(raw, key="key", where=is_current_gen)
    lines = []
    for e in examples:
        r = recs.get(e.qid)
        comp = r["completions"][0] if r and r["completions"] else {"text": "", "finish_reason": None}
        sql, how = extract_sql(comp["text"])
        lines.append(json.dumps({"qid": e.qid, "sql": sql or "", "extraction": how,
                                 "finish_reason": comp.get("finish_reason"),
                                 "prompt_level": r.get("prompt_level") if r else None, "arm": arm, "tag": tag,
                                 "model": backend.model_id, "model_revision": backend.model_revision,
                                 "adapter": getattr(backend, "adapter", None)}))
    out = run_dir / f"predictions_{split}.jsonl"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats.update(n_questions=len(examples), predictions=str(out), arm=arm, tag=tag)
    write_json(run_dir / f"predict_stats_{split}.json", stats)
    log.info("predictions written to %s", out)
    return stats
