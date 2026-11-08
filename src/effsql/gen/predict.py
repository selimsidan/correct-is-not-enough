"""`effsql predict`: one greedy prediction per dev or minidev question, for `effsql evaluate`.

Writes runs/<run>/predictions_<split>_<prompt>.jsonl with {qid, sql, extraction, finish_reason,
prompt_level}. These are evaluation outputs only and never enter training data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from effsql.config import load_config
from effsql.data.bird import load_examples, select_questions
from effsql.gen.backend import Backend, GenParams
from effsql.gen.parse import extract_sql
from effsql.gen.prompts import build_question_prompt, load_templates
from effsql.gen.sample import Request, is_current_gen, run_requests, schemas_for
from effsql.io import latest_by, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)


def predict(
    split: str, prompt: str, run: str, backend: Backend, *, paths: Paths | None = None,
    cfg: dict[str, Any] | None = None, limit: int | None = None, adapter: str | None = None,
) -> dict[str, Any]:
    if adapter:
        raise NotImplementedError("LoRA adapters arrive with the training phase")
    if split not in ("dev", "minidev"):
        raise ValueError("predict is for dev and minidev")
    paths = paths or load_paths()
    cfg = cfg or load_config("generate.yaml")
    examples = select_questions(load_examples(split, paths), limit)
    schemas = schemas_for(split, {e.db_id for e in examples}, paths)
    templates = load_templates()
    g = cfg["sampling"]["greedy"]
    params = GenParams(n=1, temperature=0.0, top_p=1.0, max_tokens=g["max_tokens"])
    requests = []
    for e in sorted(examples, key=lambda x: x.db_id):
        msgs, level, n_tok = build_question_prompt(templates, prompt, e, schemas[e.db_id],
                                                   cfg["prompt"]["max_prompt_tokens"], backend.count_tokens)
        requests.append(Request(e.qid, msgs, params, {"qid": e.qid, "split": split, "db_id": e.db_id,
                                                      "variant": prompt, "source": "predict",
                                                      "prompt_level": level, "prompt_tokens": n_tok}))
    run_dir = paths.run_dir(run)
    raw = run_dir / f"predict_raw_{split}_{prompt}.jsonl"
    stats = run_requests(requests, backend, raw, cfg["chunk_size"], f"predict {split} {prompt}")

    recs = latest_by(raw, key="key", where=is_current_gen)
    out = run_dir / f"predictions_{split}_{prompt}.jsonl"
    lines = []
    for e in examples:
        r = recs.get(e.qid)
        comp = r["completions"][0] if r and r["completions"] else {"text": "", "finish_reason": None}
        sql, how = extract_sql(comp["text"])
        lines.append(json.dumps({"qid": e.qid, "sql": sql or "", "extraction": how,
                                 "finish_reason": comp.get("finish_reason"),
                                 "prompt_level": r.get("prompt_level") if r else None,
                                 "model": backend.model_id, "model_revision": backend.model_revision}))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats.update(n_questions=len(examples), predictions=str(out))
    write_json(run_dir / f"predict_stats_{split}_{prompt}.json", stats)
    log.info("predictions written to %s", out)
    return stats
