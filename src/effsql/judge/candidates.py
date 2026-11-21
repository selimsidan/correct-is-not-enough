"""Candidates for validation, dev and mini-dev (`python -m effsql.judge candidates`, GPU) and their labels
(`python -m effsql.judge label`, CPU; used to choose delta on validation and for the analysis, never at
answer time).

Candidates per question: the generator's greedy answer (read from runs/<generator>/predictions_<split>
.jsonl, already made) plus `samples` answers per prompt variant at temperature 1 (base: the prompt the
generator was trained with; efficient: the same with an instruction to write the fastest query). Distinct
by whitespace-normalised text, greedy first, each with how often it was written (n) and by which prompts.
Files in runs/<run>/: gen_raw_<split>.jsonl (resumable), candidates_<split>.jsonl.

Labels (labels_<split>.jsonl, resumable by question): every distinct candidate is executed like an RL
answer: correct = same result as gold (evaluate.yaml timeouts); status `correct` also requires no lucky
first-row pick and no foreign constant; steps for answers with the gold result (capped at step_cap_factor x
gold); rel_cost = (steps + N) / (gold + N) as in the evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from effsql.config import load_config
from effsql.data.bird import load_examples, select_questions
from effsql.gen.backend import GenParams, stable_seed
from effsql.gen.parse import extract_sql
from effsql.gen.prompts import build_question_prompt, load_templates
from effsql.gen.sample import Request, is_current_gen, run_requests, schemas_for
from effsql.io import (
    append_jsonl,
    done_ids,
    iter_jsonl,
    latest_by,
    read_json,
    write_json,
)
from effsql.judge.data import norm_sql
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
SPLITS = ("val", "dev", "minidev")
CORRECT = "correct"


def data_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return "train" if split == "val" else split


def split_examples(split: str, paths: Paths, export_run: str, limit: int | None = None) -> list[Any]:
    examples = load_examples(data_split(split), paths)
    if split == "val":
        keep = set(read_json(paths.run_dir(export_run, create=False) / "export_meta.json")["val_qids"])
        examples = [e for e in examples if e.qid in keep]
    return select_questions(examples, limit)


def add_candidate(cands: dict[str, dict[str, Any]], sql: str | None, source: str) -> None:
    if not sql:
        return
    c = cands.setdefault(norm_sql(sql), {"sql": sql.strip(), "n": 0, "sources": []})
    c["n"] += 1
    if source not in c["sources"]:
        c["sources"].append(source)


def generate_candidates(run: str, split: str, backend: Any, *, paths: Paths | None = None,
                        jcfg: dict[str, Any] | None = None, cfg: dict[str, Any] | None = None,
                        gen_cfg: dict[str, Any] | None = None, limit: int | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    cfg = cfg or load_config("train.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    cc = jcfg["candidates"]
    dsplit = data_split(split)
    examples = split_examples(split, paths, jcfg["data"]["export_run"], limit)
    schemas = schemas_for(dsplit, {e.db_id for e in examples}, paths)
    templates = load_templates()
    budget = cfg["prompt"]["max_prompt_tokens"]
    requests = []
    for e in sorted(examples, key=lambda x: (x.db_id, x.qid)):
        for variant, n in cc["samples"].items():
            if not n:
                continue
            msgs, level, _ = build_question_prompt(templates, variant, e, schemas[e.db_id], budget,
                                                       backend.count_tokens)
            params = GenParams(n=int(n), temperature=cc["temperature"], top_p=1.0, max_tokens=cc["max_tokens"],
                               seed=stable_seed(cc["seed"], "judge-cand", e.qid, variant))
            requests.append(Request(f"{e.qid}|{variant}", msgs, params,
                                    {"qid": e.qid, "split": dsplit, "variant": variant, "prompt_level": level}))
    run_dir = paths.run_dir(run)
    raw = run_dir / f"gen_raw_{split}.jsonl"
    stats = run_requests(requests, backend, raw, gen_cfg["chunk_size"], f"judge candidates {split}")
    greedy_path = paths.run_dir(cc["generator"], create=False) / f"predictions_{split}.jsonl"
    greedy = {r["qid"]: r.get("sql") or "" for r in iter_jsonl(greedy_path)}
    recs = latest_by(raw, key="key", where=is_current_gen)
    lines, n_cands = [], 0
    for e in examples:
        cands: dict[str, dict[str, Any]] = {}
        add_candidate(cands, greedy.get(e.qid), "greedy")
        for variant in cc["samples"]:
            r = recs.get(f"{e.qid}|{variant}")
            for comp in (r or {}).get("completions", []):
                if comp.get("finish_reason") != "length":
                    add_candidate(cands, extract_sql(comp["text"])[0], variant)
        n_cands += len(cands)
        lines.append(json.dumps({"qid": e.qid, "db_id": e.db_id, "cands": list(cands.values())}))
    (run_dir / f"candidates_{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats.update(split=split, n_questions=len(examples), n_candidates=n_cands,
                 candidates_per_question=n_cands / max(len(examples), 1), generator=cc["generator"],
                 n_greedy_missing=sum(1 for e in examples if not greedy.get(e.qid)))
    write_json(run_dir / f"candidates_stats_{split}.json", stats)
    log.info("candidates %s: %d questions, %.1f distinct candidates per question", split, len(examples),
             stats["candidates_per_question"])
    return stats


def read_candidates(paths: Paths, run: str, split: str) -> list[dict[str, Any]]:
    path = paths.run_dir(run, create=False) / f"candidates_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m effsql.judge candidates` first")
    return list(iter_jsonl(path))


def label_candidates(run: str, split: str, *, paths: Paths | None = None, jcfg: dict[str, Any] | None = None,
                     exec_cfg: dict[str, Any] | None = None, workers: int | None = None, chunk: int = 100
                     ) -> dict[str, Any]:
    from effsql.exec.engine import batched, execute_two_pass
    from effsql.measure.correctness import hashes_match
    from effsql.measure.cost import rel_cost
    from effsql.measure.gold import ensure_gold_costs
    from effsql.pool.quality import analyze, foreign_literals

    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    exec_cfg = exec_cfg or load_config("evaluate.yaml")["exec"]
    workers = workers or exec_cfg.get("workers")
    cc = jcfg["candidates"]
    dsplit = data_split(split)
    examples = {e.qid: e for e in load_examples(dsplit, paths)}
    cands = read_candidates(paths, run, split)
    out = paths.run_dir(run) / f"labels_{split}.jsonl"
    done = done_ids(out)
    todo = [r for r in cands if r["qid"] not in done]
    golds = ensure_gold_costs(dsplit, {r["qid"]: (examples[r["qid"]].db_id, examples[r["qid"]].gold_sql) for r in cands},
                              paths, exec_cfg, workers)
    ecfg = {**exec_cfg, "count_timeout_s": cc["count_timeout_s"]}
    log.info("labels %s: %d questions to label (%d done)", split, len(todo), len(done))
    for part in batched(todo, chunk):
        jobs, keys = [], []
        for r in part:
            g = golds[r["qid"]]
            cap = int(cc["step_cap_factor"] * max(g["vm_steps"] or 0, 1000))
            for j, c in enumerate(r["cands"]):
                keys.append((r["qid"], j))
                jobs.append({"key": len(jobs), "db_path": paths.local_db_path(dsplit, r["db_id"]), "sql": c["sql"],
                             "count_exec": {"step_budget": cap}})

        def matches(x: dict[str, Any], keys: list[tuple[str, int]] = keys) -> bool:
            return bool(x["ok"] and hashes_match(x["result_hash"], golds[keys[x["key"]][0]]["result_hash"]))

        res = execute_two_pass(jobs, paths, ecfg, count_if=matches, workers=workers) if jobs else {}
        recs, k = [], 0
        for r in part:
            e, g = examples[r["qid"]], golds[r["qid"]]
            gflags = set(analyze(e.gold_sql)["flags"])
            labs = []
            for c in r["cands"]:
                x = res[k]
                k += 1
                match = matches(x)
                if not match:
                    status = x["error_type"] or "wrong_result"
                elif set(analyze(c["sql"])["flags"]) - gflags:
                    status = "first_row_pick"
                elif foreign_literals(c["sql"], e.gold_sql, e.question, e.evidence):
                    status = "foreign_constant"
                else:
                    status = CORRECT
                steps = x.get("vm_steps") if match else None
                rc = rel_cost(steps, g["vm_steps"], exec_cfg["progress_n"]) if steps is not None and g["vm_steps"] is not None else None
                labs.append({"correct": match, "status": status, "steps": steps, "rel_cost": rc})
            recs.append({"qid": r["qid"], "gold_steps": g["vm_steps"], "gold_ok": g["ok"], "labels": labs})
        append_jsonl(out, recs)
        log.info("labels %s: %d more questions", split, len(recs))
    n = sum(1 for _ in iter_jsonl(out))
    return {"split": split, "n_questions_labelled": n, "n_questions": len(cands)}


def read_labels(paths: Paths, run: str, split: str) -> dict[str, dict[str, Any]] | None:
    path = paths.run_dir(run, create=False) / f"labels_{split}.jsonl"
    return latest_by(path) if path.exists() else None
