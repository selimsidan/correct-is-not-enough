"""`effsql rewrite`: ask the model for equivalent rewrites of a correct seed query (train only).

For each flagged question (not diverse in the latest diversity round, gold ok), one greedy request
per (seed, style): the seed is the gold SQL by default (optionally also the cheapest correct
candidate). The prompt includes the seed's EXPLAIN QUERY PLAN. `optimized` is always requested;
the remaining styles rotate per question so every style is used across the dataset.
Rewrites are written to rewrites_<split>.jsonl with source `rewrite:<style>`, then go through the
same parse, screen and diversity steps as any candidate: a rewrite whose result differs from gold
is simply incorrect and never trusted.
"""

from __future__ import annotations

import logging
from typing import Any

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.exec.engine import connect_readonly
from effsql.gen.backend import Backend, GenParams, stable_seed
from effsql.gen.prompts import build_rewrite_prompt, load_templates
from effsql.gen.sample import Request, run_requests, schemas_for
from effsql.io import latest_by, read_jsonl
from effsql.paths import Paths, load_paths
from effsql.pool.diversity import candidate_cost, latest_round
from effsql.pool.screen import is_current_screen, load_candidates

log = logging.getLogger(__name__)


def styles_for(qid: str, cfg: dict[str, Any]) -> list[str]:
    """`always` styles plus a deterministic rotation of the others, k in total."""
    names = list(cfg["styles"])
    always = [s for s in cfg.get("always", []) if s in names]
    rest = [s for s in names if s not in always]
    k = max(cfg["k"] - len(always), 0)
    if not rest or not k:
        return always[: cfg["k"]]
    start = stable_seed("styles", qid) % len(rest)
    rotated = rest[start:] + rest[:start]
    return always + rotated[:k]


def plan_text(conn: Any, sql: str) -> str:
    try:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    except Exception as e:  # noqa: BLE001
        return f"(plan unavailable: {e})"
    depth: dict[int, int] = {0: -1}
    lines = []
    for node_id, parent, _unused, detail in rows:
        depth[node_id] = depth.get(parent, -1) + 1
        lines.append("  " * depth[node_id] + detail)
    return "\n".join(lines)


def rewrite(
    run: str, backend: Backend, *, split: str = "train", round_: int = 1, paths: Paths | None = None,
    cfg: dict[str, Any] | None = None, gen_cfg: dict[str, Any] | None = None, limit: int | None = None,
) -> dict[str, Any]:
    if split != "train":
        raise ValueError("rewrite builds training data and runs on train only")
    paths = paths or load_paths()
    cfg = cfg or load_config("rewrite.yaml")
    gen_cfg = gen_cfg or load_config("generate.yaml")
    run_dir = paths.run_dir(run)
    div_round = latest_round(paths, run, split)
    div = {r["qid"]: r for r in read_jsonl(run_dir / f"diversity_{split}_r{div_round}.jsonl")}
    if cfg.get("target", "flagged") == "all":
        targets = [q for q, r in div.items() if r["gold_status"] == "ok"]
    else:
        targets = [q for q, r in div.items() if r["gold_status"] == "ok" and not r["diverse"]]
    targets = sorted(targets)[:limit] if limit else sorted(targets)
    examples = {e.qid: e for e in load_examples(split, paths)}
    schemas = schemas_for(split, {examples[q].db_id for q in targets}, paths)

    cheapest: dict[str, str] = {}
    if "cheapest_correct" in cfg.get("seeds", []):
        cands = load_candidates(paths, run, split)
        screen = latest_by(run_dir / f"screen_{split}.jsonl", key="cand_id", where=is_current_screen)
        for cid, s in screen.items():
            cost = candidate_cost(s)
            if s["correct"] and cost is not None and cid in cands and not cands[cid]["is_gold"]:
                q = s["qid"]
                if q not in cheapest or cost < cheapest[q][0]:
                    cheapest[q] = (cost, cands[cid]["sql_original"])
        cheapest = {q: v[1] for q, v in cheapest.items()}

    templates = load_templates()
    budget = gen_cfg["prompt"]["max_prompt_tokens"]
    sp = cfg["sampling"]
    requests: list[Request] = []
    conns: dict[str, Any] = {}
    try:
        for q in targets:
            e = examples[q]
            seeds = {"gold": e.gold_sql}
            if q in cheapest:
                seeds["cheapest_correct"] = cheapest[q]
            if e.db_id not in conns:
                conns[e.db_id] = connect_readonly(paths.local_db_path(split, e.db_id), paths)
            for seed_kind, seed_sql in seeds.items():
                plan = plan_text(conns[e.db_id], seed_sql)
                for style in styles_for(q, cfg):
                    msgs, level, n_tok = build_rewrite_prompt(
                        templates, e, schemas[e.db_id], seed_sql, plan, style, cfg["styles"][style],
                        budget, backend.count_tokens)
                    params = GenParams(n=sp["n"], temperature=sp["temperature"], top_p=sp["top_p"],
                                       max_tokens=sp["max_tokens"],
                                       seed=stable_seed(gen_cfg.get("seed", 0), q, style, seed_kind, round_))
                    requests.append(Request(
                        f"{q}|rewrite:{style}|{seed_kind}|r{round_}", msgs, params,
                        {"qid": q, "split": split, "db_id": e.db_id, "variant": "rewrite", "source": f"rewrite:{style}",
                         "style": style, "seed_kind": seed_kind, "round": round_, "prompt_level": level,
                         "prompt_tokens": n_tok, "analysis_only": False}))
    finally:
        for c in conns.values():
            c.close()
    stats = run_requests(requests, backend, run_dir / f"rewrites_{split}.jsonl", gen_cfg["chunk_size"],
                         f"rewrite round {round_}")
    stats.update(n_flagged_questions=len(targets), diversity_round_used=div_round)
    return stats
