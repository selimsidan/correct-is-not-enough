"""`effsql parse`: model answers -> SQL candidates, deduplicated per question.

Extraction: the last ```sql block; else the last ``` block; else an unclosed ```sql block (answer
cut off); else raw text starting with SELECT or WITH; else `parse_fail`.
Canonical form: sqlglot (SQLite dialect) pretty-printed with upper-case keywords and lower-cased
unquoted identifiers (SQLite identifiers are case-insensitive). It is only the dedup key and
display text: the model's original SQL is what gets executed, so a sqlglot rewrite can never change
a result. If sqlglot cannot parse a query, the whitespace-normalized original is the key.

The candidates file is rebuilt from all generation and rewrite records on every call (cheap and
deterministic). cand_id is derived from the dedup key, so it is stable across rebuilds and
screening can resume by cand_id.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import Any

import sqlglot

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.gen.sample import is_current_gen
from effsql.io import latest_by, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
logging.getLogger("sqlglot").setLevel(logging.ERROR)
PARSE_VERSION = 1
_BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)
_UNCLOSED = re.compile(r"```[ \t]*(?:sql|sqlite)?[^\n]*\n(.*)$", re.S | re.I)
_RAW_START = re.compile(r"^\s*(SELECT|WITH)\b", re.I)
# Order in which occurrences are listed; the first occurrence provides sql_original.
SOURCE_RANK = {"gold": 0, "greedy": 1, "sample": 2}


def _clean(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def extract_sql(text: str | None) -> tuple[str | None, str]:
    """(sql, how) where how is sql_block, code_block, unclosed_block, raw or parse_fail."""
    if not text or not text.strip():
        return None, "parse_fail"
    blocks = _BLOCK.findall(text)
    sql_blocks = [b for lang, b in blocks if lang.lower() in ("sql", "sqlite")]
    for candidates, how in ((sql_blocks, "sql_block"), ([b for _, b in blocks], "code_block")):
        if candidates and _clean(candidates[-1]):
            return _clean(candidates[-1]), how
    # A fence opened and never closed (answer cut off): take what follows the last fence.
    m = _UNCLOSED.match(text[text.rfind("```"):]) if text.count("```") % 2 == 1 else None
    if m and _clean(m.group(1)):
        return _clean(m.group(1)), "unclosed_block"
    if _RAW_START.match(text):
        return _clean(text), "raw"
    return None, "parse_fail"


def canonicalize(sql: str) -> tuple[str, bool]:
    try:
        exprs = [e for e in sqlglot.parse(sql, read="sqlite") if e is not None]
        if len(exprs) != 1:
            raise ValueError(f"{len(exprs)} statements")
        return exprs[0].sql(dialect="sqlite", pretty=True, normalize=True), True
    except Exception:  # noqa: BLE001 (sqlglot raises several error types; the database is the judge)
        return " ".join(sql.split()), False


def cand_id(qid: str, key: str) -> str:
    return f"{qid}#{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def _source_rank(source: str) -> int:
    return SOURCE_RANK.get(source, 3)


def parse_run(split: str, run: str, *, paths: Paths | None = None, pool_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    pool_cfg = pool_cfg or load_config("pool.yaml")
    run_dir = paths.run_dir(run)
    gens = list(latest_by(run_dir / f"generations_{split}.jsonl", key="key", where=is_current_gen).values())
    gens += list(latest_by(run_dir / f"rewrites_{split}.jsonl", key="key", where=is_current_gen).values())
    if not gens:
        raise RuntimeError(f"No generations in {run_dir}. Run `effsql generate` first.")
    examples = {e.qid: e for e in load_examples(split, paths)}
    qids = sorted({g["qid"] for g in gens})

    occurrences: list[tuple[str, dict[str, Any], str | None]] = []  # (qid, occurrence, sql)
    if pool_cfg.get("include_gold", True) and split == "train":
        for q in qids:
            occurrences.append((q, {"source": "gold", "variant": None, "idx": 0, "extraction": "gold",
                                    "finish_reason": None, "round": 0}, examples[q].gold_sql))
    for g in sorted(gens, key=lambda r: (r["qid"], _source_rank(r["source"]), r.get("variant") or "", r["key"])):
        for i, c in enumerate(g["completions"]):
            sql, how = extract_sql(c["text"])
            occurrences.append((g["qid"], {"source": g["source"], "variant": g.get("variant"), "idx": i,
                                           "extraction": how, "finish_reason": c.get("finish_reason"),
                                           "round": g.get("round", 0), "gen_key": g["key"],
                                           "model": g.get("model")}, sql))

    cands: dict[str, dict[str, Any]] = {}
    for q, occ, sql in occurrences:
        if sql is None:
            cid, canon, ok = f"{q}#parse_fail", None, False
        else:
            canon, ok = canonicalize(sql)
            cid = cand_id(q, canon)
        c = cands.get(cid)
        if c is None:
            e = examples[q]
            c = cands[cid] = {
                "cand_id": cid, "qid": q, "split": split, "db_id": e.db_id, "difficulty": e.difficulty,
                "sql_original": sql, "sql_canonical": canon, "canonical_ok": ok,
                "first_source": occ["source"], "first_variant": occ["variant"], "model": occ.get("model"),
                "occurrences": [], "parse_version": PARSE_VERSION,
            }
        c["occurrences"].append(occ)
    for c in cands.values():
        c["sources"] = sorted({o["source"] for o in c["occurrences"]}, key=_source_rank)
        c["variants"] = sorted({o["variant"] for o in c["occurrences"] if o["variant"]})
        c["n_occurrences"] = len(c["occurrences"])
        c["is_gold"] = "gold" in c["sources"]

    out = run_dir / f"candidates_{split}.jsonl"
    tmp = out.with_suffix(".tmp")
    import json

    with open(tmp, "w", encoding="utf-8") as f:
        for c in sorted(cands.values(), key=lambda c: (c["qid"], c["cand_id"])):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(out)

    gen_occ = [o for _, o, _ in occurrences if o["source"] != "gold"]
    unique_exec = [c for c in cands.values() if c["sql_original"] is not None]
    per_q = Counter(c["qid"] for c in unique_exec)
    stats = {
        "n_questions": len(qids),
        "n_completions": len(gen_occ),
        "extraction": dict(Counter(o["extraction"] for o in gen_occ)),
        "finish_reason": dict(Counter(o["finish_reason"] for o in gen_occ)),
        "n_unique_candidates": len(unique_exec),
        "unique_per_question_mean": len(unique_exec) / max(len(qids), 1),
        "unique_per_question_hist": dict(sorted(Counter(per_q.values()).items())),
        "n_canonical_failed": sum(1 for c in unique_exec if not c["canonical_ok"]),
        "n_gold_reproduced_by_model": sum(1 for c in cands.values() if c["is_gold"] and len(c["sources"]) > 1),
    }
    write_json(run_dir / f"parse_stats_{split}.json", stats)
    log.info("parse %s: %d completions -> %d unique candidates (%.2f per question), %d parse failures",
             split, stats["n_completions"], stats["n_unique_candidates"], stats["unique_per_question_mean"],
             stats["extraction"].get("parse_fail", 0))
    return stats
