"""Deterministic cost: relative cost and efficiency score from vm_steps, plus plan and text features.

`rel_cost` and `eff_score` are the single place where these formulas live. Both add a smoothing
constant c (default: the progress handler period N) because vm_steps is a multiple of N and can be
0 for tiny queries.
"""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any

import sqlglot
from sqlglot import exp


def rel_cost(steps: int, gold_steps: int, c: float) -> float:
    """steps(pred) / steps(gold), smoothed. Below 1 means cheaper than gold."""
    return (steps + c) / (gold_steps + c)


def eff_score(steps: int, gold_steps: int, c: float) -> float:
    """log2(gold / pred), smoothed. Positive means cheaper than gold."""
    return math.log2((gold_steps + c) / (steps + c))


_NUM = re.compile(r"\b\d+\b")
_TARGET = re.compile(r"^(SCAN|SEARCH)( TABLE)? (\S+)(.*)$")


def alias_map(sql: str) -> dict[str, str]:
    """Map lowercased aliases and CTE names to what they stand for, so plan signatures do not
    depend on alias spelling (SQLite prints the alias: `SCAN T2`)."""
    out: dict[str, str] = {}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return out
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            out[cte.alias.lower()] = "<cte>"
    for sub in tree.find_all(exp.Subquery):
        if sub.alias:
            out[sub.alias.lower()] = "<subquery>"
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if t.alias:
            out[t.alias.lower()] = name
        out.setdefault(name, name)
    return out


def plan_features(conn: sqlite3.Connection, sql: str) -> dict[str, Any]:
    """EXPLAIN QUERY PLAN counts plus a normalized plan signature (depth:detail;...)."""
    try:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    except Exception as e:  # noqa: BLE001
        return {"plan_ok": False, "plan_error": f"{type(e).__name__}: {e}"[:200]}
    aliases = alias_map(sql)
    depth: dict[int, int] = {0: -1}
    parts = []
    feats = {"n_scan": 0, "n_search": 0, "n_temp_btree": 0, "n_correlated": 0,
             "n_subquery": 0, "n_automatic_index": 0, "n_covering_index": 0}
    scanned = []
    for node_id, parent, _unused, detail in rows:
        depth[node_id] = depth.get(parent, -1) + 1
        d = detail.strip()
        m = _TARGET.match(d)
        if m:
            kind, _, target, rest = m.groups()
            target = aliases.get(target.lower(), target.lower())
            d = f"{kind} {target}{rest}"
            if kind == "SCAN":
                feats["n_scan"] += 1
                scanned.append(target)
            else:
                feats["n_search"] += 1
        feats["n_temp_btree"] += "USE TEMP B-TREE" in d
        feats["n_correlated"] += "CORRELATED" in d
        feats["n_subquery"] += "SUBQUERY" in d or d.startswith(("MATERIALIZE", "CO-ROUTINE"))
        feats["n_automatic_index"] += "AUTOMATIC" in d
        feats["n_covering_index"] += "COVERING INDEX" in d
        parts.append(f"{depth[node_id]}:{_NUM.sub('N', d)}")
    return {"plan_ok": True, **feats, "scanned_tables": scanned, "plan_signature": ";".join(parts)}


def _depth(node: exp.Expression) -> int:
    d, p = 0, node.parent
    while p is not None:
        d += isinstance(p, exp.Select)
        p = p.parent
    return d


def text_features(sql: str) -> dict[str, Any]:
    """Structure of the SQL text (sqlglot, SQLite dialect)."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        n_tokens = len(sqlglot.tokenize(sql, read="sqlite"))
    except Exception:  # noqa: BLE001
        return {"text_ok": False}
    selects = list(tree.find_all(exp.Select))
    return {
        "text_ok": True,
        "n_joins": len(list(tree.find_all(exp.Join))),
        "n_selects": len(selects),
        "nesting_depth": max((_depth(s) for s in selects), default=0),
        "n_ctes": len(list(tree.find_all(exp.CTE))),
        "has_window": any(True for _ in tree.find_all(exp.Window)),
        "n_tokens": n_tokens,
    }
