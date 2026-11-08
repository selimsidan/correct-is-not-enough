"""Prompt construction: schema rendering, templates and the shrink ladder.

The instructions sit in the system message and the schema leads the user message, so all prompts
for one database share the longest possible prefix (vLLM prefix caching). Schemas that do not fit
the token budget are shortened step by step (SHRINK_LADDER); the level used is recorded.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from effsql.config import config_dir

Messages = list[dict[str, str]]
CountTokens = Callable[[Messages], int]

# Each level keeps less schema detail than the previous one.
SHRINK_LADDER: list[dict[str, Any]] = [
    {"name": "full", "rows": 3, "value_chars": 80, "value_desc": True, "col_desc": True},
    {"name": "short_values", "rows": 3, "value_chars": 30, "value_desc": True, "col_desc": True},
    {"name": "one_row", "rows": 1, "value_chars": 30, "value_desc": True, "col_desc": True},
    {"name": "no_value_desc", "rows": 1, "value_chars": 30, "value_desc": False, "col_desc": True},
    {"name": "no_rows", "rows": 0, "value_chars": 30, "value_desc": False, "col_desc": True},
    {"name": "no_desc", "rows": 0, "value_chars": 30, "value_desc": False, "col_desc": False},
]
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def estimate_tokens(messages: Messages) -> int:
    """Rough count (about 3.5 characters per token) for CPU previews without a tokenizer."""
    return int(sum(len(m["content"]) for m in messages) / 3.5) + 10


def fill(template: str, values: dict[str, Any]) -> str:
    """Single-pass placeholder substitution: inserted text is never re-scanned, so braces inside
    schemas or questions are safe."""
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)


def load_templates() -> dict[str, str]:
    d = config_dir() / "prompts"
    return {p.stem: p.read_text(encoding="utf-8").strip() for p in d.glob("*.txt")}


def _one_line(v: Any, n: int) -> str:
    s = "NULL" if v is None else str(v)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def render_schema(schema: dict[str, Any], level: dict[str, Any]) -> str:
    """CREATE TABLE text plus row counts, column descriptions and example rows as SQL comments."""
    parts = []
    for t in schema["tables"]:
        lines = [t["ddl"].strip().rstrip(";") + ";", f"-- {t['row_count']} rows"]
        if level["col_desc"] and t.get("column_descriptions"):
            lines.append("-- Column descriptions:")
            for d in t["column_descriptions"]:
                col = d.get("original_column_name") or d.get("column_name") or "?"
                bits = []
                full = d.get("column_name")
                if full and full.strip().lower() != col.strip().lower():
                    bits.append(full)
                if d.get("column_description"):
                    bits.append(d["column_description"])
                if d.get("data_format"):
                    bits.append(f"format {d['data_format']}")
                if level["value_desc"] and d.get("value_description"):
                    bits.append(f"values: {d['value_description']}")
                lines.append(f"--   {col}: " + _one_line("; ".join(bits), 300))
        if level["rows"] and t.get("sample_rows"):
            lines.append(f"-- Example rows ({min(level['rows'], len(t['sample_rows']))}):")
            lines.append("--   " + " | ".join(t["sample_columns"]))
            for r in t["sample_rows"][: level["rows"]]:
                lines.append("--   " + " | ".join(_one_line(v, level["value_chars"]) for v in r))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def fit_messages(
    make: Callable[[str], Messages], schema: dict[str, Any], budget: int, count_tokens: CountTokens
) -> tuple[Messages, str, int]:
    """Walk the shrink ladder until the messages fit the budget. Returns (messages, level, tokens);
    if even the last level is too long, it is returned anyway and the caller decides."""
    for level in SHRINK_LADDER:
        msgs = make(render_schema(schema, level))
        n = count_tokens(msgs)
        if n <= budget:
            return msgs, level["name"], n
    return msgs, SHRINK_LADDER[-1]["name"], n


def question_messages(
    templates: dict[str, str], variant: str, db_id: str, question: str, evidence: str, schema_text: str
) -> Messages:
    system = templates[f"system_{variant}"]
    user = fill(templates["user"], {"db_id": db_id, "schema": schema_text,
                                    "question": question, "evidence": evidence or "(none)"})
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_question_prompt(
    templates: dict[str, str], variant: str, example: Any, schema: dict[str, Any], budget: int,
    count_tokens: CountTokens,
) -> tuple[Messages, str, int]:
    return fit_messages(
        lambda s: question_messages(templates, variant, example.db_id, example.question, example.evidence, s),
        schema, budget, count_tokens,
    )


def build_rewrite_prompt(
    templates: dict[str, str], example: Any, schema: dict[str, Any], seed_sql: str, plan: str, style: str,
    style_instruction: str, budget: int, count_tokens: CountTokens,
) -> tuple[Messages, str, int]:
    def make(schema_text: str) -> Messages:
        user = fill(templates["rewrite_user"], {
            "db_id": example.db_id, "schema": schema_text, "question": example.question,
            "evidence": example.evidence or "(none)", "seed_sql": seed_sql, "plan": plan,
            "style": style, "style_instruction": style_instruction,
        })
        return [{"role": "system", "content": templates["rewrite_system"]}, {"role": "user", "content": user}]

    return fit_messages(make, schema, budget, count_tokens)
