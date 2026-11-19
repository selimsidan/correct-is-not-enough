"""Published models on dev and mini-dev in our harness, for comparison only (python -m effsql.gen.reference).

Greedy predictions written like `effsql predict` output, so `effsql evaluate` and the training report
score them unchanged: runs/<run>/predictions_<split>.jsonl, predict_raw_<split>.jsonl,
predict_stats_<split>.json. Each model gets the prompt it was published with (`style` in
configs/reference.yaml):

  effsql   our zero-shot prompt (configs/prompts/system_base.txt and user.txt), as for our own models
  omnisql  the OmniSQL template: one user message, think step by step, SQL in a ```sql block
           (github.com/RUCKBReasoning/OmniSQL, train_and_evaluate/infer.py and the model card)
  arctic   the Arctic-Text2SQL-R1 prompt: system and user message, and the answer starts with
           "Let me solve this step by step. \\n<think>" (github.com/snowflakedb/ArcticTraining,
           projects/arctic_text2sql_r1/bird_eval/infer.py)

For omnisql and arctic the schema is written as OmniSQL writes it: CREATE TABLE with one line per
column, its description and up to two example values as a comment, then the keys; the question is the
evidence, a newline and the question. Two differences from the published setup: the example values
come from our cached sample rows, and values matching the question are not retrieved (OmniSQL's BM25
index). The SQL is the last ```sql block of the answer, as in both papers' code.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any

from effsql.config import load_config
from effsql.data.bird import load_examples, select_questions
from effsql.gen.backend import GenParams, VLLMBackend
from effsql.gen.parse import extract_sql
from effsql.gen.prompts import Messages, build_question_prompt, load_templates
from effsql.gen.sample import Request, is_current_gen, run_requests, schemas_for
from effsql.io import latest_by, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)

TASK = ("You are a data science expert. Below, you are provided with a database schema and a natural language "
        "question. Your task is to understand the schema and generate a valid SQL query to answer the question.")
SCHEMA_NOTE = ("This schema describes the database's structure, including tables, columns, primary keys, foreign "
               "keys, and any relevant relationships or constraints.")
INSTRUCTIONS = (
    "Instructions:\n"
    "- Make sure you only output the information that is asked in the question. If the question asks for a "
    "specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
    "- The generated query should return all of the information asked in the question without any missing or "
    "extra information.\n"
    "- Before generating the final SQL query, please think through the steps of how to write the query.")
OMNISQL_OUTPUT = ("Output Format:\nIn your answer, please enclose the generated SQL query in a code block:\n"
                  "```sql\n-- Your SQL query\n```\n\n"
                  "Take a deep breath and think step by step to find the correct SQL query.")
ARCTIC_OUTPUT = (
    "Output Format:\n"
    "Please provide a detailed chain-of-thought reasoning process and include your thought process within "
    "`<think>` tags. Your final answer should be enclosed within `<answer>` tags.\n\n"
    "Ensure that your SQL query follows the correct syntax and is formatted as follows:\n\n"
    "```sql\n-- Your SQL query here\n```\n\n"
    "Example format:\n"
    "<think> Step-by-step reasoning, including self-reflection and corrections if necessary. "
    "[Limited by 4K tokens] </think>\n"
    "<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]\n\n"
    "```sql\nCorrect SQL query here\n```\n</answer>")
ARCTIC_PREFILL = "Let me solve this step by step. \n<think>"

# OmniSQL's list (process_dataset.py): these names, and names with other characters, get backticks.
RESERVED = set("""
    ACCESS ADD AFTER ALL ALTER ANALYZE AND AS ASC AVG BEFORE BEGIN BETWEEN BIGINT BINARY BIT BLOB BOOLEAN BOTH
    BREAK BY CALL CASCADE CASE CAST CHAR CHARACTER CHECK CLOB COLLATE COLUMN COMMENT COMMIT CONDITION CONNECT
    CONSTRAINT CONTINUE CONVERT COUNT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP CURSOR
    DATABASE DATE DAY DEC DECIMAL DECLARE DEFAULT DELETE DESC DESCRIBE DISTINCT DO DOUBLE DROP EACH ELSE ELSEIF
    END ESCAPE EXCEPT EXEC EXECUTE EXISTS EXIT EXPLAIN EXTEND EXTERNAL FALSE FETCH FILTER FLOAT FOR FOREIGN FROM
    FULL FUNCTION GENERATED GLOBAL GRANT GROUP HAVING HOLD HOUR IDENTIFIED IF IGNORE ILIKE IN INDEX INNER INOUT
    INSENSITIVE INSERT INT INTEGER INTERSECT INTERVAL INTO IS ITERATE JOIN KEY LANGUAGE LEADING LEAVE LEFT LIKE
    LIMIT LOCAL LOCALTIME LOCALTIMESTAMP LOCK LOOP MATCH MAX MERGE MIN MINUTE MODIFIES MODULE MONTH NATURAL
    NCHAR NEW NO NOT NULL NUMERIC OF OFFSET OLD ON ONLY OPEN OR ORDER OUT OUTER OVER PARTITION PERCENT PLACING
    POSITION PRECISION PRIMARY PROCEDURE RANGE READS REAL RECURSIVE REFERENCES REGEXP RELEASE RENAME REPEAT
    REPLACE RESIGNAL RESTRICT RETURN REVOKE RIGHT RLIKE ROLE ROLLBACK ROW ROWNUM ROWS SAVEPOINT SCHEMA SELECT
    SENSITIVE SESSION SET SIGNAL SMALLINT SOME SPECIFIC SQL SQLEXCEPTION SQLSTATE SQLWARNING START STATIC
    SUBSTRING SUM SYSDATE TABLE TERMINATED THEN TIME TIMESTAMP TO TOP TRAILING TRIGGER TRUE UNDO UNION UNIQUE
    UNKNOWN UNLOCK UPDATE USAGE USER USING VALUE VALUES VARCHAR VARYING VIEW WHEN WHERE WHILE WITH WITHIN YEAR
    ZONE""".split())  # noqa: SIM905 (one readable block)
_PLAIN = re.compile(r"[A-Za-z0-9_]+")


def ident(name: str) -> str:
    return name if _PLAIN.fullmatch(name) and name.upper() not in RESERVED else f"`{name}`"


def _examples(table: dict[str, Any], col: str, n: int = 2) -> list[Any]:
    """Up to n distinct example values of a column from the cached sample rows (strings cut at 40)."""
    cols = table.get("sample_columns") or []
    if col not in cols:
        return []
    i, out = cols.index(col), []
    for row in table.get("sample_rows") or []:
        v = row[i]
        if v is None or v == "":
            continue
        v = v[:40] if isinstance(v, str) else v
        if v not in out:
            out.append(v)
    return out[:n]


def omnisql_schema(schema: dict[str, Any], values: bool = True) -> str:
    """The schema as OmniSQL writes it (process_dataset.py, obtain_db_details for evaluation)."""
    fks: dict[str, list[tuple[str, str, str]]] = {}
    for fk in schema.get("foreign_keys_pragma") or []:
        t, _, c = fk["from"].partition(".")
        tt, _, tc = fk["to"].partition(".")
        fks.setdefault(t.lower(), []).append((c, tt, tc))
    parts = []
    for t in schema["tables"]:
        desc = {d.get("original_column_name"): d for d in t.get("column_descriptions") or []}
        lines, pks = [], []
        for c in t["columns"]:
            name = c["name"]
            d = desc.get(name) or {}
            comment = (d.get("column_name") or d.get("column_description") or "").strip()
            low = comment.lower()
            if name.lower() in (low, low.replace(" ", "_"), low.replace(" ", "")):  # only respells the name
                comment = ""
            ex = _examples(t, name) if values else []
            line = f"    {ident(name)} {(c.get('type') or '').lower()},"
            if comment:
                line += f" -- {' '.join(comment.split())}" + (f", example: {ex}" if ex else "")
            elif ex:
                line += f" -- example: {ex}"
            lines.append(line)
            if c.get("pk"):
                pks.append((c["pk"], name))
        if pks:
            lines.append("    PRIMARY KEY (" + ", ".join(ident(n) for _, n in sorted(pks)) + "),")
        for c, tt, tc in dict.fromkeys(fks.get(t["name"].lower(), [])):
            lines.append(f"    CONSTRAINT fk_{t['name'].lower().replace(' ', '_')}_{c.lower().replace(' ', '_')} "
                         f"FOREIGN KEY ({ident(c)}) REFERENCES {ident(tt)} ({ident(tc)}),")
        body = "\n".join(lines)
        parts.append(f"CREATE TABLE {ident(t['name'])} (\n{body.removesuffix(',')}\n);")
    return "\n\n".join(parts)


def omnisql_question(e: Any) -> str:
    return f"{e.evidence}\n{e.question}" if (e.evidence or "").strip() else e.question


def style_messages(style: str, db_text: str, question: str) -> Messages:
    body = (f"Database Engine:\nSQLite\n\nDatabase Schema:\n{db_text}\n{SCHEMA_NOTE}\n\n"
            f"Question:\n{question}\n\n{INSTRUCTIONS}\n\n")
    if style == "omnisql":
        return [{"role": "user", "content": f"Task Overview:\n{TASK}\n\n{body}{OMNISQL_OUTPUT}"}]
    if style == "arctic":
        return [{"role": "system", "content": TASK}, {"role": "user", "content": body + ARCTIC_OUTPUT}]
    raise ValueError(f"unknown prompt style {style!r}")


def reference_prompt(style: str, templates: dict[str, str], e: Any, schema: dict[str, Any], budget: int,
                     count_tokens: Any) -> tuple[Messages, str, int]:
    """(messages, schema level, prompt tokens). A schema too long with example values goes without them."""
    if style == "effsql":
        return build_question_prompt(templates, "base", e, schema, budget, count_tokens)
    for level, values in (("values", True), ("no_values", False)):
        msgs = style_messages(style, omnisql_schema(schema, values), omnisql_question(e))
        n = count_tokens(msgs)
        if n <= budget:
            break
    return msgs, level, n


class PrefillBackend(VLLMBackend):
    """vLLM engine whose answers start with fixed text after the chat template (Arctic's "<think>")."""

    prefill = ""

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        return super()._prompt(messages) + self.prefill


def make_reference_backend(model_cfg: dict[str, Any], seed: int = 0) -> PrefillBackend:
    b = PrefillBackend(model_cfg, seed=seed)
    b.prefill = ARCTIC_PREFILL if model_cfg["style"] == "arctic" else ""
    return b


def predict_reference(name: str, split: str, run: str, backend: Any, *, paths: Paths | None = None,
                      cfg: dict[str, Any] | None = None, limit: int | None = None) -> dict[str, Any]:
    if split not in ("dev", "minidev"):
        raise ValueError("reference predictions are for dev and minidev")
    paths = paths or load_paths()
    cfg = cfg or load_config("reference.yaml")
    m = cfg["models"][name]
    examples = select_questions(load_examples(split, paths), limit)
    schemas = schemas_for(split, {e.db_id for e in examples}, paths)
    templates = load_templates()
    params = GenParams(n=1, temperature=0.0, top_p=1.0, max_tokens=m["max_tokens"])
    requests = []
    for e in sorted(examples, key=lambda x: x.db_id):
        msgs, level, n_tok = reference_prompt(m["style"], templates, e, schemas[e.db_id], m["max_prompt_tokens"],
                                              backend.count_tokens)
        requests.append(Request(e.qid, msgs, params, {"qid": e.qid, "split": split, "db_id": e.db_id,
                                                      "reference": name, "style": m["style"],
                                                      "source": "reference", "prompt_level": level,
                                                      "prompt_tokens": n_tok}))
    run_dir = paths.run_dir(run)
    raw = run_dir / f"predict_raw_{split}.jsonl"
    stats = run_requests(requests, backend, raw, cfg["chunk_size"], f"reference {name} {split}")
    recs = latest_by(raw, key="key", where=is_current_gen)
    lines, n_tokens, cut = [], [], 0
    for e in examples:
        r = recs.get(e.qid)
        comp = r["completions"][0] if r and r["completions"] else {"text": "", "finish_reason": None, "n_tokens": 0}
        sql, how = extract_sql(comp["text"])
        n_tokens.append(comp.get("n_tokens") or 0)
        cut += comp.get("finish_reason") == "length"
        lines.append(json.dumps({"qid": e.qid, "sql": sql or "", "extraction": how,
                                 "finish_reason": comp.get("finish_reason"), "n_tokens": comp.get("n_tokens"),
                                 "prompt_level": r.get("prompt_level") if r else None,
                                 "model": backend.model_id, "model_revision": backend.model_revision}))
    out = run_dir / f"predictions_{split}.jsonl"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats.update(n_questions=len(examples), predictions=str(out), reference=name, style=m["style"],
                 mean_answer_tokens=sum(n_tokens) / max(len(n_tokens), 1), n_cut_off=cut)
    write_json(run_dir / f"predict_stats_{split}.json", stats)
    log.info("%s %s: %d predictions, %.0f answer tokens on average, %d cut off", name, split, len(examples),
             stats["mean_answer_tokens"], cut)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m effsql.gen.reference", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="a key of configs/reference.yaml models")
    ap.add_argument("--splits", default="dev,minidev")
    ap.add_argument("--run", help="output run (default ref_<model>)")
    ap.add_argument("--limit", type=int, help="smoke run: this many questions per split")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config("reference.yaml")
    backend = make_reference_backend(cfg["models"][a.model], seed=cfg["seed"])
    for split in a.splits.split(","):
        s = predict_reference(a.model, split, a.run or f"ref_{a.model}", backend, cfg=cfg, limit=a.limit)
        print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
