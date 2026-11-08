"""BIRD question loader.

Field names verified on the real files (2026-11-08, via the Drive API):
  dev.json, mini_dev_sqlite.json: question_id, db_id, question, evidence, SQL, difficulty
  train.json:                     db_id, question, evidence, SQL  (no question_id, no difficulty)
Gold .sql files hold one `<sql>\\t<db_id>` per line, in the same order as the JSON.
The gold file is what the official evaluation reads, so it is the source of `gold_sql`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from effsql.paths import Paths, load_paths


@dataclass
class Example:
    qid: str
    bird_question_id: int | None
    split: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str | None
    # JSON "SQL" field, kept only when it differs from the gold file line.
    json_sql: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoldLine:
    sql: str
    db_id: str


def make_qid(split: str, record: dict[str, Any], index: int) -> str:
    """Stable id: `<split>_<question_id>` when the record has one, else `<split>_<index>`."""
    qid = record.get("question_id")
    return f"{split}_{qid if qid is not None else index}"


def read_questions(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    return data


def read_gold_file(path: Path) -> list[GoldLine]:
    """Parse `<sql>\\t<db_id>` lines. Splits on the last tab so a tab inside the SQL survives."""
    out: list[GoldLine] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{lineno}: no tab separator in gold line: {line[:120]!r}")
            sql, db_id = line.rsplit("\t", 1)
            out.append(GoldLine(sql=sql.strip(), db_id=db_id.strip()))
    return out


def load_examples(split: str, paths: Paths | None = None) -> list[Example]:
    """Load one split, aligning JSON records with gold lines by position (as official eval does)."""
    paths = paths or load_paths()
    records = read_questions(paths.split_file(split, "questions"))
    gold = read_gold_file(paths.split_file(split, "gold"))
    if len(records) != len(gold):
        raise ValueError(f"{split}: {len(records)} questions but {len(gold)} gold lines")

    examples: list[Example] = []
    for i, (rec, g) in enumerate(zip(records, gold)):
        if rec["db_id"] != g.db_id:
            raise ValueError(f"{split} record {i}: db_id {rec['db_id']!r} != gold db_id {g.db_id!r}")
        json_sql = (rec.get("SQL") or "").strip()
        examples.append(
            Example(
                qid=make_qid(split, rec, i),
                bird_question_id=rec.get("question_id"),
                split=split,
                db_id=rec["db_id"],
                question=rec["question"],
                evidence=rec.get("evidence") or "",
                gold_sql=g.sql,
                difficulty=rec.get("difficulty"),
                json_sql=json_sql if json_sql != g.sql else None,
            )
        )
    qids = [e.qid for e in examples]
    if len(set(qids)) != len(qids):
        raise ValueError(f"{split}: duplicate qids")
    return examples


def load_tied_gold(paths: Paths | None = None) -> dict[int, GoldLine]:
    """Alternative gold SQL for dev questions with tied answers, keyed by BIRD question_id.

    Used only for the secondary EX-tied metric; headline metrics use the official gold.
    Applies to minidev too, through bird_question_id.
    """
    paths = paths or load_paths()
    records = read_questions(paths.split_file("dev", "tied_gold"))
    return {int(r["question_id"]): GoldLine(sql=r["SQL"].strip(), db_id=r["db_id"]) for r in records}
