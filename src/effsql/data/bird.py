"""BIRD question loader.

Field names verified on the real files (2026-11-08, via the Drive API):
  dev.json, mini_dev_sqlite.json: question_id, db_id, question, evidence, SQL, difficulty
  train.json:                     db_id, question, evidence, SQL  (no question_id, no difficulty)
Gold .sql files hold one `<sql>\\t<db_id>` per line, in the same order as the JSON.
The gold file is what the official evaluation reads, so it is the source of `gold_sql`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)

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


def make_qid(split: str, record: dict[str, Any], index: int, mode: str) -> str:
    """Stable id per configs/paths.yaml: `<split>_<question_id>` or `<split>_<position>`.

    Mini-dev uses the position because its question_ids repeat (inspect report 2026-11-08);
    the official evaluation also aligns mini-dev predictions by position.
    """
    if mode == "index":
        return f"{split}_{index}"
    if record.get("question_id") is None:
        raise ValueError(f"{split} record {index} has no question_id (set qid: index in paths.yaml)")
    return f"{split}_{record['question_id']}"


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

    mode = paths.qid_mode(split)
    examples: list[Example] = []
    for i, (rec, g) in enumerate(zip(records, gold)):
        if rec["db_id"] != g.db_id:
            raise ValueError(f"{split} record {i}: db_id {rec['db_id']!r} != gold db_id {g.db_id!r}")
        json_sql = (rec.get("SQL") or "").strip()
        examples.append(
            Example(
                qid=make_qid(split, rec, i, mode),
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

    Used only for the secondary EX-tied metric on dev; headline metrics use the official gold.
    Records whose question text differs from dev's are dropped (3 of 42 in the real file),
    so an alternative gold never answers a reworded question.
    """
    paths = paths or load_paths()
    records = read_questions(paths.split_file("dev", "tied_gold"))
    dev_q = {r["question_id"]: r["question"] for r in read_questions(paths.split_file("dev", "questions"))}
    kept = {
        int(r["question_id"]): GoldLine(sql=r["SQL"].strip(), db_id=r["db_id"])
        for r in records
        if dev_q.get(r["question_id"]) == r["question"]
    }
    log.info("tied gold: kept %d of %d (dropped reworded or unknown questions)", len(kept), len(records))
    return kept


def select_questions(examples: list[Example], limit: int | None) -> list[Example]:
    """First `limit` questions taken round-robin across databases (sorted db_id, file order within
    a database), so a small smoke run covers many databases instead of the first one only."""
    if not limit:
        return examples
    by_db: dict[str, list[Example]] = {}
    for e in examples:
        by_db.setdefault(e.db_id, []).append(e)
    queues = [by_db[d] for d in sorted(by_db)]
    out: list[Example] = []
    i = 0
    while len(out) < limit and any(i < len(q) for q in queues):
        for q in queues:
            if i < len(q) and len(out) < limit:
                out.append(q[i])
        i += 1
    return out
