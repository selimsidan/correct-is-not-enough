"""`effsql inspect`: verify the real BIRD files before relying on any assumption about them.

Checks (all read-only, nothing is extracted):
  1. zip structure: prefixes, junk entries, sample members, databases found
  2. gold files: first lines, separator, line counts vs JSON records, db_id order
  3. counts: records, distinct db_ids, every JSON db_id present in the zip
  4. JSON SQL field vs gold file line by line
  5. minidev vs dev by question_id
  6. column descriptions: dev zip copy vs minidev/MINIDEV copy
plus JSON keys per split and a summary of dev_tied_append.json.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from effsql.data.bird import read_gold_file, read_questions
from effsql.data.zipindex import index_zip
from effsql.paths import Paths, is_junk, load_paths

log = logging.getLogger(__name__)
MAX_EXAMPLES = 5


def _short(s: str | None, n: int = 160) -> str | None:
    return s if s is None or len(s) <= n else s[:n] + "..."


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def inspect_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        idx = index_zip(zf)
        names = [i.filename for i in zf.infolist() if not is_junk(i.filename)]
    sizes = {d: m.sqlite.file_size for d, m in idx.dbs.items() if m.sqlite}
    return {
        "path": str(path),
        "n_entries": idx.n_entries,
        "n_junk_entries": idx.n_junk,
        "top_level": dict(idx.top_level.most_common()),
        "sample_entries": names[:15],
        "nested_zips": idx.nested_zips[:10],
        "n_databases": len(sizes),
        "db_ids": sorted(sizes),
        "description_only_db_ids": sorted(d for d, m in idx.dbs.items() if not m.sqlite),
        "description_files_per_db": {d: len(m.descriptions) for d, m in sorted(idx.dbs.items())},
        "sqlite_sizes_mb": {d: round(s / 2**20, 1) for d, s in sorted(sizes.items())},
        "total_uncompressed_gb": round(sum(m.uncompressed_bytes for m in idx.dbs.values()) / 2**30, 2),
        "other_members_sample": idx.other[:10],
        "n_other_members": len(idx.other),
    }


def inspect_questions(records: list[dict[str, Any]]) -> dict[str, Any]:
    key_counts = Counter(k for r in records for k in r)
    qids = [r.get("question_id") for r in records if "question_id" in r]
    out: dict[str, Any] = {
        "n_records": len(records),
        "first_record_keys": list(records[0].keys()) if records else [],
        "key_presence": dict(key_counts),
        "n_distinct_db_ids": len({r["db_id"] for r in records}),
        "difficulty": dict(Counter(r.get("difficulty") for r in records)),
        "n_empty_evidence": sum(1 for r in records if not (r.get("evidence") or "").strip()),
    }
    if qids:
        out["question_id_range"] = [min(qids), max(qids)]
        out["question_id_unique"] = len(set(qids)) == len(qids)
        out["first_question_ids"] = qids[:5]
    return out


def inspect_gold(gold_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    with open(gold_path, encoding="utf-8") as f:
        raw = f.read().splitlines()
    nonblank = [l for l in raw if l.strip()]
    gold = read_gold_file(gold_path)
    db_mismatch = [i for i, (r, g) in enumerate(zip(records, gold)) if r["db_id"] != g.db_id]
    exact, ws_only, diffs = 0, 0, []
    for i, (r, g) in enumerate(zip(records, gold)):
        j = (r.get("SQL") or "").strip()
        if j == g.sql:
            exact += 1
        elif _ws(j) == _ws(g.sql):
            ws_only += 1
        elif len(diffs) < MAX_EXAMPLES:
            diffs.append({"index": i, "json_sql": _short(j), "gold_sql": _short(g.sql)})
    n = min(len(records), len(gold))
    return {
        "path": str(gold_path),
        "first_lines_repr": [repr(l[:200]) for l in raw[:3]],
        "n_lines": len(raw),
        "n_nonblank_lines": len(nonblank),
        "tabs_per_line": dict(Counter(l.count("\t") for l in nonblank)),
        "line_count_matches_json": len(gold) == len(records),
        "n_db_id_order_mismatches": len(db_mismatch),
        "first_db_id_mismatches": db_mismatch[:MAX_EXAMPLES],
        "json_sql_vs_gold": {
            "compared": n,
            "identical": exact,
            "whitespace_only_diff": ws_only,
            "different": n - exact - ws_only,
            "examples": diffs,
        },
    }


def inspect_coverage(records: list[dict[str, Any]], zip_db_ids: list[str]) -> dict[str, Any]:
    used = {r["db_id"] for r in records}
    return {
        "missing_from_zip": sorted(used - set(zip_db_ids)),
        "in_zip_but_unused": sorted(set(zip_db_ids) - used),
    }


def inspect_minidev_vs_dev(
    mini: list[dict[str, Any]], dev: list[dict[str, Any]], mini_gold: list, dev_gold: list
) -> dict[str, Any]:
    dev_by_id = {r["question_id"]: (r, g) for r, g in zip(dev, dev_gold)}
    fields = ("db_id", "question", "evidence", "SQL", "difficulty")
    same = Counter()
    examples: dict[str, list] = {f: [] for f in fields + ("gold",)}
    missing = []
    for r, g in zip(mini, mini_gold):
        qid = r.get("question_id")
        if qid not in dev_by_id:
            missing.append(qid)
            continue
        d, dg = dev_by_id[qid]
        for f in fields:
            if (r.get(f) or "") == (d.get(f) or ""):
                same[f] += 1
            elif len(examples[f]) < MAX_EXAMPLES:
                examples[f].append({"question_id": qid, "minidev": _short(r.get(f)), "dev": _short(d.get(f))})
        if g.sql == dg.sql:
            same["gold"] += 1
        elif len(examples["gold"]) < MAX_EXAMPLES:
            examples["gold"].append({"question_id": qid, "minidev": _short(g.sql), "dev": _short(dg.sql)})
    matched = len(mini) - len(missing)
    return {
        "n_minidev": len(mini),
        "n_found_in_dev": matched,
        "question_ids_not_in_dev": missing[:20],
        "n_identical": {f: same[f] for f in fields + ("gold",)},
        "n_question_and_evidence_identical": sum(
            1 for r in mini
            if r.get("question_id") in dev_by_id
            and r.get("question") == dev_by_id[r["question_id"]][0].get("question")
            and (r.get("evidence") or "") == (dev_by_id[r["question_id"]][0].get("evidence") or "")
        ),
        "examples_of_differences": {f: v for f, v in examples.items() if v},
    }


def inspect_descriptions(dev_zip: Path, minidev_dir: Path) -> dict[str, Any]:
    """Compare description CSVs in the dev zip with the copies under minidev/MINIDEV."""
    def norm(b: bytes) -> bytes:
        return b"\n".join(l.rstrip() for l in b.replace(b"\r\n", b"\n").split(b"\n")).strip()

    per_db = {}
    with zipfile.ZipFile(dev_zip) as zf:
        idx = index_zip(zf)
        for db_id, m in sorted(idx.dbs.items()):
            zip_files = {PurePosixPath(i.filename).name: i for i in m.descriptions}
            mdir = minidev_dir / db_id / "database_description"
            mini_files = {p.name: p for p in mdir.glob("*.csv") if not is_junk(p.name)} if mdir.exists() else {}
            identical, norm_identical, different = 0, 0, []
            for name in sorted(set(zip_files) & set(mini_files)):
                a, b = zf.read(zip_files[name]), mini_files[name].read_bytes()
                if a == b:
                    identical += 1
                elif norm(a) == norm(b):
                    norm_identical += 1
                else:
                    different.append(name)
            per_db[db_id] = {
                "minidev_folder_exists": mdir.exists(),
                "only_in_zip": sorted(set(zip_files) - set(mini_files)),
                "only_in_minidev": sorted(set(mini_files) - set(zip_files)),
                "identical": identical,
                "identical_after_whitespace_normalization": norm_identical,
                "different": different,
            }
    return {
        "all_identical": all(
            not v["only_in_zip"] and not v["only_in_minidev"] and not v["different"] for v in per_db.values()
        ),
        "per_db": per_db,
    }


def inspect_tied(tied: list[dict[str, Any]], dev: list[dict[str, Any]], dev_gold: list) -> dict[str, Any]:
    dev_by_id = {r["question_id"]: (r, g) for r, g in zip(dev, dev_gold)}
    in_dev = [t for t in tied if t.get("question_id") in dev_by_id]
    return {
        "n_records": len(tied),
        "keys": list(tied[0].keys()) if tied else [],
        "question_ids": [t.get("question_id") for t in tied],
        "n_in_dev": len(in_dev),
        "n_same_question_text_as_dev": sum(
            1 for t in in_dev if t["question"] == dev_by_id[t["question_id"]][0]["question"]
        ),
        "n_sql_differs_from_dev_gold": sum(
            1 for t in in_dev if t["SQL"].strip() != dev_by_id[t["question_id"]][1].sql
        ),
        "sample": [
            {
                "question_id": t["question_id"],
                "tied_sql": _short(t["SQL"]),
                "dev_gold": _short(dev_by_id[t["question_id"]][1].sql),
            }
            for t in in_dev[:3]
        ],
    }


def _section(report: dict[str, Any], name: str, fn: Callable[[], Any]) -> Any:
    """Run one check; a failure is recorded in the report instead of stopping the others."""
    try:
        report[name] = fn()
    except Exception as e:  # noqa: BLE001
        log.exception("inspect section %s failed", name)
        report[name] = {"error": f"{type(e).__name__}: {e}"}
    return report[name]


def inspect_all(paths: Paths | None = None, splits: list[str] | None = None, zips: bool = True) -> dict[str, Any]:
    paths = paths or load_paths()
    splits = splits or ["train", "dev", "minidev"]
    report: dict[str, Any] = {"drive_root": str(paths.drive_root)}

    records: dict[str, list] = {}
    gold: dict[str, list] = {}
    for s in splits:
        try:
            records[s] = read_questions(paths.split_file(s, "questions"))
            gold[s] = read_gold_file(paths.split_file(s, "gold"))
        except Exception as e:  # noqa: BLE001
            report[f"{s}.load"] = {"error": f"{type(e).__name__}: {e}"}
    for s in records:
        _section(report, f"{s}.questions", lambda s=s: inspect_questions(records[s]))
        _section(report, f"{s}.gold", lambda s=s: inspect_gold(paths.split_file(s, "gold"), records[s]))

    zip_db_ids: dict[str, list[str]] = {}
    if zips:
        for s in sorted({paths.db_split(s) for s in splits}):
            z = _section(report, f"{s}.zip", lambda s=s: inspect_zip(paths.db_zip(s)))
            if "db_ids" in z:
                zip_db_ids[s] = z["db_ids"]
    for s in records:
        db_split = paths.db_split(s)
        if db_split in zip_db_ids:
            _section(report, f"{s}.coverage", lambda s=s, d=db_split: inspect_coverage(records[s], zip_db_ids[d]))

    if "minidev" in records and "dev" in records:
        _section(report, "minidev_vs_dev", lambda: inspect_minidev_vs_dev(
            records["minidev"], records["dev"], gold["minidev"], gold["dev"]))
        if zips:
            _section(report, "descriptions_dev_zip_vs_minidev", lambda: inspect_descriptions(
                paths.db_zip("dev"), paths.split_file("minidev", "descriptions_dir")))
    if "dev" in records:
        _section(report, "dev_tied_append", lambda: inspect_tied(
            read_questions(paths.split_file("dev", "tied_gold")), records["dev"], gold["dev"]))
    return report
