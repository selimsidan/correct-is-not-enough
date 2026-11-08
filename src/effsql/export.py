"""`effsql export-official`: write predictions in the official mini-dev input format, so the real
BIRD scripts (bird-bench/mini_dev evaluation/evaluation_ex.py and evaluation_ves.py) can be run on
exactly the same predictions as our port, as a cross-check.

Files written to --out:
  predict.json  {"0": "<sql>\\t----- bird -----\\t<db_id>", ...} in split order
  gold.sql      "<gold sql>\\t<db_id>" per line, same order
  diff.jsonl    one {"difficulty": ...} object per line, same order (the official --diff_json_path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from effsql.data.bird import load_examples
from effsql.evaluate import OFFICIAL_SEP, load_predictions
from effsql.io import write_json
from effsql.paths import Paths, load_paths

MINI_DEV_COMMIT = "1c49a375e76c471e218564da3f4617513d33e371"


def export_official(pred: str, split: str, out_dir: Path, limit: int | None = None,
                    paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    examples = load_examples(split, paths)
    examples = examples[:limit] if limit else examples
    preds = load_predictions(pred, examples)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "predict.json",
               {str(i): f"{preds.get(e.qid, '')}{OFFICIAL_SEP}{e.db_id}" for i, e in enumerate(examples)})
    (out_dir / "gold.sql").write_text("".join(f"{e.gold_sql}\t{e.db_id}\n" for e in examples), encoding="utf-8")
    (out_dir / "diff.jsonl").write_text(
        "".join(json.dumps({"question_id": e.bird_question_id, "db_id": e.db_id, "difficulty": e.difficulty}) + "\n"
                for e in examples),
        encoding="utf-8",
    )
    db_root = str(paths.local_split_dir(split)) + "/"
    args = (f"--db_root_path {db_root} --predicted_sql_path {out_dir}/predict.json "
            f"--ground_truth_path {out_dir}/gold.sql --diff_json_path {out_dir}/diff.jsonl "
            f"--num_cpus 1 --meta_time_out 30.0 --sql_dialect SQLite --output_log_path {out_dir}/official_log.txt")
    return {
        "n": len(examples),
        "out_dir": str(out_dir),
        "ex_command": f"python3 -u evaluation_ex.py {args}",
        "ves_command": f"python3 -u evaluation_ves.py {args}",
        "mini_dev_commit": MINI_DEV_COMMIT,
    }
