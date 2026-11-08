"""Command line entry point: `effsql <command> ...`.

CPU commands must not import vLLM or torch; GPU commands (stage 3) import them lazily.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from effsql.paths import SPLITS, load_paths

log = logging.getLogger("effsql")


def _csv_list(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _add_db_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db-ids", help="comma separated db_ids (default: all)")
    p.add_argument("--shard", help="i/n: process the i-th of n round-robin shards of the sorted db_ids")
    p.add_argument("--limit", type=int, help="process at most N databases (smoke tests)")


def cmd_inspect(args: argparse.Namespace) -> None:
    from effsql.data.inspect import inspect_all
    from effsql.io import write_json

    paths = load_paths()
    report = inspect_all(paths, splits=_csv_list(args.splits), zips=not args.skip_zips)
    for section, content in report.items():
        print(f"\n=== {section} ===")
        print(json.dumps(content, indent=2, ensure_ascii=False, default=str))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = paths.cache_dir / "inspect" / f"inspect_{stamp}.json"
    write_json(out, report)
    print(f"\nSaved report to {out}")


def cmd_stage(args: argparse.Namespace) -> None:
    from effsql.data.stage import stage

    manifest = stage(args.split, db_ids=_csv_list(args.db_ids), shard=args.shard, limit=args.limit)
    print(f"Staged databases for {args.split}: {len(manifest['dbs'])} in manifest")


def cmd_schema_cache(args: argparse.Namespace) -> None:
    from effsql.data.schema import build_schema_cache

    written = build_schema_cache(
        args.split, db_ids=_csv_list(args.db_ids), shard=args.shard, limit=args.limit, overwrite=args.overwrite
    )
    print(f"Wrote {len(written)} schema cache file(s)")


def cmd_gold_cache(args: argparse.Namespace) -> None:
    from collections import Counter

    from effsql.config import load_config
    from effsql.data.bird import load_examples, load_tied_gold
    from effsql.measure.gold import ensure_gold_costs, golds_for

    paths = load_paths()
    exec_cfg = load_config("evaluate.yaml")["exec"]
    examples = load_examples(args.split, paths)
    if args.db_ids:
        examples = [e for e in examples if e.db_id in set(_csv_list(args.db_ids))]
    examples = examples[: args.limit] if args.limit else examples
    tied = load_tied_gold(paths) if args.split == "dev" else None
    recs = ensure_gold_costs(args.split, golds_for(examples, tied), paths, exec_cfg, args.workers)
    print(f"Gold costs for {len(recs)} entries: {dict(Counter(r['gold_status'] for r in recs.values()))}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from effsql.evaluate import evaluate

    m = evaluate(
        args.pred, args.split, args.run, timing=args.timing, repeat=args.repeat, limit=args.limit,
        workers=args.workers, model=args.model, method=args.method, overwrite=args.overwrite,
    )
    print(json.dumps(m, indent=2, default=str))


def cmd_export_official(args: argparse.Namespace) -> None:
    from pathlib import Path

    from effsql.export import export_official

    info = export_official(args.pred, args.split, Path(args.out), limit=args.limit)
    print(json.dumps(info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="effsql", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="print the real BIRD layout, keys and consistency checks")
    p.add_argument("--splits", help=f"comma separated subset of {','.join(SPLITS)}")
    p.add_argument("--skip-zips", action="store_true", help="skip zip listing (faster)")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("stage", help="extract databases from Drive zips to the local disk")
    p.add_argument("--split", required=True, choices=SPLITS)
    _add_db_selection(p)
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("schema-cache", help="build per-database schema JSONs from staged databases")
    p.add_argument("--split", required=True, choices=SPLITS)
    _add_db_selection(p)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_schema_cache)

    p = sub.add_parser("gold-cache", help="execute gold SQL once per question: result hash, vm_steps, plan")
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--db-ids", help="comma separated db_ids (default: all)")
    p.add_argument("--limit", type=int, help="first N questions only")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_gold_cache)

    p = sub.add_parser("evaluate", help="EX, R-VES, VES and cost metrics for a predictions file")
    p.add_argument("--pred", required=True, help="predictions JSONL, official JSON, or `gold` for a sanity check")
    p.add_argument("--split", required=True, choices=("dev", "minidev"))
    p.add_argument("--run", required=True, help="run name (outputs in runs/<run>/)")
    p.add_argument("--timing", default="none", choices=("none", "quick", "official"))
    p.add_argument("--repeat", type=int, default=1, help="repeat the timing pass R times (mean and std)")
    p.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    p.add_argument("--workers", type=int, help="processes for the correctness pass (never for timing)")
    p.add_argument("--model", help="model id, recorded in the ledger")
    p.add_argument("--method", help="method name, recorded in the ledger")
    p.add_argument("--overwrite", action="store_true", help="discard this run's previous eval files")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export-official", help="write predictions in the official mini-dev format")
    p.add_argument("--pred", required=True)
    p.add_argument("--split", required=True, choices=("dev", "minidev"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_export_official)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args.func(args)


if __name__ == "__main__":
    main()
