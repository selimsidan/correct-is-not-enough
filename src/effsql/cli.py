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
