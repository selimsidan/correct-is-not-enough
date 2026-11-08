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
        only_predicted=args.only_predicted,
    )
    print(json.dumps(m, indent=2, default=str))


def cmd_export_official(args: argparse.Namespace) -> None:
    from pathlib import Path

    from effsql.export import export_official

    info = export_official(args.pred, args.split, Path(args.out), limit=args.limit)
    print(json.dumps(info, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    from effsql.config import load_config
    from effsql.gen.backend import make_backend
    from effsql.gen.sample import generate

    cfg = load_config(args.config)
    stats = generate(args.split, args.run, make_backend(cfg), cfg=cfg, limit=args.limit,
                     db_ids=_csv_list(args.db_ids), shard=args.shard, analysis=args.analysis)
    print(json.dumps(stats, indent=2, default=str))


def cmd_parse(args: argparse.Namespace) -> None:
    from effsql.gen.parse import parse_run

    print(json.dumps(parse_run(args.split, args.run), indent=2))


def cmd_screen(args: argparse.Namespace) -> None:
    from effsql.pool.screen import screen

    print(json.dumps(screen(args.split, args.run, workers=args.workers), indent=2))


def cmd_diversity(args: argparse.Namespace) -> None:
    from effsql.pool.diversity import diversity

    print(json.dumps(diversity(args.split, args.run, args.round), indent=2))


def cmd_rewrite(args: argparse.Namespace) -> None:
    from effsql.config import load_config
    from effsql.gen.backend import make_backend
    from effsql.gen.rewrite import rewrite

    gen_cfg = load_config(args.gen_config)
    stats = rewrite(args.run, make_backend(gen_cfg), round_=args.round, cfg=load_config(args.config),
                    gen_cfg=gen_cfg, limit=args.limit)
    print(json.dumps(stats, indent=2, default=str))


def cmd_build_pool(args: argparse.Namespace) -> None:
    from effsql.pool.build import build_pool

    df = build_pool(args.split, args.run)
    print(f"Pool: {len(df)} candidates, {df['qid'].nunique()} questions, {int(df['correct'].sum())} correct")


def cmd_verify_pool(args: argparse.Namespace) -> None:
    from effsql.pool.verify import verify_pool

    print(json.dumps(verify_pool(args.split, args.run, sample=args.sample), indent=2, default=str))


def cmd_summary(args: argparse.Namespace) -> None:
    from effsql.pool.summary import summarize

    print(json.dumps(summarize(args.split, args.run), indent=2, default=str))


def cmd_predict(args: argparse.Namespace) -> None:
    from effsql.config import load_config
    from effsql.gen.backend import make_backend
    from effsql.gen.predict import predict

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["id"] = args.model
    stats = predict(args.split, args.prompt, args.run, make_backend(cfg), cfg=cfg, limit=args.limit,
                    adapter=args.adapter)
    print(json.dumps(stats, indent=2, default=str))


def cmd_render_prompt(args: argparse.Namespace) -> None:
    from effsql.data.bird import load_examples
    from effsql.data.schema import load_schema
    from effsql.gen.prompts import build_question_prompt, estimate_tokens, load_templates

    from effsql.config import load_config

    cfg = load_config("generate.yaml")
    ex = {e.qid: e for e in load_examples(args.split)}[args.qid]
    msgs, level, n = build_question_prompt(load_templates(), args.prompt, ex, load_schema(args.split, ex.db_id),
                                           cfg["prompt"]["max_prompt_tokens"], estimate_tokens, tag=args.tag)
    for m in msgs:
        print(f"===== {m['role']} =====\n{m['content']}\n")
    print(f"(shrink level: {level}, about {n} tokens by character estimate)")


def cmd_train_export(args: argparse.Namespace) -> None:
    from effsql.config import load_config
    from effsql.train.data import train_export

    meta = train_export(args.pool_run, args.run, cfg=load_config(args.config),
                        arms=args.arms.split(",") if args.arms else None, limit=args.limit, beta=args.beta)
    print(json.dumps({k: meta[k] for k in ("n_questions", "holdout_db_ids", "inference_tag", "stats")},
                     indent=2, default=str))


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
    p.add_argument("--only-predicted", action="store_true",
                   help="score only questions that have a prediction (smoke runs; not for reported numbers)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export-official", help="write predictions in the official mini-dev format")
    p.add_argument("--pred", required=True)
    p.add_argument("--split", required=True, choices=("dev", "minidev"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_export_official)

    p = sub.add_parser("generate", help="[GPU] candidate SQL for train questions (greedy + samples per prompt)")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.add_argument("--config", default="generate.yaml")
    p.add_argument("--limit", type=int, help="N questions, round-robin across databases (smoke runs)")
    p.add_argument("--db-ids")
    p.add_argument("--shard")
    p.add_argument("--analysis", action="store_true", help="allow dev/minidev, marked analysis-only")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("parse", help="extract SQL, canonicalize, dedup per question (adds gold on train)")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("screen", help="execute candidates: correctness, steps, plan")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("diversity", help="are correct candidates at different efficiency levels?")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.add_argument("--round", type=int, default=0, help="0 before rewriting, r after rewrite round r")
    p.set_defaults(func=cmd_diversity)

    p = sub.add_parser("rewrite", help="[GPU] equivalent rewrites of gold for flagged train questions")
    p.add_argument("--run", required=True)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--config", default="rewrite.yaml")
    p.add_argument("--gen-config", default="generate.yaml")
    p.add_argument("--limit", type=int, help="first N flagged questions")
    p.set_defaults(func=cmd_rewrite)

    p = sub.add_parser("build-pool", help="one Parquet/JSONL row per candidate with eff_score")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_build_pool)

    p = sub.add_parser("verify-pool", help="[CPU runtime] step ordering vs real timing on a sample")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.add_argument("--sample", type=int, help="questions to time (default from pool.yaml)")
    p.set_defaults(func=cmd_verify_pool)

    p = sub.add_parser("summary", help="report for a pipeline run")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("predict", help="[GPU] greedy predictions for dev/minidev")
    p.add_argument("--split", required=True, choices=("dev", "minidev"))
    p.add_argument("--prompt", default="base", choices=("base", "efficient"))
    p.add_argument("--run", required=True)
    p.add_argument("--model", help="override the model id in generate.yaml")
    p.add_argument("--adapter", help="LoRA adapter path (training phase)")
    p.add_argument("--config", default="generate.yaml")
    p.add_argument("--limit", type=int, help="N questions, round-robin across databases (smoke runs)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("render-prompt", help="print the prompt for one question (CPU, no tokenizer)")
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--qid", required=True)
    p.add_argument("--prompt", default="base", choices=("base", "efficient", "tagged"))
    p.add_argument("--tag", help="efficiency tag text for --prompt tagged, e.g. +1.5")
    p.set_defaults(func=cmd_render_prompt)

    p = sub.add_parser("train-export", help="training data for the SFT arms and DPO from a train pool (CPU)")
    p.add_argument("--pool-run", required=True, help="run holding pool_train.parquet")
    p.add_argument("--run", required=True, help="training run name (outputs in runs/<run>/)")
    p.add_argument("--arms", help="comma list of plain,weighted,tagged,dpo (default: all)")
    p.add_argument("--beta", type=float, help="override export.weighted.beta (ablations)")
    p.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    p.add_argument("--config", default="train.yaml")
    p.set_defaults(func=cmd_train_export)
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
