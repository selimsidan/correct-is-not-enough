"""`python -m effsql.judge <command>`: the cost-aware judge (configs/judge.yaml).

  data        the judge's training data from executed own answers and the pool (CPU, minutes)
  train       train the judge's adapter and cost head (A100, resumable)
  candidates  the generator's candidates for val, dev and mini-dev (GPU, vLLM, resumable)
  score       the judge's p and c for every candidate (GPU, resumable)
  label       execute the candidates (CPU, resumable): for choosing delta on validation and the analysis
  select      choose delta on validation, write the selected answers and the selection report (CPU)
  exec        compile and run every candidate once with the answer-time cap (CPU, resumable)
  vote        answers by the judge, the compile check and result voting; method kept on validation (CPU)
  latency     the whole answer path timed question by question (A100, one engine with both adapters)
"""

from __future__ import annotations

import argparse
import json
import logging

from effsql.config import load_config
from effsql.paths import load_paths


def _splits(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def cmd_data(a: argparse.Namespace) -> None:
    from effsql.judge.data import build_judge_data

    print(json.dumps(build_judge_data(a.run)["stats"], indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    from effsql.judge.train import train_judge

    m = train_judge(a.run, a.data, max_steps=a.max_steps)
    print(json.dumps({k: m[k] for k in ("n_steps", "complete", "val_final", "train_questions")}, indent=2, default=str))


def cmd_candidates(a: argparse.Namespace) -> None:
    from effsql.judge.candidates import generate_candidates
    from effsql.train.predict_sft import LoRAVLLMBackend
    from effsql.train.sft import base_model_cfg

    paths = load_paths()
    jcfg, gen_cfg = load_config("judge.yaml"), load_config("generate.yaml")
    model_cfg = base_model_cfg(load_config("train.yaml"), gen_cfg)
    adapter = paths.run_dir(jcfg["candidates"]["generator"], create=False) / "adapter"
    backend = LoRAVLLMBackend(model_cfg, adapter, seed=gen_cfg.get("seed", 0))
    for split in _splits(a.splits):
        print(json.dumps(generate_candidates(a.run, split, backend, jcfg=jcfg, gen_cfg=gen_cfg, limit=a.limit),
                         indent=2, default=str))


def cmd_score(a: argparse.Namespace) -> None:
    import torch

    from effsql.judge.score import score_split
    from effsql.judge.train import load_judge

    paths = load_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, tok, yes_no, _ = load_judge(paths, a.judge, device)
    for split in _splits(a.splits):
        print(json.dumps(score_split(a.judge, a.run, split, (model, head, tok, yes_no), device=device), indent=2))


def cmd_label(a: argparse.Namespace) -> None:
    from effsql.judge.candidates import label_candidates

    for split in _splits(a.splits):
        print(json.dumps(label_candidates(a.run, split, workers=a.workers), indent=2))


def cmd_select(a: argparse.Namespace) -> None:
    from effsql.judge.select import format_selection, select

    print(format_selection(select(a.run, a.judge, a.out, splits=tuple(_splits(a.splits)))))


def cmd_exec(a: argparse.Namespace) -> None:
    from effsql.judge.vote import exec_candidates

    for split in _splits(a.splits):
        print(json.dumps(exec_candidates(a.run, split, workers=a.workers), indent=2))


def cmd_vote(a: argparse.Namespace) -> None:
    from effsql.judge.vote import format_vote, vote

    print(format_vote(vote(a.run, a.judge, a.out, splits=tuple(_splits(a.splits)))))


def cmd_latency(a: argparse.Namespace) -> None:
    from effsql.judge.latency import format_latency, measure_latency

    print(format_latency(measure_latency(a.judge, a.out, a.split, n=a.n)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m effsql.judge", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("train")
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True, help="run holding judge_train.jsonl and judge_val.jsonl")
    p.add_argument("--max-steps", type=int, help="stop early (smoke tests only)")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("candidates")
    p.add_argument("--run", required=True, help="output run for the candidates (and later scores and labels)")
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_candidates)
    p = sub.add_parser("score")
    p.add_argument("--run", required=True, help="candidates run")
    p.add_argument("--judge", required=True, help="finished judge run")
    p.add_argument("--splits", default="val,dev,minidev")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("label")
    p.add_argument("--run", required=True, help="candidates run")
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_label)
    p = sub.add_parser("select")
    p.add_argument("--run", required=True, help="candidates run")
    p.add_argument("--judge", required=True)
    p.add_argument("--out", required=True, help="run name of the selected answers (also <out>_p, <out>_vote)")
    p.add_argument("--splits", default="val,dev,minidev")
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("exec")
    p.add_argument("--run", required=True, help="candidates run")
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_exec)
    p = sub.add_parser("vote")
    p.add_argument("--run", required=True, help="candidates run (with scores, labels and exec results)")
    p.add_argument("--judge", required=True)
    p.add_argument("--out", required=True, help="run name of the kept method's answers (also <out>_<method>)")
    p.add_argument("--splits", default="val,dev,minidev")
    p.set_defaults(func=cmd_vote)
    p = sub.add_parser("latency")
    p.add_argument("--judge", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, help="questions (default: judge.yaml latency.questions)")
    p.set_defaults(func=cmd_latency)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
