"""`python -m effsql.judge <command>`: the cost-aware judge (configs/judge.yaml).

  data        the judge's training data from executed own answers and the pool (CPU, minutes)
  derive-data a data run keeping only some sources of another, with its results (ablation; CPU, seconds)
  untrained   a judge with no judge training: the model asked the judge's question as it is (ablation; GPU)
  exec-data   run the training data's candidates with the answer-time cap, for a judge that reads results
  train       train the judge's adapter and cost head (A100, resumable; --with-results: prompts show results)
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


def _jcfg(a: argparse.Namespace) -> dict:
    """judge.yaml with the command line's overrides (used by the ablations)."""
    jcfg = load_config("judge.yaml")
    if getattr(a, "generator", None):
        jcfg["candidates"]["generator"] = a.generator
    if getattr(a, "init_from", None):
        jcfg["train"]["init_from"] = None if a.init_from == "none" else a.init_from
    if getattr(a, "cost_weight", None) is not None:
        jcfg["train"]["cost_weight"] = a.cost_weight
    return jcfg


def cmd_data(a: argparse.Namespace) -> None:
    from effsql.judge.data import build_judge_data

    print(json.dumps(build_judge_data(a.run)["stats"], indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    from effsql.judge.train import train_judge

    m = train_judge(a.run, a.data, jcfg=_jcfg(a), max_steps=a.max_steps, with_results=a.with_results)
    print(json.dumps({k: m[k] for k in ("n_steps", "complete", "val_final", "train_questions")}, indent=2, default=str))


def cmd_candidates(a: argparse.Namespace) -> None:
    from effsql.judge.candidates import generate_candidates
    from effsql.train.predict_sft import LoRAVLLMBackend
    from effsql.train.sft import base_model_cfg

    paths = load_paths()
    jcfg, gen_cfg = _jcfg(a), load_config("generate.yaml")
    todo = [s for s in _splits(a.splits) if a.limit or not (paths.run_dir(a.run) / f"candidates_{s}.jsonl").exists()]
    if not todo:
        print(f"candidates of {a.run} are already made for {a.splits}; nothing to do")
        return
    model_cfg = base_model_cfg(load_config("train.yaml"), gen_cfg)
    adapter = paths.run_dir(jcfg["candidates"]["generator"], create=False) / "adapter"
    backend = LoRAVLLMBackend(model_cfg, adapter, seed=gen_cfg.get("seed", 0))
    for split in todo:
        print(json.dumps(generate_candidates(a.run, split, backend, jcfg=jcfg, gen_cfg=gen_cfg, limit=a.limit),
                         indent=2, default=str))


def cmd_score(a: argparse.Namespace) -> None:
    import torch

    from effsql.judge.score import done_path, score_split
    from effsql.judge.train import load_judge

    paths = load_paths()
    todo = [s for s in _splits(a.splits) if not done_path(paths, a.run, a.judge, s).exists()]
    if not todo:
        print(f"{a.judge} has already scored {a.run} for {a.splits}; nothing to do")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, tok, yes_no, meta = load_judge(paths, a.judge, device)
    for split in todo:
        print(json.dumps(score_split(a.judge, a.run, split, (model, head, tok, yes_no), device=device,
                                     with_results=bool(meta.get("with_results"))), indent=2))


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


def cmd_exec_data(a: argparse.Namespace) -> None:
    from effsql.judge.vote import exec_judge_data

    print(json.dumps(exec_judge_data(a.run, workers=a.workers), indent=2))


def cmd_vote(a: argparse.Namespace) -> None:
    from effsql.judge.vote import format_vote, vote

    print(format_vote(vote(a.run, a.judge, a.out, jcfg=_jcfg(a), splits=tuple(_splits(a.splits)))))


def cmd_derive_data(a: argparse.Namespace) -> None:
    from effsql.judge.data import derive_judge_data

    print(json.dumps(derive_judge_data(a.src, a.run, tuple(_splits(a.sources)))["stats"], indent=2))


def cmd_untrained(a: argparse.Namespace) -> None:
    from effsql.judge.train import untrained_judge

    init = None if a.init_from in (None, "none") else a.init_from
    print(json.dumps(untrained_judge(a.run, init_from=init, with_results=not a.sql_only), indent=2, default=str))


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
    p.add_argument("--with-results", action="store_true", help="prompts show each candidate's result (exec-data first)")
    p.add_argument("--init-from", help="adapter to start from (a run name, or none for the plain base model); "
                                       "default: judge.yaml train.init_from")
    p.add_argument("--cost-weight", type=float, help="weight of the cost loss (0: none); default: judge.yaml")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("derive-data")
    p.add_argument("--src", required=True, help="judge data run with exec_train.jsonl and exec_val.jsonl")
    p.add_argument("--run", required=True, help="new judge data run")
    p.add_argument("--sources", default="own", help="candidate sources to keep (own, pool)")
    p.set_defaults(func=cmd_derive_data)
    p = sub.add_parser("untrained")
    p.add_argument("--run", required=True)
    p.add_argument("--init-from", help="adapter to copy (a run name); default: none, the plain base model")
    p.add_argument("--sql-only", action="store_true", help="prompts without the results")
    p.set_defaults(func=cmd_untrained)
    p = sub.add_parser("candidates")
    p.add_argument("--run", required=True, help="output run for the candidates (and later scores and labels)")
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--limit", type=int)
    p.add_argument("--generator", help="generator run (default: judge.yaml candidates.generator)")
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
    p = sub.add_parser("exec-data")
    p.add_argument("--run", required=True, help="judge data run (judge_train.jsonl, judge_val.jsonl)")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_exec_data)
    p = sub.add_parser("vote")
    p.add_argument("--run", required=True, help="candidates run (with scores, labels and exec results)")
    p.add_argument("--judge", required=True)
    p.add_argument("--out", required=True, help="run name of the kept method's answers (also <out>_<method>)")
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--generator", help="generator of the candidates, recorded with the answers")
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
