"""`python -m effsql.train <command>`: the SFT phase.

  export    training data for every arm from a quality-checked pool (CPU, minutes)
  pilot     a few training steps on the current GPU: speed, memory, projected hours and units
  sft       train one arm for one epoch (resumable: rerun the same command after a disconnect)
  predict   greedy predictions of a trained adapter for val, dev or minidev (GPU, vLLM)
  eval-val  score the validation predictions (CPU: correctness and step counts, no timing)
  select    the method selection rule over several trained runs
  winner    print the selected run and arm
"""

from __future__ import annotations

import argparse
import json
import logging

from effsql.config import load_config
from effsql.io import read_json
from effsql.paths import load_paths


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def cmd_export(a: argparse.Namespace) -> None:
    from effsql.train.data import train_export

    m = train_export(a.pool_run, a.run, arms=_csv(a.arms), limit=a.limit)
    print(json.dumps({k: m[k] for k in ("n_questions", "holdout_db_ids", "quality_filter", "inference_tag")}, indent=2))
    for k, s in m["stats"].items():
        print(k, {x: s.get(x) for x in ("n_examples", "n_questions", "by_type", "share_gold")})


def cmd_pilot(a: argparse.Namespace) -> None:
    from effsql.train.sft import train_sft

    rep = train_sft(a.arm, a.export, f"{a.export}_pilot", max_steps=a.steps, pilot=True)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("train_examples", "val_examples")}, indent=2, default=str))


def cmd_sft(a: argparse.Namespace) -> None:
    from effsql.train.sft import train_sft

    meta = train_sft(a.arm, a.export, a.run, epoch=a.epoch, init_from=a.init_from, max_steps=a.max_steps)
    print(json.dumps({k: meta[k] for k in ("arm", "epoch", "n_steps", "complete", "val_loss_final",
                                           "tokens_per_s_this_session")}, indent=2, default=str))


def cmd_predict(a: argparse.Namespace) -> None:
    from effsql.train.predict_sft import LoRAVLLMBackend, predict_sft

    paths = load_paths()
    gen_cfg = load_config("generate.yaml")
    meta = read_json(paths.run_dir(a.run, create=False) / "adapter_meta.json")
    if not meta.get("complete") and not a.allow_incomplete:
        raise RuntimeError(f"{a.run} has not finished training (adapter_meta.json: complete is false)")
    from effsql.train.sft import base_model_cfg

    model_cfg = base_model_cfg(load_config("train.yaml"), gen_cfg)
    if meta.get("base_revision") != model_cfg.get("revision"):
        raise RuntimeError(f"adapter trained on base revision {meta.get('base_revision')!r}, config has "
                           f"{model_cfg.get('revision')!r}")
    backend = LoRAVLLMBackend(model_cfg, paths.run_dir(a.run, create=False) / "adapter", seed=gen_cfg.get("seed", 0))
    for split in _csv(a.splits):
        print(json.dumps(predict_sft(meta["arm"], split, a.run, meta["export_run"], backend, gen_cfg=gen_cfg,
                                     tag=a.tag, limit=a.limit), indent=2, default=str))


def cmd_eval_val(a: argparse.Namespace) -> None:
    from effsql.evaluate import evaluate

    paths = load_paths()
    meta = read_json(paths.run_dir(a.run, create=False) / "adapter_meta.json")
    pred = paths.run_dir(a.run, create=False) / "predictions_val.jsonl"
    m = evaluate(str(pred), "train", f"{a.run}_val", only_predicted=True, model=load_config("generate.yaml")["model"]["id"],
                 method=f"sft-{meta['arm']}-e{meta['epoch'] + 1}")
    print(json.dumps({k: m["overall"].get(k) for k in ("n", "ex", "rel_cost_median", "eff_acc@1.0", "outcomes")}, indent=2))


def cmd_select(a: argparse.Namespace) -> None:
    from effsql.train.select import format_table, select_method

    paths = load_paths()
    res = select_method({r: paths.run_dir(f"{r}_val", create=False) for r in _csv(a.runs)},
                        out=paths.run_dir(a.out) / "selection.json")
    print(format_table(res))


def cmd_winner(a: argparse.Namespace) -> None:
    paths = load_paths()
    run = read_json(paths.run_dir(a.selection, create=False) / "selection.json")["winner"]
    arm = read_json(paths.run_dir(run, create=False) / "adapter_meta.json")["arm"]
    print(json.dumps({"run": run, "arm": arm}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m effsql.train", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--pool-run", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--arms")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("pilot")
    p.add_argument("--export", required=True)
    p.add_argument("--arm", default="weighted")
    p.add_argument("--steps", type=int, default=6)
    p.set_defaults(func=cmd_pilot)
    p = sub.add_parser("sft")
    p.add_argument("--export", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", type=int, default=0, help="which exported epoch to train (0 first, 1 second)")
    p.add_argument("--init-from", help="finished run whose adapter starts this one (second epoch)")
    p.add_argument("--max-steps", type=int, help="stop early (smoke tests only)")
    p.set_defaults(func=cmd_sft)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--tag", help="override the tagged arm's test-time tag, e.g. +1.0")
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-incomplete", action="store_true", help="smoke tests: adapter of a --max-steps run")
    p.set_defaults(func=cmd_predict)
    p = sub.add_parser("eval-val")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_eval_val)
    p = sub.add_parser("select")
    p.add_argument("--runs", required=True, help="comma list of trained runs")
    p.add_argument("--out", required=True, help="run name for selection.json")
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("winner")
    p.add_argument("--selection", required=True)
    p.set_defaults(func=cmd_winner)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
