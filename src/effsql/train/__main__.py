"""`python -m effsql.train <command>`: the SFT phase.

  export    training data for every arm from a quality-checked pool (CPU, minutes)
  pilot     a few training steps on the current GPU: speed, memory, projected hours and units
  sft       train one arm for one epoch (resumable: rerun the same command after a disconnect)
  dpo       DPO from a finished SFT adapter on the export's preference pairs (resumable)
  predict   greedy predictions of a trained adapter for val, dev or minidev (GPU, vLLM)
  eval-val  score the validation predictions (CPU: correctness and step counts, no timing)
  select    the method selection rule over several trained runs
  winner    print the selected run, arm and tag
  report    results table over models and splits from saved evaluations (R-VES, VES, paired tests)
  rl        reinforcement learning with an execution reward from a finished adapter (resumable; --pilot
            measures the cost first)
"""

from __future__ import annotations

import argparse
import json
import logging

from effsql.config import load_config
from effsql.io import read_json, write_json
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


def cmd_dpo(a: argparse.Namespace) -> None:
    from effsql.train.dpo import train_dpo

    cfg = load_config("train.yaml")
    for k in ("lr", "epochs", "beta", "rpo_alpha"):  # command-line settings override train.yaml dpo
        if getattr(a, k) is not None:
            cfg["dpo"][k] = getattr(a, k)
    meta = train_dpo(a.export, a.run, a.init_from, tag=a.tag, cfg=cfg, max_steps=a.max_steps)
    print(json.dumps({k: meta[k] for k in ("arm", "tag", "init_from", "lr", "epochs", "n_steps", "complete", "val_final")},
                     indent=2, default=str))


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
    adapter_run = meta.get("adapter_run", a.run)  # tag-sweep runs share their model's adapter
    targets = [(a.run, meta.get("tag"))]  # a DPO or tag-sweep run predicts with its own tag
    if a.tags:  # the tagged model asked for other efficiency targets, each into its own run
        if meta["arm"] != "tagged":
            raise ValueError(f"--tags needs a tagged model, {a.run} is {meta['arm']!r}")
        from effsql.train.data import format_tag

        targets = []
        for t in _csv(a.tags):
            tag = format_tag(float(t), 1)
            out = f"{a.run}_tag{tag}"
            write_json(paths.run_dir(out) / "adapter_meta.json", {**meta, "tag": tag, "adapter_run": adapter_run})
            targets.append((out, tag))
    backend = LoRAVLLMBackend(model_cfg, paths.run_dir(adapter_run, create=False) / "adapter", seed=gen_cfg.get("seed", 0))
    for out, tag in targets:
        for split in _csv(a.splits):
            print(json.dumps(predict_sft(meta["arm"], split, out, meta["export_run"], backend, gen_cfg=gen_cfg,
                                         tag=tag, limit=a.limit), indent=2, default=str))


def cmd_eval_val(a: argparse.Namespace) -> None:
    from effsql.evaluate import evaluate

    paths = load_paths()
    meta = read_json(paths.run_dir(a.run, create=False) / "adapter_meta.json")
    pred = paths.run_dir(a.run, create=False) / "predictions_val.jsonl"
    tag = f"-tag{meta['tag']}" if meta.get("tag") else ""
    m = evaluate(str(pred), "train", f"{a.run}_val", only_predicted=True, model=load_config("generate.yaml")["model"]["id"],
                 method=f"{meta.get('method', 'sft-' + meta['arm'])}-e{meta['epoch'] + 1}{tag}")
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
    meta = read_json(paths.run_dir(run, create=False) / "adapter_meta.json")
    print(json.dumps({"run": run, "arm": meta["arm"], "tag": meta.get("tag"),
                      "adapter_run": meta.get("adapter_run", run)}))


def cmd_report(a: argparse.Namespace) -> None:
    from effsql.train.report import build_report, format_report

    paths = load_paths()
    rep = build_report(paths.runs_dir, _csv(a.runs), ref=a.ref, splits=tuple(_csv(a.splits)))
    labels = dict(x.split("=", 1) for x in _csv(a.labels) or [])
    text = format_report(rep, labels)
    print(text)
    if a.out:
        write_json(paths.run_dir(a.out) / "report.json", rep)
        (paths.run_dir(a.out) / "report.txt").write_text(text + "\n", encoding="utf-8")


def cmd_rl(a: argparse.Namespace) -> None:
    from effsql.train.rl import format_summary, run_rl

    s = run_rl(a.run, a.init_from, pilot=a.pilot, workers=a.workers, settings=a.settings)
    print(format_summary(s))


def cmd_rl_step(a: argparse.Namespace) -> None:
    from effsql.train import rl

    if a.step == "gen":
        print(json.dumps(rl.gen_iter(a.run, a.iter), indent=2, default=str))
    else:
        print(json.dumps(rl.train_iter(a.run, a.iter), indent=2, default=str))


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
    p = sub.add_parser("dpo")
    p.add_argument("--export", required=True, help="export run holding dpo_train.jsonl and dpo_val.jsonl")
    p.add_argument("--run", required=True)
    p.add_argument("--init-from", required=True, help="finished SFT run whose adapter starts DPO (and is the reference)")
    p.add_argument("--tag", help="tagged model: the efficiency target in every prompt (e.g. +1.5)")
    p.add_argument("--lr", type=float, help="override train.yaml dpo.lr")
    p.add_argument("--epochs", type=int, help="override train.yaml dpo.epochs")
    p.add_argument("--beta", type=float, help="override train.yaml dpo.beta")
    p.add_argument("--rpo-alpha", type=float, help="override train.yaml dpo.rpo_alpha")
    p.add_argument("--max-steps", type=int, help="stop early (smoke tests only)")
    p.set_defaults(func=cmd_dpo)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--tags", "--tag", dest="tags",
                   help="tagged model only: comma list of efficiency targets, e.g. +1.5,+3.0; each is predicted "
                        "into its own run <run>_tag<tag> (default: the run's own tag if it has one, else the export's "
                        "inference tag, into --run)")
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
    p = sub.add_parser("report")
    p.add_argument("--runs", required=True, help="comma list of runs; one containing {split} is a template "
                   "(zero-shot: pool_v1_{split}_base)")
    p.add_argument("--ref", help="reference run for the paired comparisons (default: the first)")
    p.add_argument("--splits", default="val,dev,minidev")
    p.add_argument("--labels", help="comma list run=label for shorter names")
    p.add_argument("--out", help="run name to save report.json and report.txt in")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("rl")
    p.add_argument("--run", required=True)
    p.add_argument("--init-from", required=True, help="finished SFT (or DPO) run whose adapter RL starts from")
    p.add_argument("--pilot", action="store_true", help="train.yaml rl.pilot settings; projects the full run's cost")
    p.add_argument("--workers", type=int, help="processes executing answers (default: all CPUs)")
    p.add_argument("--settings", help="train.yaml section whose values override `rl` (e.g. rl_eff)")
    p.set_defaults(func=cmd_rl)
    p = sub.add_parser("rl-step", help="one GPU step of an RL iteration (started by `rl` in its own process)")
    p.add_argument("step", choices=["gen", "train"])
    p.add_argument("--run", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.set_defaults(func=cmd_rl_step)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
