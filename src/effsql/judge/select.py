"""`python -m effsql.judge select`: choose delta on validation, write the selected answers, report (CPU).

Rule per question (no execution): best = the highest p among its candidates; shortlist = candidates with
p >= best - delta; answer = the shortlisted candidate with the lowest c (ties: higher p).

Delta is chosen on validation only, by the project's usual rule: among the deltas whose validation EX is
not significantly below the best delta's (exact McNemar test at `alpha`), the one with the lowest cost
(geometric mean of rel_cost over its correct answers); ties go to the smaller delta. Dev and mini-dev are
then answered with that delta, and their sweeps are reported for the analysis only.

Methods written as prediction runs (predictions_<split>.jsonl and adapter_meta.json, so `eval-val`,
`effsql evaluate` and `report` read them like any model):
  <out>        the judge with the chosen delta (correctness first, then predicted cost)
  <out>_p      delta 0: the most likely correct candidate, cost ignored (a correctness-only judge)
  <out>_vote   the candidate written most often (text majority, no judge, no execution)
The greedy answer is the generator's own run. selection.json / selection.txt hold the delta sweep, label-
based numbers per method (including the oracle: some candidate is correct) and the judge's own metrics.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from effsql.config import load_config
from effsql.io import write_json
from effsql.judge.candidates import CORRECT, SPLITS, read_candidates, read_labels
from effsql.judge.model import judge_metrics
from effsql.judge.score import read_scores
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)


def greedy_index(cands: list[dict[str, Any]]) -> int | None:
    for i, c in enumerate(cands):
        if "greedy" in c["sources"]:
            return i
    return 0 if cands else None


def vote_index(cands: list[dict[str, Any]]) -> int | None:
    if not cands:
        return None
    top = max(c["n"] for c in cands)
    g = greedy_index(cands)
    if g is not None and cands[g]["n"] == top:
        return g
    return next(i for i, c in enumerate(cands) if c["n"] == top)


def judge_index(cands: list[dict[str, Any]], score: dict[str, Any] | None, delta: float) -> int | None:
    if not cands or not score or len(score["p"]) != len(cands):
        return greedy_index(cands)
    p, c = score["p"], score["c"]
    best = max(p)
    short = [i for i in range(len(p)) if p[i] >= best - delta - 1e-12]
    return min(short, key=lambda i: (c[i], -p[i]))


def geomean(x: list[float]) -> float | None:
    return float(math.exp(np.mean(np.log(x)))) if x else None


def outcome(choice: dict[str, int | None], labels: dict[str, dict[str, Any]], qids: list[str],
            ref: dict[str, int | None] | None = None) -> dict[str, Any]:
    """Label-based EX and cost of one choice per question (ref: paired cheaper/costlier counts)."""
    ok, rels, cheaper, costlier = [], [], 0, 0
    for q in qids:
        i = choice[q]
        lab = labels[q]["labels"][i] if i is not None else {"correct": False, "rel_cost": None}
        ok.append(bool(lab["correct"]))
        if lab["correct"] and lab["rel_cost"] is not None:
            rels.append(lab["rel_cost"])
            if ref is not None and ref[q] is not None:
                r = labels[q]["labels"][ref[q]]
                if r["correct"] and r["rel_cost"] is not None:
                    cheaper += lab["rel_cost"] <= 0.9 * r["rel_cost"]
                    costlier += lab["rel_cost"] >= 1.1 * r["rel_cost"]
    return {"ex": 100 * float(np.mean(ok)) if ok else None, "cost_geomean": geomean(rels),
            "pct_2x_cheaper": 100 * float(np.mean(np.array(rels) <= 0.5)) if rels else None,
            "cheaper_vs_greedy": cheaper, "costlier_vs_greedy": costlier, "correct": ok}


def oracle(labels: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    any_ok, cheapest = [], []
    for q in qids:
        ok = [x for x in labels[q]["labels"] if x["correct"]]
        any_ok.append(bool(ok))
        rc = [x["rel_cost"] for x in ok if x["rel_cost"] is not None]
        if rc:
            cheapest.append(min(rc))
    return {"ex": 100 * float(np.mean(any_ok)) if any_ok else None, "cheapest_cost_geomean": geomean(cheapest)}


def metric_rows(cands: list[dict[str, Any]], scores: dict[str, Any], labels: dict[str, Any], smooth: float
                ) -> list[dict[str, Any]]:
    """The judge's p and c against the labels, in the form judge_metrics reads (labels as in training)."""
    rows = []
    for r in cands:
        q = r["qid"]
        if q not in scores or q not in labels or len(scores[q]["p"]) != len(r["cands"]):
            continue
        labs = labels[q]["labels"]
        y = [int(x["status"] == CORRECT) for x in labs]
        v = {i: math.log2(x["steps"] + smooth) for i, x in enumerate(labs) if x["status"] == CORRECT and x["steps"] is not None}
        m = float(np.mean(list(v.values()))) if len(v) >= 2 else None
        costs = [v[i] - m if m is not None and i in v else None for i in range(len(labs))]
        rows.append({"lp": scores[q]["lp"], "c": scores[q]["c"], "labels": y, "costs": costs})
    return rows


def choose_delta(sweep: dict[float, dict[str, Any]], alpha: float) -> float:
    from effsql.train.select import mcnemar_p

    best = max(sweep, key=lambda d: (sweep[d]["ex"], -d))
    allowed = [d for d in sweep if sweep[d]["ex"] >= sweep[best]["ex"]
               or mcnemar_p(sweep[best]["correct"], sweep[d]["correct"]) >= alpha]
    return min(allowed, key=lambda d: (sweep[d]["cost_geomean"] if sweep[d]["cost_geomean"] is not None else 9e9, d))


def write_predictions(paths: Paths, run: str, split: str, method: str, cands_by_q: dict[str, list[dict[str, Any]]],
                      choice: dict[str, int | None], info: dict[str, Any]) -> None:
    d = paths.run_dir(run)
    lines = [json.dumps({"qid": q, "sql": cands_by_q[q][i]["sql"] if i is not None else "", "method": method})
             for q, i in choice.items()]
    (d / f"predictions_{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(d / "adapter_meta.json", {"arm": "best", "epoch": 0, "tag": None, "method": method, "complete": True,
                                         **info})


def select(cand_run: str, judge_run: str, out: str, *, paths: Paths | None = None,
           jcfg: dict[str, Any] | None = None, splits: tuple[str, ...] = SPLITS) -> dict[str, Any]:
    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    sc = jcfg["select"]
    deltas = [float(d) for d in sc["deltas"]]
    res: dict[str, Any] = {"cand_run": cand_run, "judge_run": judge_run, "splits": {}}
    data: dict[str, Any] = {}
    for split in splits:
        cands = read_candidates(paths, cand_run, split)
        data[split] = (cands, read_scores(paths, cand_run, judge_run, split), read_labels(paths, cand_run, split))
    if data.get("val", (None, None, None))[2] is None:
        raise RuntimeError("validation candidates have no labels: run `python -m effsql.judge label --split val` first")

    sweeps: dict[str, dict[float, dict[str, Any]]] = {}
    for split, (cands, scores, labels) in data.items():
        qids = [r["qid"] for r in cands]
        by_q = {r["qid"]: r["cands"] for r in cands}
        greedy = {q: greedy_index(by_q[q]) for q in qids}
        s: dict[str, Any] = {"n_questions": len(qids), "n_scored": sum(q in scores for q in qids),
                             "candidates_per_question": float(np.mean([len(by_q[q]) for q in qids]))}
        if labels is not None and all(q in labels for q in qids):
            sweeps[split] = {d: outcome({q: judge_index(by_q[q], scores.get(q), d) for q in qids}, labels, qids, greedy)
                             for d in deltas}
            s["greedy"] = outcome(greedy, labels, qids)
            s["vote"] = outcome({q: vote_index(by_q[q]) for q in qids}, labels, qids, greedy)
            s["oracle"] = oracle(labels, qids)
            s["judge_metrics"] = judge_metrics(metric_rows(cands, scores, labels, jcfg["data"]["smooth_steps"]))
        res["splits"][split] = s
    chosen = choose_delta(sweeps["val"], sc["alpha"])
    res["delta"] = chosen
    info = {"cand_run": cand_run, "judge_run": judge_run, "delta": chosen,
            "export_run": jcfg["data"]["export_run"], "generator": jcfg["candidates"]["generator"]}
    for split, (cands, scores, _) in data.items():
        by_q = {r["qid"]: r["cands"] for r in cands}
        picks = {(out, "judge"): {q: judge_index(c, scores.get(q), chosen) for q, c in by_q.items()},
                 (f"{out}_p", "judge-p"): {q: judge_index(c, scores.get(q), 0.0) for q, c in by_q.items()},
                 (f"{out}_vote", "vote"): {q: vote_index(c) for q, c in by_q.items()}}
        for (run, method), choice in picks.items():
            write_predictions(paths, run, split, method, by_q, choice, info)
        if split in sweeps:
            res["splits"][split]["sweep"] = {str(d): {k: v for k, v in x.items() if k != "correct"}
                                             for d, x in sweeps[split].items()}
            for k in ("greedy", "vote"):
                res["splits"][split][k].pop("correct", None)
    text = format_selection(res)
    write_json(paths.run_dir(out) / "selection.json", res)
    (paths.run_dir(out) / "selection.txt").write_text(text + "\n", encoding="utf-8")
    return res


def _f(x: Any, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def format_selection(res: dict[str, Any]) -> str:
    lines = [f"delta chosen on validation: {res['delta']}", ""]
    for split, s in res["splits"].items():
        lines.append(f"== {split} ({s['n_questions']} questions, {s['candidates_per_question']:.1f} distinct candidates "
                     f"each, {s['n_scored']} scored){'' if split == 'val' else '  [analysis only]'} ==")
        if "sweep" not in s:
            lines.append("  no labels (run `python -m effsql.judge label`)")
            continue
        lines.append(f"  {'method':<18}{'EX':>7}{'cost':>8}{'2x cheaper':>12}{'cheaper':>9}{'costlier':>9}")
        rows = [("greedy", s["greedy"]), ("text majority", s["vote"])]
        rows += [(f"judge d={d}", x) for d, x in s["sweep"].items()]
        for name, x in rows:
            lines.append(f"  {name:<18}{_f(x['ex'], '7.1f')}{_f(x['cost_geomean'], '8.3f')}"
                         f"{_f(x['pct_2x_cheaper'], '11.1f')}%{x.get('cheaper_vs_greedy', 0):>9}{x.get('costlier_vs_greedy', 0):>9}")
        o = s["oracle"]
        lines.append(f"  {'oracle':<18}{_f(o['ex'], '7.1f')}{_f(o['cheapest_cost_geomean'], '8.3f')}   "
                     "(some candidate correct; cost: cheapest correct)")
        m = s["judge_metrics"]
        lines.append(f"  judge: AUC {_f(m['auc'], '.3f')}, accuracy {_f(m['accuracy'], '.3f')}, top-1 "
                     f"{_f(m['top1'], '.3f')} (of {m['n_top1']} questions with a correct candidate), cost order "
                     f"{_f(m['cost_pair_acc'], '.3f')} over {m['n_cost_pairs']} pairs")
        lines.append("")
    lines.append("EX and cost from executing the candidates (labels); cost: geometric mean of rel_cost over correct "
                 "answers. cheaper/costlier: both correct, at least 10% apart, against greedy.")
    return "\n".join(lines)
