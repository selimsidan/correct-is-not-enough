"""The results table: every model on validation, dev and mini-dev from its saved evaluation files (CPU,
no databases needed, nothing is executed).

Per model and split: EX, step-based R-VES (the main metric) with a 95% bootstrap interval, efficiency
per correct answer (R-VES over correct answers only, gold = 100), VES, answers >= 2x cheaper and >= 2x
slower than gold, geometric mean cost relative to gold. Then each model against a reference model on
the same questions: R-VES difference (Wilcoxon), EX (McNemar), cheaper / costlier when both are
correct (sign test). Dev and mini-dev are reported here, never used to choose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from effsql.evaluate import is_current
from effsql.io import latest_by
from effsql.measure.metrics import step_scores
from effsql.train.select import method_stats, paired

SPLITS = {"val": "train", "dev": "dev", "minidev": "minidev"}  # report name -> evaluated split


def eval_file(runs_dir: Path, run: str, split: str) -> Path:
    """`<run>_<split>/eval_<file split>.jsonl`; a run name containing {split} is a template
    (the zero-shot runs are pool_v1_{split}_base)."""
    d = run.format(split=split) if "{split}" in run else f"{run}_{split}"
    return runs_dir / d / f"eval_{SPLITS[split]}.jsonl"


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_report(runs_dir: Path, runs: list[str], ref: str | None = None,
                 splits: tuple[str, ...] = ("val", "dev", "minidev")) -> dict[str, Any]:
    ref = ref or runs[0]
    out: dict[str, Any] = {"ref": ref, "splits": {}}
    for split in splits:
        recs = {}
        for r in runs:
            f = eval_file(runs_dir, r, split)
            if f.exists():
                recs[r] = latest_by(f, where=is_current)
        recs = {r: x for r, x in recs.items() if x}
        if not recs:
            continue
        qids = sorted(set.intersection(*(set(x) for x in recs.values())))
        rows = {}
        for r, x in recs.items():
            s = method_stats(x, qids)
            s["rves_ci"] = bootstrap_ci(np.array([step_scores(x[q])[0] for q in qids]))
            s["vs_ref"] = paired(recs[ref], x, qids) if ref in recs and r != ref else None
            rows[r] = s
        out["splits"][split] = {"n_questions": len(qids), "models": rows}
    return out


def _f(x: float | None, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def format_report(rep: dict[str, Any], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    name = lambda r: labels.get(r, r)  # noqa: E731
    w = max(24, max(len(name(r)) for s in rep["splits"].values() for r in s["models"]) + 2)
    lines = []
    for split, s in rep["splits"].items():
        lines.append(f"== {split} ({s['n_questions']} questions) ==")
        lines.append(f"{'model':<{w}}{'EX':>6}{'R-VES':>7}{'95% CI':>15}{'R-VES/EX':>10}{'VES':>8}"
                     f"{'>=2x cheaper':>14}{'>=2x slower':>13}{'geomean cost':>14}")
        for r, m in s["models"].items():
            lo, hi = m["rves_ci"]
            lines.append(f"{name(r):<{w}}{m['ex']:>6.1f}{m['rves']:>7.1f}{f'[{lo:.1f}, {hi:.1f}]':>15}"
                         f"{_f(m['eff_per_correct'], '.1f'):>10}{m['ves']:>8.1f}{m['cheap_2x']:>13.1f}%"
                         f"{m['slower_2x']:>12.1f}%{_f(m['rel_cost_geomean'], '.3f'):>14}")
        if any(m["vs_ref"] for m in s["models"].values()):
            lines.append(f"  against {name(rep['ref'])}, same questions:")
            lines.append(f"  {'model':<{w - 2}}{'R-VES diff':>11}{'p':>8}{'EX diff':>9}{'p':>8}"
                         f"{'cheaper':>9}{'costlier':>10}{'p':>8}{'cost ratio':>12}")
            ref_ex = s["models"][rep["ref"]]["ex"]
            for r, m in s["models"].items():
                v = m["vs_ref"]
                if v:
                    lines.append(f"  {name(r):<{w - 2}}{v['rves_diff']:>+11.2f}{v['p_rves']:>8.3f}"
                                 f"{m['ex'] - ref_ex:>+9.1f}{v['p_ex']:>8.3f}{v['cheaper']:>9}{v['costlier']:>10}"
                                 f"{v['p_cost']:>8.3f}{_f(v.get('cost_ratio'), '.3f'):>12}")
        lines.append("")
    lines.append("R-VES: BIRD's reward with step counts for seconds (gold = 100, wrong = 0); R-VES/EX: the same over "
                 "correct answers only; VES: 100 * sqrt(speed-up), unbounded, so a few extreme questions dominate it. "
                 "cheaper/costlier: both correct, cost at least 10% lower / higher than the reference's; cost ratio: geometric "
                 "mean of the model's cost / the reference's on the questions both answer correctly.")
    return "\n".join(lines)
