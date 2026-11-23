"""`python -m effsql.ablate`: the ablation tables from saved files (CPU; nothing is executed, no databases).

  zero-shot-val  the base model's greedy answers on validation, taken from the pool's generations (the same
                 model, prompt and settings as the zero-shot dev and mini-dev runs), as a prediction run
                 for `python -m effsql.train eval-val`
  report         every table below, to runs/<out>/report.txt and report.json

Correctness of a chosen candidate comes from its label (`python -m effsql.judge label`), which runs it with
the evaluation's limits; it equals the evaluation's EX for every final answer (checked on all splits).
Every choice uses only what is known when answering: the candidates, how often each was written, the
verifier's scores and the answer-time execution results. p: exact McNemar test on the same questions.

Tables:
  ladder       each part of the system added in turn, each step tested against the step before (A4), plus
               the verifier and vote on the SFT generator's candidates (B3)
  verifiers    ways to choose on the same candidates: nothing learned (most written SQL, execution
               majority vote: A1), the untrained verifier (B5), v1 (SQL only), v2 (reads results), and v2
               trained on own answers only (B1), from the base model (B2), without the cost loss (B4);
               each alone and with the weighted vote; AUC of each verifier's score
  samples      accuracy as if only k of the 16 samples had been drawn, averaged over random draws (A2)
  prompts      candidates from the normal prompt, the efficient prompt, or both at equal count (A3)
  difficulty   EX by BIRD difficulty on dev and mini-dev (A5)
  learned      the verifiers' scores by what the candidate returned, and calibration (A6)
  cap          questions whose only correct candidates hit the 1 s cap (A7, one number)
  cost         the cost spread among the model's own correct candidates (A8)
  examples     dev questions where reading the result changed the choice (A9)
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

from effsql.io import iter_jsonl, latest_by, read_json, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
SPLITS = ("val", "dev", "minidev")
CAND = "judge_cand_v1"          # the final generator's candidates (rl_v1_it8)
CAND_SFT = "judge_cand_sft"     # B3: the SFT generator's candidates
JUDGES = {                      # run -> label; missing runs are left out of the tables
    "judge_untrained": "untrained verifier (B5)",
    "judge_v1": "verifier v1 (SQL only)",
    "judge_v2": "verifier v2 (reads results)",
    "judge_v2_own": "v2, own answers only (B1)",
    "judge_v2_base": "v2, from the base model (B2)",
    "judge_v2_nocost": "v2, no cost loss (B4)",
}
FINAL_JUDGE = "judge_v2"
FINAL_METHOD = "weighted_vote"
LADDER_RUNS = [                 # prediction runs scored by the evaluation: (run per split, label)
    ({"val": "zero_shot_v1", "dev": "pool_v1_{split}_base", "minidev": "pool_v1_{split}_base"}, "base model (zero-shot)"),
    ("sft_v1_best_e1", "+ SFT"),
    ("rl_v1_it8", "+ RL (generator alone)"),
]
REFS = {"ref_omnisql7b": "OmniSQL-7B", "ref_arctic7b": "Arctic-R1-7B", "ref_qwen32b": "Qwen2.5-Coder-32B zero-shot"}
SAMPLE_KS = (0, 1, 2, 4, 8, 16)


# Loading -------------------------------------------------------------------------------------------
def eval_correct(paths: Paths, run: str | dict[str, str], split: str) -> dict[str, dict[str, Any]] | None:
    """qid -> evaluation record of a prediction run (None when not evaluated). run: a run name (its
    evaluations are <run>_<split>), a template with {split}, or one of these per split."""
    from effsql.evaluate import is_current
    from effsql.train.report import eval_file

    f = eval_file(paths.runs_dir, run[split] if isinstance(run, dict) else run, split)
    if not f.exists():
        return None
    return latest_by(f, where=is_current) or None


class Split:
    """Candidates, labels, answer-time results and scores of one candidates run and split."""

    def __init__(self, paths: Paths, cand_run: str, split: str, judges: list[str]):
        from effsql.judge.candidates import read_candidates, read_labels
        from effsql.judge.score import read_scores
        from effsql.judge.vote import read_exec

        self.split, self.cand_run = split, cand_run
        recs = read_candidates(paths, cand_run, split)
        self.qids = [r["qid"] for r in recs]
        self.cands = {r["qid"]: r["cands"] for r in recs}
        labels = read_labels(paths, cand_run, split)
        if labels is None or any(q not in labels for q in self.qids):
            raise FileNotFoundError(f"{cand_run} {split}: labels missing (python -m effsql.judge label)")
        self.correct = {q: [bool(x["correct"]) for x in labels[q]["labels"]] for q in self.qids}
        self.steps = {q: [x.get("steps") if x["correct"] else None for x in labels[q]["labels"]] for q in self.qids}
        ex = read_exec(paths, cand_run, split)
        self.res = {q: (ex.get(q) or {}).get("results") for q in self.qids}
        self.p: dict[str, dict[str, list[float] | None]] = {}
        for j in judges:
            try:
                s = read_scores(paths, cand_run, j, split)
            except FileNotFoundError:
                continue
            self.p[j] = {q: (s.get(q) or {}).get("p") for q in self.qids}

    def pick(self, method: str, judge: str | None, q: str, keep: list[int] | None = None,
             n: list[int] | None = None) -> int | None:
        """Index (into the full candidate list) chosen by a method, on all candidates or the subset `keep`
        with counts `n`. method: greedy, text_vote, exec_vote, oracle, or a judge/vote.py method."""
        from effsql.judge.select import greedy_index, vote_index
        from effsql.judge.vote import method_index

        cs = self.cands[q]
        idx = list(range(len(cs))) if keep is None else keep
        if not idx:
            return None
        sub = [{**cs[i], "n": cs[i]["n"] if n is None else n[k]} for k, i in enumerate(idx)]
        if method == "oracle":
            return next((i for i in idx if self.correct[q][i]), idx[0])
        if method == "greedy":
            k = greedy_index(sub)
        elif method == "text_vote":
            k = vote_index(sub)
        else:
            res = self.res[q]
            res = [res[i] for i in idx] if res is not None and len(res) == len(cs) else None
            if method == "exec_vote":
                k = method_index("result_vote", sub, None, res)
            else:
                p = self.p[judge][q] if judge else None
                p = [p[i] for i in idx] if p is not None and len(p) == len(cs) else None
                k = method_index(method, sub, p, res)
        return None if k is None else idx[k]

    def ok(self, method: str, judge: str | None = None) -> list[bool]:
        out = []
        for q in self.qids:
            i = self.pick(method, judge, q)
            out.append(i is not None and self.correct[q][i])
        return out


def ex(x: list[bool]) -> float:
    return 100 * float(np.mean(x)) if len(x) else float("nan")


def ci(x: list[bool]) -> tuple[float, float]:
    from effsql.train.report import bootstrap_ci

    lo, hi = bootstrap_ci(np.asarray(x, dtype=float))
    return 100 * lo, 100 * hi


def test(a: list[bool], b: list[bool]) -> dict[str, Any]:
    from effsql.train.select import mcnemar_p

    return {"diff": ex(b) - ex(a), "p": mcnemar_p(a, b), "gained": sum(y and not x for x, y in zip(a, b)),
            "lost": sum(x and not y for x, y in zip(a, b))}


# Tables -------------------------------------------------------------------------------------------
def ladder(paths: Paths, data: dict[str, Split], sft: dict[str, Split]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, d in data.items():
        rows: list[tuple[str, list[bool]]] = []
        for run, label in LADDER_RUNS:
            ev = eval_correct(paths, run, split)
            if ev is None or any(q not in ev for q in d.qids):
                log.warning("ladder %s: %s not evaluated; row left out", split, label)
                continue
            rows.append((label, [bool(ev[q]["correct"]) for q in d.qids]))
        greedy = d.ok("greedy")
        if rows and rows[-1][1] != greedy:
            log.warning("ladder %s: the candidates' greedy answers differ from the RL run's evaluation", split)
        if "judge_v1" in d.p:
            rows.append(("+ 17 candidates, verifier v1 (SQL only)", d.ok("judge", "judge_v1")))
        rows.append(("verifier v2 (reads results)", d.ok("judge", FINAL_JUDGE)))
        rows.append(("+ weighted vote (final system)", d.ok(FINAL_METHOD, FINAL_JUDGE)))
        table = []
        for k, (label, x) in enumerate(rows):
            r = {"label": label, "ex": ex(x), "ci": ci(x)}
            if k:
                r["vs_previous"] = test(rows[k - 1][1], x)
            table.append(r)
        final = rows[-1][1]
        extra = []
        if split in sft:
            s = sft[split]
            if s.qids != d.qids:
                raise ValueError(f"{split}: the SFT candidates cover different questions")
            sft_alone = s.ok("greedy")
            for label, x in (("SFT generator + verifier v2", s.ok("judge", FINAL_JUDGE)),
                             ("SFT generator + verifier v2 + weighted vote", s.ok(FINAL_METHOD, FINAL_JUDGE))):
                extra.append({"label": label, "ex": ex(x), "ci": ci(x), "vs_sft_alone": test(sft_alone, x),
                              "vs_final": test(x, final)})
            extra.append({"label": "oracle, SFT candidates", "ex": ex(s.ok("oracle"))})
        extra.append({"label": "oracle, RL candidates", "ex": ex(d.ok("oracle"))})
        out[split] = {"n_questions": len(d.qids), "rows": table, "extra": extra}
    return out


def verifiers(data: dict[str, Split]) -> dict[str, Any]:
    from effsql.judge.model import auc

    out: dict[str, Any] = {}
    for split, d in data.items():
        final = d.ok(FINAL_METHOD, FINAL_JUDGE)
        alone = d.ok("judge", FINAL_JUDGE)
        rows = [{"label": "generator alone", "alone": ex(d.ok("greedy"))},
                {"label": "most written SQL (nothing learned)", "alone": ex(d.ok("text_vote"))},
                {"label": "execution majority vote (nothing learned, A1)", "with_vote": ex(d.ok("exec_vote")),
                 "vote_vs_final": test(d.ok("exec_vote"), final)}]
        for j, label in JUDGES.items():
            if j not in d.p:
                continue
            a, v = d.ok("judge", j), d.ok(FINAL_METHOD, j)
            ps = [(x, c) for q in d.qids if d.p[j][q] is not None for x, c in zip(d.p[j][q], d.correct[q])]
            r = {"label": label, "judge": j, "alone": ex(a), "with_vote": ex(v),
                 "auc": auc(np.array([x for x, _ in ps]), np.array([int(c) for _, c in ps])) if ps else None}
            if j != FINAL_JUDGE:
                r["alone_vs_v2"] = test(a, alone)
                r["vote_vs_final"] = test(v, final)
            rows.append(r)
        out[split] = {"n_questions": len(d.qids), "rows": rows, "oracle": ex(d.ok("oracle"))}
    return out


def sample_slots(paths: Paths, d: Split) -> dict[str, list[tuple[str, int | None]]]:
    """Per question: (prompt variant, candidate index or None) for every sampled completion, in order."""
    from effsql.gen.parse import extract_sql
    from effsql.gen.sample import is_current_gen
    from effsql.judge.data import norm_sql

    raw = latest_by(paths.run_dir(d.cand_run, create=False) / f"gen_raw_{d.split}.jsonl", key="key",
                    where=is_current_gen)
    out: dict[str, list[tuple[str, int | None]]] = {}
    for q in d.qids:
        index = {norm_sql(c["sql"]): i for i, c in enumerate(d.cands[q])}
        slots = []
        for key in (f"{q}|base", f"{q}|efficient"):
            for comp in (raw.get(key) or {}).get("completions", []):
                i = None
                if comp.get("finish_reason") != "length":
                    sql = extract_sql(comp["text"])[0]
                    i = index.get(norm_sql(sql)) if sql else None
                slots.append((key.split("|")[1], i))
        out[q] = slots
    return out


def subset_ok(d: Split, q: str, chosen: list[int | None], method: str, judge: str | None) -> bool:
    """Correctness of a method on the greedy answer plus the chosen samples (counts recomputed)."""
    from effsql.judge.select import greedy_index

    g = greedy_index(d.cands[q])
    counts: dict[int, int] = {}
    if g is not None:
        counts[g] = 1
    for i in chosen:
        if i is not None:
            counts[i] = counts.get(i, 0) + 1
    keep = sorted(counts)
    i = d.pick(method, judge, q, keep, [counts[k] for k in keep])
    return i is not None and d.correct[q][i]


CURVE_METHODS = [("oracle", None, "oracle (a correct candidate exists)"), ("exec_vote", None, "execution majority vote"),
                 ("judge", "judge_v1", "verifier v1"), ("judge", FINAL_JUDGE, "verifier v2"),
                 (FINAL_METHOD, FINAL_JUDGE, "v2 + weighted vote (final)")]


def sample_curve(paths: Paths, data: dict[str, Split], draws: int = 20, seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, d in data.items():
        slots = sample_slots(paths, d)
        full = {q: [i for _, i in s] for q, s in slots.items()}
        n_full = {q: len(s) for q, s in full.items()}
        rng = np.random.default_rng(seed)
        methods = [(m, j, label) for m, j, label in CURVE_METHODS if j is None or j in d.p]
        res: dict[str, dict[int, float]] = {label: {} for _, _, label in methods}
        for k in SAMPLE_KS:
            reps = 1 if k in (0, 16) else draws
            acc = {label: [] for _, _, label in methods}
            for _ in range(reps):
                pick = {q: [full[q][i] for i in rng.permutation(n_full[q])[:k]] for q in d.qids}
                for m, j, label in methods:
                    acc[label].append(ex([subset_ok(d, q, pick[q], m, j) for q in d.qids]))
            for label, v in acc.items():
                res[label][k] = float(np.mean(v))
        out[split] = {"n_questions": len(d.qids), "draws": draws, "ex": res}
    return out


def prompt_sources(paths: Paths, data: dict[str, Split], draws: int = 20, seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    methods = [x for x in CURVE_METHODS if x[0] in ("oracle", "exec_vote", FINAL_METHOD)]
    for split, d in data.items():
        slots = sample_slots(paths, d)
        rng = np.random.default_rng(seed)
        by = {q: {v: [i for vv, i in s if vv == v] for v in ("base", "efficient")} for q, s in slots.items()}
        res: dict[str, dict[str, float]] = {}
        for name, v in (("normal prompt, 8 samples", "base"), ("efficient prompt, 8 samples", "efficient")):
            res[name] = {label: ex([subset_ok(d, q, by[q][v], m, j) for q in d.qids]) for m, j, label in methods
                         if j is None or j in d.p}
        mixed: dict[str, list[float]] = {label: [] for m, j, label in methods if j is None or j in d.p}
        for _ in range(draws):
            pick = {q: [by[q][v][i] for v in ("base", "efficient") for i in rng.permutation(len(by[q][v]))[:4]]
                    for q in d.qids}
            for m, j, label in methods:
                if j is None or j in d.p:
                    mixed[label].append(ex([subset_ok(d, q, pick[q], m, j) for q in d.qids]))
        res["both prompts, 4 + 4 samples"] = {k: float(np.mean(v)) for k, v in mixed.items()}
        res["both prompts, 8 + 8 samples (final)"] = {
            label: ex([subset_ok(d, q, [i for _, i in slots[q]], m, j) for q in d.qids]) for m, j, label in methods
            if j is None or j in d.p}
        out[split] = res
    return out


def difficulty(paths: Paths, data: dict[str, Split]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "minidev"):
        if split not in data:
            continue
        d = data[split]
        runs = [(LADDER_RUNS[0][0], "base model (zero-shot)"), ("rl_v1_it8", "generator alone (RL)")] + list(REFS.items())
        ev0 = eval_correct(paths, "rl_v1_it8", split)
        if ev0 is None:
            continue
        diff = {q: ev0[q].get("difficulty") for q in d.qids}
        levels = [x for x in ("simple", "moderate", "challenging") if x in diff.values()]
        rows = []
        final = d.ok(FINAL_METHOD, FINAL_JUDGE)
        cols = []
        for run, label in runs:
            ev = eval_correct(paths, run, split)
            if ev is not None and all(q in ev for q in d.qids):
                cols.append((label, [bool(ev[q]["correct"]) for q in d.qids]))
        cols.insert(2, ("final system", final))
        for label, x in cols:
            r = {"label": label, "all": ex(x)}
            for lv in levels:
                r[lv] = ex([v for q, v in zip(d.qids, x) if diff[q] == lv])
            rows.append(r)
        out[split] = {"n": {lv: sum(diff[q] == lv for q in d.qids) for lv in levels}, "levels": levels, "rows": rows}
    return out


def result_kind(r: dict[str, Any] | None) -> str:
    if not r or r.get("compiles") is False:
        return "does not compile"
    if not r.get("ok"):
        return "timeout" if r.get("error_type") == "timeout" else "error"
    return "rows" if r.get("n_rows") else "empty"


def learned(data: dict[str, Split]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, d in data.items():
        per: dict[str, Any] = {}
        for j in d.p:
            groups: dict[str, list[float]] = {}
            pairs = []
            for q in d.qids:
                p, res = d.p[j][q], d.res[q]
                if p is None or res is None or len(p) != len(res):
                    continue
                for x, r, c in zip(p, res, d.correct[q]):
                    kind = result_kind(r)
                    if kind == "rows":
                        kind = "rows, correct" if c else "rows, wrong"
                    groups.setdefault(kind, []).append(x)
                    pairs.append((x, c))
            bins = []
            arr = np.array([x for x, _ in pairs])
            lab = np.array([c for _, c in pairs], dtype=float)
            which = np.minimum((arr * 10).astype(int), 9)  # ten bins of width 0.1, the last one closed
            for b in range(10):
                m = which == b
                if m.any():
                    bins.append({"from": b / 10, "n": int(m.sum()), "mean_p": float(arr[m].mean()),
                                 "share_correct": float(lab[m].mean())})
            per[j] = {"by_result": {k: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                                        "share_above_half": float(np.mean(np.asarray(v) > 0.5))}
                                    for k, v in sorted(groups.items())},
                      "calibration": bins,
                      "ece": float(sum(b["n"] * abs(b["mean_p"] - b["share_correct"]) for b in bins) / max(len(arr), 1))}
        out[split] = per
    return out


def cap(data: dict[str, Split]) -> dict[str, Any]:
    out = {}
    for split, d in data.items():
        lost = 0
        for q in d.qids:
            res = d.res[q] or []
            ok = [i for i, c in enumerate(d.correct[q]) if c]
            if ok and len(res) == len(d.correct[q]) and all(result_kind(res[i]) == "timeout" for i in ok):
                lost += 1
        out[split] = {"n_questions": len(d.qids), "n_lost": lost, "pct": 100 * lost / max(len(d.qids), 1)}
    return out


def cost_spread(data: dict[str, Split]) -> dict[str, Any]:
    out = {}
    for split, d in data.items():
        ratios, pick_ratios = [], []
        for q in d.qids:
            st = [s for s in d.steps[q] if s is not None and s > 0]
            if len(set(st)) >= 1 and len(st) >= 2:
                ratios.append(max(st) / min(st))
            i = d.pick(FINAL_METHOD, FINAL_JUDGE, q)
            if i is not None and d.steps[q][i] and st:
                pick_ratios.append(d.steps[q][i] / min(st))
        r, pr = np.asarray(ratios), np.asarray(pick_ratios)
        out[split] = {
            "n_questions": len(d.qids),
            "n_two_or_more_correct": len(r),
            "spread_median": float(np.median(r)) if len(r) else None,
            "pct_spread_2x": 100 * float(np.mean(r >= 2)) if len(r) else None,
            "n_final_correct": len(pr),
            "final_vs_cheapest_geomean": float(np.exp(np.mean(np.log(pr)))) if len(pr) else None,
            "pct_final_2x_costlier_than_cheapest": 100 * float(np.mean(pr >= 2)) if len(pr) else None,
        }
    return out


def examples(paths: Paths, d: Split, n: int = 6, timeout_s: float = 1.0) -> list[dict[str, Any]]:
    """Dev questions where v1 alone picked a candidate with no rows (empty, error, timeout) and v2 alone
    picked a correct one; shortest SQL first."""
    from effsql.data.bird import load_examples
    from effsql.judge.model import result_text

    if "judge_v1" not in d.p or FINAL_JUDGE not in d.p:
        return []
    by = {e.qid: e for e in load_examples(d.split, paths)}
    found = []
    for q in d.qids:
        i1, i2 = d.pick("judge", "judge_v1", q), d.pick("judge", FINAL_JUDGE, q)
        res = d.res[q]
        if i1 is None or i2 is None or res is None or d.correct[q][i1] or not d.correct[q][i2]:
            continue
        if result_kind(res[i1]) == "rows":
            continue
        e = by[q]
        found.append({"qid": q, "db_id": e.db_id, "question": e.question, "evidence": e.evidence,
                      "v1_pick": {"sql": d.cands[q][i1]["sql"], "result": result_text(res[i1], timeout_s),
                                  "p_v1": d.p["judge_v1"][q][i1], "p_v2": d.p[FINAL_JUDGE][q][i1]},
                      "v2_pick": {"sql": d.cands[q][i2]["sql"], "result": result_text(res[i2], timeout_s),
                                  "p_v1": d.p["judge_v1"][q][i2], "p_v2": d.p[FINAL_JUDGE][q][i2]}})
    found.sort(key=lambda x: len(x["v1_pick"]["sql"]) + len(x["v2_pick"]["sql"]))
    return found[:n]


# Output -------------------------------------------------------------------------------------------
def _p(x: float) -> str:
    return "<.001" if x < 0.001 else f"{x:.3f}"


def _t(t: dict[str, Any] | None) -> str:
    return "" if not t else f"{t['diff']:+6.1f}  p {_p(t['p']):>6}  (+{t['gained']}/-{t['lost']})"


def format_report(rep: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("COMPONENT LADDER (A4, B3): EX, 95% CI, and the change against the row above (McNemar)")
    for split, s in rep["ladder"].items():
        L.append(f"== {split} ({s['n_questions']} questions) ==")
        for r in s["rows"]:
            L.append(f"  {r['label']:<44}{r['ex']:6.1f}  [{r['ci'][0]:4.1f}, {r['ci'][1]:4.1f}]  {_t(r.get('vs_previous'))}")
        for r in s["extra"]:
            line = f"  {r['label']:<44}{r['ex']:6.1f}"
            if "vs_sft_alone" in r:
                line += f"  vs SFT alone {_t(r['vs_sft_alone'])};  final vs this {_t(r['vs_final'])}"
            L.append(line)
    L.append("")
    L.append("VERIFIERS (A1, B1, B2, B4, B5): EX on the final generator's candidates; tests against v2 (alone) "
             "and the final system (with vote)")
    for split, s in rep["verifiers"].items():
        L.append(f"== {split} ({s['n_questions']} questions; oracle {s['oracle']:.1f}) ==")
        L.append(f"  {'':<46}{'alone':>7}{'+ vote':>8}{'AUC':>7}   alone vs v2 / vote vs final")
        for r in s["rows"]:
            a = f"{r['alone']:7.1f}" if "alone" in r else " " * 7
            v = f"{r['with_vote']:8.1f}" if "with_vote" in r else " " * 8
            u = f"{r['auc']:7.3f}" if r.get("auc") is not None else " " * 7
            t = "   ".join(x for x in (_t(r.get("alone_vs_v2")), _t(r.get("vote_vs_final"))) if x)
            L.append(f"  {r['label']:<46}{a}{v}{u}   {t}")
    L.append("")
    L.append("SAMPLE COUNT (A2): EX with the greedy answer plus k of the 16 samples, mean over random draws")
    for split, s in rep["samples"].items():
        L.append(f"== {split} ==")
        L.append(f"  {'k samples':<40}" + "".join(f"{k:>7}" for k in SAMPLE_KS))
        for label, v in s["ex"].items():
            L.append(f"  {label:<40}" + "".join(f"{v[k]:7.1f}" for k in SAMPLE_KS))
    L.append("")
    L.append("PROMPT SOURCES (A3): EX with the greedy answer plus samples from each prompt")
    for split, s in rep["prompts"].items():
        L.append(f"== {split} ==")
        labels = list(next(iter(s.values())))
        L.append(f"  {'candidates':<38}" + "".join(f"{x[:22]:>24}" for x in labels))
        for name, v in s.items():
            L.append(f"  {name:<38}" + "".join(f"{v[x]:24.1f}" for x in labels))
    L.append("")
    L.append("BY DIFFICULTY (A5): EX per BIRD difficulty level")
    for split, s in rep["difficulty"].items():
        L.append(f"== {split} (" + ", ".join(f"{k} {v}" for k, v in s["n"].items()) + ") ==")
        L.append(f"  {'model':<32}{'all':>7}" + "".join(f"{lv:>13}" for lv in s["levels"]))
        for r in s["rows"]:
            L.append(f"  {r['label']:<32}{r['all']:7.1f}" + "".join(f"{r[lv]:13.1f}" for lv in s["levels"]))
    L.append("")
    L.append("WHAT THE VERIFIERS LEARNED (A6): score p by what the candidate returned (mean, share above 0.5); "
             "ECE = calibration error")
    for split, s in rep["learned"].items():
        L.append(f"== {split} ==")
        for j, v in s.items():
            parts = [f"{k} {x['mean']:.2f} ({100 * x['share_above_half']:.0f}%, n {x['n']})" for k, x in v["by_result"].items()]
            L.append(f"  {JUDGES.get(j, j):<32} ECE {v['ece']:.3f}; " + "; ".join(parts))
    L.append("")
    L.append("ANSWER-TIME CAP (A7): questions whose every correct candidate was stopped at 1 s")
    L.append("  " + "; ".join(f"{k} {v['n_lost']} of {v['n_questions']} ({v['pct']:.1f}%)" for k, v in rep["cap"].items()))
    L.append("")
    L.append("COST SPREAD OF THE MODEL'S OWN CORRECT CANDIDATES (A8)")
    for split, v in rep["cost"].items():
        if v["spread_median"] is None:
            continue
        L.append(f"  {split}: {v['n_two_or_more_correct']} questions with 2+ correct candidates; costliest/cheapest "
                 f"median {v['spread_median']:.2f}, at least 2x apart in {v['pct_spread_2x']:.1f}%; the final answer "
                 f"costs {v['final_vs_cheapest_geomean']:.3f}x the cheapest correct one (geomean), at least 2x in "
                 f"{v['pct_final_2x_costlier_than_cheapest']:.1f}%")
    L.append("")
    L.append("WORKED EXAMPLES (A9): v1 chose a query with no rows, v2 chose a correct one")
    for e in rep["examples"]:
        L.append(f"-- {e['qid']} ({e['db_id']}): {e['question']}  [evidence: {e['evidence'] or 'none'}]")
        for k in ("v1_pick", "v2_pick"):
            x = e[k]
            L.append(f"   {k} (p v1 {x['p_v1']:.2f}, p v2 {x['p_v2']:.2f}): {' '.join(x['sql'].split())}")
            L.append("      -> " + x["result"].replace("\n", " / "))
    return "\n".join(L)


def build_report(out: str, *, paths: Paths | None = None, draws: int = 20) -> dict[str, Any]:
    paths = paths or load_paths()
    judges = list(JUDGES)
    data = {s: Split(paths, CAND, s, judges) for s in SPLITS}
    sft = {}
    for s in SPLITS:
        try:
            sft[s] = Split(paths, CAND_SFT, s, [FINAL_JUDGE])
        except FileNotFoundError as e:
            log.warning("B3 left out for %s: %s", s, e)
    rep = {"ladder": ladder(paths, data, sft), "verifiers": verifiers(data), "samples": sample_curve(paths, data, draws),
           "prompts": prompt_sources(paths, data, draws), "difficulty": difficulty(paths, data),
           "learned": learned(data), "cap": cap(data), "cost": cost_spread(data),
           "examples": examples(paths, data["dev"]) if "dev" in data else []}
    d = paths.run_dir(out)
    write_json(d / "report.json", rep)
    (d / "report.txt").write_text(format_report(rep) + "\n", encoding="utf-8")
    return rep


def zero_shot_val(run: str, *, paths: Paths | None = None, pool_run: str = "pool_v1", export_run: str = "sft_v1"
                  ) -> dict[str, Any]:
    """runs/<run>/predictions_val.jsonl: the base model's greedy answer (normal prompt) to every validation
    question, from the pool's generations."""
    from effsql.gen.parse import extract_sql

    paths = paths or load_paths()
    val = read_json(paths.run_dir(export_run, create=False) / "export_meta.json")["val_qids"]
    want = {f"{q}|base|greedy": q for q in val}
    found: dict[str, dict[str, Any]] = {}
    for r in iter_jsonl(paths.run_dir(pool_run, create=False) / "generations_train.jsonl"):
        if r.get("key") in want:
            found[want[r["key"]]] = r
    missing = [q for q in val if q not in found]
    if missing:
        raise RuntimeError(f"{len(missing)} validation questions have no greedy answer in {pool_run}: {missing[:5]}")
    d = paths.run_dir(run)
    lines = []
    for q in val:
        r = found[q]
        comp = r["completions"][0]
        sql, how = extract_sql(comp["text"])
        lines.append(json.dumps({"qid": q, "sql": sql or "", "extraction": how, "finish_reason": comp.get("finish_reason"),
                                 "prompt_level": r.get("prompt_level"), "model": r.get("model"),
                                 "model_revision": r.get("model_revision")}))
    (d / "predictions_val.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"arm": "base", "epoch": 0, "tag": None, "method": "zero-shot-base", "complete": True,
            "source": f"{pool_run}/generations_train.jsonl (greedy, base prompt)", "n_questions": len(val)}
    write_json(d / "adapter_meta.json", meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m effsql.ablate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("zero-shot-val")
    p.add_argument("--run", default="zero_shot_v1")
    p = sub.add_parser("report")
    p.add_argument("--out", default="ablations_report")
    p.add_argument("--draws", type=int, default=20)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if a.cmd == "zero-shot-val":
        print(json.dumps(zero_shot_val(a.run), indent=2))
    else:
        print(format_report(build_report(a.out, draws=a.draws)))


if __name__ == "__main__":
    main()
