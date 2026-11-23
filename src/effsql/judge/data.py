"""`python -m effsql.judge data`: the judge's training data from already executed candidates (CPU, minutes).

Sources, train questions only (dev and mini-dev never enter):
  own   the generator's own answers (runs/<own_run>/iters/*/rewards.jsonl: 8 samples per question with
        status and step count, written by `rl --settings own_answers`).
  pool  the base model's candidates (runs/<pool_run>/pool_train.parquet), gold rows excluded.

Label yes: the status is `correct` (own: same result as gold and no lucky first-row pick or foreign
constant; pool: same result as gold and a quality verdict that is not fragile or unverifiable). Anything
else is no. Candidates are deduplicated per question by their whitespace-normalised text (own wins).

Per question up to `max_candidates`: the distinct own answers first, then pool candidates, each part in a
seeded random order. Cost label of a correct candidate with a step count: log2(steps + smooth) minus the
mean of that over all correct candidates of the question with a step count (own and pool, selected or
not); None when fewer than two exist (no relative information). The judge learns cost only from these.

Questions of the SFT export's held-out databases form the validation part.
Output: runs/<run>/judge_train.jsonl, judge_val.jsonl (one line per question) and data_meta.json.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from effsql.config import load_config
from effsql.gen.backend import stable_seed
from effsql.io import iter_jsonl, read_json, write_json
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)
DATA_VERSION = 1
CORRECT = "correct"
BAD_QUALITY = ("fragile", "unverifiable")  # same as effsql.pool.quality.BAD_VERDICTS


def norm_sql(sql: str) -> str:
    return " ".join(sql.split()).rstrip(";").strip()


def own_candidates(paths: Paths, own_run: str) -> dict[str, list[dict[str, Any]]]:
    """qid -> distinct own answers [{sql, status, steps}] (the last record of a question wins)."""
    files = sorted((paths.run_dir(own_run, create=False) / "iters").glob("*/rewards.jsonl"))
    if not files:
        raise FileNotFoundError(f"no rewards.jsonl under runs/{own_run}/iters: run the own-answers sampling first")
    out: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        for r in iter_jsonl(f):
            seen: dict[str, dict[str, Any]] = {}
            for sql, status, steps in zip(r["sql"], r["status"], r["steps"]):
                if sql and norm_sql(sql) not in seen:
                    seen[norm_sql(sql)] = {"sql": sql.strip(), "status": status, "steps": steps, "source": "own"}
            out[r["qid"]] = list(seen.values())
    return out


def pool_candidates(paths: Paths, pool_run: str) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    cols = ["qid", "split", "is_gold", "sql_original", "correct", "error_type", "quality", "vm_steps", "steps_capped"]
    df = pd.read_parquet(paths.run_dir(pool_run, create=False) / "pool_train.parquet", columns=cols)
    if (df["split"] != "train").any() or not df["qid"].astype(str).str.startswith("train_").all():
        raise ValueError("the pool holds rows that are not train data; refusing to build training data from it")
    df = df[~df["is_gold"].astype(bool)]
    out: dict[str, list[dict[str, Any]]] = {}
    for r in df.itertuples(index=False):
        ok = bool(r.correct) and r.quality not in BAD_QUALITY
        status = CORRECT if ok else ("lucky_match" if r.correct else (r.error_type or "wrong_result"))
        steps = None if (not ok or r.steps_capped or r.vm_steps is None or math.isnan(r.vm_steps)) else int(r.vm_steps)
        out.setdefault(r.qid, []).append({"sql": str(r.sql_original).strip(), "status": status, "steps": steps,
                                          "source": "pool"})
    return out


def cost_labels(cands: list[dict[str, Any]], smooth: float) -> dict[str, float]:
    """norm_sql -> relative log cost, over the correct candidates with a step count."""
    v = {norm_sql(c["sql"]): math.log2(c["steps"] + smooth) for c in cands
         if c["status"] == CORRECT and c["steps"] is not None}
    if len(v) < 2:
        return {}
    m = float(np.mean(list(v.values())))
    return {k: x - m for k, x in v.items()}


def question_record(qid: str, db_id: str, own: list[dict[str, Any]], pool: list[dict[str, Any]],
                    dc: dict[str, Any]) -> dict[str, Any] | None:
    merged: dict[str, dict[str, Any]] = {}
    for c in own + pool:  # own first: an identical pool candidate is dropped
        merged.setdefault(norm_sql(c["sql"]), c)
    if not merged:
        return None
    costs = cost_labels(list(merged.values()), dc["smooth_steps"])
    rng = np.random.default_rng(stable_seed(dc["seed"], "judge-data", qid))
    parts = [[c for c in merged.values() if c["source"] == s] for s in ("own", "pool")]
    chosen = [p[i] for p in parts for i in rng.permutation(len(p))][: dc["max_candidates"]]
    return {"qid": qid, "db_id": db_id, "cands": [
        {"sql": c["sql"], "source": c["source"], "status": c["status"], "steps": c["steps"],
         "label": int(c["status"] == CORRECT), "cost": costs.get(norm_sql(c["sql"]))} for c in chosen]}


def add_stats(stats: dict[str, Any], part: str, rec: dict[str, Any]) -> None:
    s = stats.setdefault(part, {"n_questions": 0, "n_candidates": 0, "n_yes": 0, "n_cost": 0, "n_own": 0,
                                "n_questions_all_no": 0, "n_questions_all_yes": 0})
    labels = [c["label"] for c in rec["cands"]]
    s["n_questions"] += 1
    s["n_candidates"] += len(labels)
    s["n_yes"] += sum(labels)
    s["n_cost"] += sum(c["cost"] is not None for c in rec["cands"])
    s["n_own"] += sum(c["source"] == "own" for c in rec["cands"])
    s["n_questions_all_no"] += not any(labels)
    s["n_questions_all_yes"] += all(labels)


def finish_stats(stats: dict[str, Any]) -> None:
    for s in stats.values():
        s["share_yes"] = s["n_yes"] / max(s["n_candidates"], 1)
        s["candidates_per_question"] = s["n_candidates"] / max(s["n_questions"], 1)


def build_judge_data(run: str, *, paths: Paths | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    from effsql.data.bird import load_examples
    from effsql.io import append_jsonl

    paths = paths or load_paths()
    cfg = cfg or load_config("judge.yaml")
    dc = cfg["data"]
    holdout = set(read_json(paths.run_dir(dc["export_run"], create=False) / "export_meta.json")["holdout_db_ids"])
    own = own_candidates(paths, dc["own_run"])
    pool = pool_candidates(paths, dc["pool_run"])
    run_dir = paths.run_dir(run)
    outs = {part: run_dir / f"judge_{part}.jsonl" for part in ("train", "val")}
    for p in outs.values():
        if p.exists():
            p.unlink()
    stats: dict[str, Any] = {}
    for e in load_examples("train", paths):
        rec = question_record(e.qid, e.db_id, own.get(e.qid, []), pool.get(e.qid, []), dc)
        if rec is None:
            continue
        part = "val" if e.db_id in holdout else "train"
        append_jsonl(outs[part], [rec])
        add_stats(stats, part, rec)
    finish_stats(stats)
    meta = {"data_version": DATA_VERSION, "config": dc, "holdout_db_ids": sorted(holdout), "stats": stats}
    write_json(run_dir / "data_meta.json", meta)
    log.info("judge data %s: %s", run, stats)
    return meta


def read_part(run_dir: Path, part: str) -> list[dict[str, Any]]:
    from effsql.io import read_jsonl

    return read_jsonl(run_dir / f"judge_{part}.jsonl")


def derive_judge_data(src: str, run: str, sources: tuple[str, ...] = ("own",), *,
                      paths: Paths | None = None) -> dict[str, Any]:
    """A judge data run that keeps only the candidates of `sources` from the data run `src`, with the answer-time
    results of `src` (exec_<part>.jsonl) carried over, so nothing is executed again. Cost labels are recomputed
    over the kept candidates; questions left without a candidate are dropped. Used for the ablation that trains
    the judge on its generator's own answers only (sources=("own",)). Since the own answers come first in a
    question and number at most max_candidates, the kept candidates equal what `data` would select from own
    answers alone."""
    from effsql.io import append_jsonl
    from effsql.judge.vote import read_exec

    paths = paths or load_paths()
    src_dir, run_dir = paths.run_dir(src, create=False), paths.run_dir(run)
    smeta = read_json(src_dir / "data_meta.json")
    stats: dict[str, Any] = {}
    for part in ("train", "val"):
        ex = read_exec(paths, src, part)
        out_d, out_x = run_dir / f"judge_{part}.jsonl", run_dir / f"exec_{part}.jsonl"
        for f in (out_d, out_x):
            if f.exists():
                f.unlink()
        for rec in read_part(src_dir, part):
            res = ex[rec["qid"]]["results"]
            if len(res) != len(rec["cands"]):
                raise ValueError(f"{src}: exec_{part} does not match the candidates of {rec['qid']}")
            keep = [i for i, c in enumerate(rec["cands"]) if c["source"] in sources]
            if not keep:
                continue
            cands = [dict(rec["cands"][i]) for i in keep]
            costs = cost_labels(cands, smeta["config"]["smooth_steps"])
            for c in cands:
                c["cost"] = costs.get(norm_sql(c["sql"]))
            new = {**rec, "cands": cands}
            append_jsonl(out_d, [new])
            append_jsonl(out_x, [{"qid": rec["qid"], "results": [res[i] for i in keep]}])
            add_stats(stats, part, new)
    finish_stats(stats)
    meta = {"data_version": DATA_VERSION, "config": {**smeta["config"], "sources": list(sources)},
            "derived_from": src, "holdout_db_ids": smeta["holdout_db_ids"], "stats": stats}
    write_json(run_dir / "data_meta.json", meta)
    log.info("judge data %s (from %s, %s only): %s", run, src, "+".join(sources), stats)
    return meta
