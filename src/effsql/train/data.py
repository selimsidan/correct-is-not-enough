"""`effsql train-export`: training data for the four arms, built from a candidate pool (CPU only).

Arms. The three SFT arms have exactly one example per train question per epoch, so they cost the
same GPU time and differ only in which targets they use:
  plain     the gold SQL.
  weighted  one correct candidate, drawn with probability proportional to 2^(beta * eff_score),
            i.e. (gold_cost / cost)^beta. beta = 0 is uniform over correct candidates, a large beta
            is best-only. In expectation this equals a loss weighted by the same probabilities.
  tagged    one correct candidate per epoch (a different one each epoch while they last), with its
            eff_score shown to the model as a tag in the prompt. At test time a high tag is asked for.
  dpo       preference pairs: a cheaper correct candidate over a pricier one at a different
            efficiency level, and a correct candidate over an incorrect one.

Rules:
  - Only pool rows of split "train" are used; dev and minidev never enter training data.
  - Validation uses held-out train databases (whole databases, so no schema is shared with training).
  - Censored candidates (steps_capped: counting stopped, the cost is only a bound) are never SFT
    targets or DPO chosen answers. They can be DPO rejected answers (they are surely expensive).
  - Draws use per-question seeds, so a question's examples do not depend on which other questions
    are exported (a --limit run draws the same examples for the questions it keeps).

The export holds references (qid, candidate, SQL, tag), not rendered prompts: prompts are built at
training time from the schema cache with the real tokenizer, by the same function `predict` uses.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from effsql.config import load_config, save_config
from effsql.gen.backend import stable_seed
from effsql.gen.prompts import CountTokens, Messages, build_question_prompt
from effsql.io import append_jsonl, iter_jsonl, write_json
from effsql.paths import Paths, load_paths
from effsql.runmeta import sha256_file, write_run_meta

log = logging.getLogger(__name__)

SFT_ARMS = ("plain", "weighted", "tagged")
ARMS = SFT_ARMS + ("dpo",)
EXPORT_VERSION = 1
CHARS_PER_TOKEN = 3.5  # same rough rate as gen.prompts.estimate_tokens


# Prompts and targets (shared with training and prediction) ---------------------------------------
def arm_prompt_variant(arm: str) -> str:
    """Prompt variant an arm trains and predicts with."""
    return "tagged" if arm == "tagged" else "base"


def target_completion(sql: str) -> str:
    """Assistant answer in exactly the form `extract_sql` reads back."""
    return f"```sql\n{sql.strip()}\n```"


def tag_value(eff: float, clip: float, decimals: int) -> float:
    return round(min(max(float(eff), -clip), clip), decimals)


def format_tag(value: float, decimals: int) -> str:
    """Signed fixed-point text, e.g. +1.4, -0.3, +0.0."""
    v = 0.0 if value == 0 else value  # no "-0.0"
    return f"{v:+.{decimals}f}"


def train_messages(templates: dict[str, str], arm: str, example: Any, schema: dict[str, Any], budget: int,
                   count_tokens: CountTokens, tag: str | None = None) -> tuple[Messages, str, int]:
    """Prompt for one training example: the same function and budget `predict` uses at test time."""
    return build_question_prompt(templates, arm_prompt_variant(arm), example, schema, budget, count_tokens, tag=tag)


# Pool selection -------------------------------------------------------------------------------
def load_pool(paths: Paths, pool_run: str) -> pd.DataFrame:
    path = paths.run_dir(pool_run) / "pool_train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `effsql build-pool --split train --run {pool_run}`.")
    return check_train_only(pd.read_parquet(path))


def check_train_only(pool: pd.DataFrame) -> pd.DataFrame:
    """Refuse anything that is not train data (dev and minidev are evaluation only)."""
    bad = pool[(pool["split"] != "train") | ~pool["qid"].astype(str).str.startswith("train_")]
    if len(bad):
        raise ValueError(f"{len(bad)} pool rows are not train data (e.g. {bad['qid'].iloc[0]!r}); refusing to "
                         "build training data from them")
    return pool


def holdout_dbs(db_ids: list[str], fraction: float, seed: int) -> list[str]:
    """Whole databases held out for validation (at least one when there are two or more)."""
    dbs = sorted(set(db_ids))
    if len(dbs) < 2 or fraction <= 0:
        return []
    n = min(len(dbs) - 1, max(1, round(fraction * len(dbs))))
    rng = np.random.default_rng(stable_seed(seed, "holdout"))
    return sorted(rng.choice(dbs, size=n, replace=False).tolist())


def sft_targets(pool: pd.DataFrame) -> pd.DataFrame:
    """Correct candidates with a counted (not censored) cost: the only valid SFT targets."""
    return pool[pool["correct"] & ~pool["steps_capped"] & pool["eff_score"].notna()]


def select_qids(pool: pd.DataFrame, limit: int | None) -> list[str]:
    """Questions with a correct gold row, sorted; --limit keeps the first N (smoke runs)."""
    g = pool[pool["is_gold"] & pool["correct"]]
    qids = sorted(g["qid"].unique().tolist())
    return qids[:limit] if limit else qids


# Arm builders ---------------------------------------------------------------------------------
def weighted_probs(eff: np.ndarray, beta: float) -> np.ndarray:
    """p_i proportional to 2^(beta * eff_i), computed in log space so large beta cannot overflow."""
    logits = beta * math.log(2.0) * np.asarray(eff, dtype=float)
    w = np.exp(logits - logits.max())
    return w / w.sum()


def _base(arm: str, part: str, epoch: int, row: pd.Series) -> dict[str, Any]:
    return {"arm": arm, "part": part, "epoch": epoch, "qid": row["qid"], "db_id": row["db_id"],
            "cand_id": row["cand_id"], "source": row["source"], "sql": row["sql_original"],
            "eff_score": None if pd.isna(row["eff_score"]) else float(row["eff_score"]),
            "export_version": EXPORT_VERSION}


def build_plain(targets: pd.DataFrame, qids: list[str], part: str, epochs: int) -> list[dict[str, Any]]:
    by_q = {q: g for q, g in targets[targets["is_gold"]].groupby("qid")}
    out = []
    for e in range(epochs):
        for q in qids:
            out.append(_base("plain", part, e, by_q[q].iloc[0]))
    return out


def build_weighted(targets: pd.DataFrame, qids: list[str], part: str, epochs: int, beta: float,
                   seed: int) -> list[dict[str, Any]]:
    by_q = {q: g.sort_values("cand_id") for q, g in targets.groupby("qid")}
    out = []
    for q in qids:
        g = by_q[q]
        p = weighted_probs(g["eff_score"].to_numpy(), beta)
        rng = np.random.default_rng(stable_seed(seed, "weighted", q))
        for e in range(epochs):
            i = int(rng.choice(len(g), p=p))
            out.append({**_base("weighted", part, e, g.iloc[i]), "p": float(p[i]), "n_correct": len(g),
                        "beta": beta})
    return sorted(out, key=lambda r: (r["epoch"], r["qid"]))


def build_tagged(targets: pd.DataFrame, qids: list[str], part: str, epochs: int, clip: float, decimals: int,
                 seed: int) -> list[dict[str, Any]]:
    """Each epoch takes the next candidate of a seeded shuffle, so epochs show different tags."""
    by_q = {q: g.sort_values("cand_id") for q, g in targets.groupby("qid")}
    out = []
    for q in qids:
        g = by_q[q]
        order = np.random.default_rng(stable_seed(seed, "tagged", q)).permutation(len(g))
        for e in range(epochs):
            row = g.iloc[int(order[e % len(g)])]
            t = tag_value(row["eff_score"], clip, decimals)
            out.append({**_base("tagged", part, e, row), "tag": t, "tag_text": format_tag(t, decimals),
                        "n_correct": len(g)})
    return sorted(out, key=lambda r: (r["epoch"], r["qid"]))


def _pair(part: str, kind: str, q: str, ch: pd.Series, rj: pd.Series) -> dict[str, Any]:
    def lv(r: pd.Series) -> int | None:
        return None if pd.isna(r.get("efficiency_level")) else int(r["efficiency_level"])

    return {"arm": "dpo", "part": part, "epoch": 0, "qid": q, "db_id": ch["db_id"], "pair_type": kind,
            "chosen_cand": ch["cand_id"], "rejected_cand": rj["cand_id"],
            "chosen_sql": ch["sql_original"], "rejected_sql": rj["sql_original"],
            "chosen_source": ch["source"], "rejected_source": rj["source"],
            "chosen_level": lv(ch), "rejected_level": lv(rj),
            "chosen_eff": float(ch["eff_score"]),
            "rejected_eff": None if pd.isna(rj["eff_score"]) else float(rj["eff_score"]),
            # an incorrect candidate without an error type ran fine but returned the wrong result
            "rejected_error": None if rj["correct"] else (rj["error_type"] if pd.notna(rj["error_type"]) else "wrong_result"),
            "export_version": EXPORT_VERSION}


def question_pairs(g: pd.DataFrame) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """All candidate pairs of one question, as (chosen, rejected) row positions in g.
    Efficiency: chosen counted-cost correct, rejected correct at a strictly higher level.
    Correctness: chosen counted-cost correct at the question's lowest level, rejected incorrect with
    SQL to show (parse failures have none)."""
    lvl = g["efficiency_level"]
    ok = (g["correct"] & ~g["steps_capped"] & g["eff_score"].notna()).to_numpy()
    correct = g["correct"].to_numpy()
    eff = [(i, j) for i in range(len(g)) for j in range(len(g))
           if ok[i] and correct[j] and pd.notna(lvl.iloc[i]) and pd.notna(lvl.iloc[j]) and lvl.iloc[i] < lvl.iloc[j]]
    ok_lv = lvl[ok]
    best = [i for i in range(len(g)) if ok[i] and pd.notna(lvl.iloc[i]) and lvl.iloc[i] == ok_lv.min()]
    wrong = [j for j in range(len(g)) if not correct[j] and g["error_type"].iloc[j] != "parse_fail"
             and str(g["sql_original"].iloc[j]).strip()]
    corr = [(i, j) for j in wrong for i in best]
    return eff, corr


def build_dpo(pool: pd.DataFrame, qids: list[str], part: str, per_question: int, efficiency_share: float,
              seed: int) -> list[dict[str, Any]]:
    """Up to per_question pairs per question, efficiency_share of them efficiency pairs; a type with too
    few pairs is topped up from the other. Each rejected answer is used at most once per question."""
    by_q = {q: g.sort_values("cand_id").reset_index(drop=True) for q, g in pool.groupby("qid")}
    n_eff_target = round(per_question * efficiency_share)
    out = []
    for q in qids:
        g = by_q[q]
        eff, corr = question_pairs(g)
        rng = np.random.default_rng(stable_seed(seed, "dpo", q))
        eff = [eff[k] for k in rng.permutation(len(eff))]
        corr = [corr[k] for k in rng.permutation(len(corr))]
        picked: list[tuple[str, tuple[int, int]]] = []
        used_rej: set[int] = set()

        def take(cands: list[tuple[int, int]], kind: str, n: int) -> None:
            for c, r in cands:
                if len([1 for k, _ in picked if k == kind]) >= n or len(picked) >= per_question:
                    return
                if r not in used_rej:
                    used_rej.add(r)
                    picked.append((kind, (c, r)))

        take(eff, "efficiency", n_eff_target)
        take(corr, "correctness", per_question - len(picked))
        take(eff, "efficiency", per_question)  # top up when correctness pairs ran out
        out += [_pair(part, kind, q, g.iloc[c], g.iloc[r]) for kind, (c, r) in picked]
    return out


# Statistics -----------------------------------------------------------------------------------
def _quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float)
    return {"n": len(xs), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def prompt_tokens_from_generation(paths: Paths, pool_run: str) -> dict[str, int]:
    """Real prompt token counts per qid (base prompt, from the generation records of the pool run)."""
    out: dict[str, int] = {}
    for r in iter_jsonl(paths.run_dir(pool_run) / "generations_train.jsonl"):
        if r.get("variant") == "base" and r.get("prompt_tokens") is not None:
            out[r["qid"]] = int(r["prompt_tokens"])
    return out


def inference_tag(targets: pd.DataFrame, qids: list[str], quantile: float, clip: float, decimals: int) -> float:
    """Default test-time tag: a high quantile of each train question's best achievable eff_score."""
    best = targets[targets["qid"].isin(qids)].groupby("qid")["eff_score"].max()
    return tag_value(float(np.quantile(best.to_numpy(), quantile)), clip, decimals) if len(best) else 0.0


def arm_stats(arm: str, recs: list[dict[str, Any]], prompt_tokens: dict[str, int]) -> dict[str, Any]:
    s: dict[str, Any] = {"n_examples": len(recs), "n_questions": len({r["qid"] for r in recs})}
    if arm == "dpo":
        s["by_type"] = dict(Counter(r["pair_type"] for r in recs))
        s["rejected_error"] = dict(Counter(str(r["rejected_error"]) for r in recs if r["pair_type"] == "correctness"))
        per_q = Counter(r["qid"] for r in recs)
        s["pairs_per_question"] = _quantiles(list(per_q.values()))
        sql_chars = [len(r["chosen_sql"]) + len(r["rejected_sql"]) for r in recs]
    else:
        s["by_source"] = dict(Counter(r["source"] for r in recs))
        s["share_gold"] = sum(r["source"] == "gold" for r in recs) / max(len(recs), 1)
        s["eff_score"] = _quantiles([r["eff_score"] for r in recs if r["eff_score"] is not None])
        if arm == "weighted":
            s["p_drawn"] = _quantiles([r["p"] for r in recs])
            s["n_correct_per_question"] = _quantiles([r["n_correct"] for r in recs if r["epoch"] == 0])
        if arm == "tagged":
            s["tag_histogram"] = dict(sorted(Counter(r["tag_text"] for r in recs).items(),
                                             key=lambda kv: float(kv[0])))
        sql_chars = [len(r["sql"]) for r in recs]
    known = [prompt_tokens[r["qid"]] for r in recs if r["qid"] in prompt_tokens]
    mean_prompt = float(np.mean(known)) if known else None
    completion = sum(sql_chars) / CHARS_PER_TOKEN
    s["prompt_tokens"] = _quantiles(known)
    s["prompt_tokens_coverage"] = len(known) / max(len(recs), 1)
    if mean_prompt is not None:
        # DPO runs the prompt with both answers (and once more for the precomputed reference).
        mult = 2 if arm == "dpo" else 1
        s["total_tokens_estimate"] = int(mean_prompt * len(recs) * mult + completion)
    return s


# Export ---------------------------------------------------------------------------------------
def train_export(pool_run: str, run: str, *, paths: Paths | None = None, cfg: dict[str, Any] | None = None,
                 arms: list[str] | None = None, limit: int | None = None,
                 beta: float | None = None) -> dict[str, Any]:
    paths = paths or load_paths()
    cfg = cfg or load_config("train.yaml")
    ex = cfg["export"]
    if beta is not None:
        ex = {**ex, "weighted": {**ex["weighted"], "beta": beta}}
    arms = arms or list(ARMS)
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {ARMS}")
    seed, epochs = ex["seed"], ex["epochs"]

    pool_path = paths.run_dir(pool_run) / "pool_train.parquet"
    pool = load_pool(paths, pool_run)
    targets = sft_targets(pool)
    qids = select_qids(pool, limit)
    q_db = pool.drop_duplicates("qid").set_index("qid")["db_id"]
    held = holdout_dbs(q_db.loc[qids].tolist(), ex["val_db_fraction"], seed)
    parts = {"train": [q for q in qids if q_db[q] not in held], "val": [q for q in qids if q_db[q] in held]}
    tc = ex["tagged"]

    def build(arm: str, part: str) -> list[dict[str, Any]]:
        qs, n_ep = parts[part], (epochs if part == "train" else 1)
        if arm == "plain":
            return build_plain(targets, qs, part, n_ep)
        if arm == "weighted":
            return build_weighted(targets, qs, part, n_ep, ex["weighted"]["beta"], seed)
        if arm == "tagged":
            return build_tagged(targets, qs, part, n_ep, tc["clip"], tc["decimals"], seed)
        d = ex["dpo"]
        return build_dpo(pool, qs, part, d["pairs_per_question"], d["efficiency_share"], seed)

    run_dir = paths.run_dir(run)
    data_dir = run_dir / "data"
    prompt_tokens = prompt_tokens_from_generation(paths, pool_run)
    stats: dict[str, Any] = {}
    files: dict[str, str] = {}
    for arm in arms:
        for part in ("train", "val"):
            recs = build(arm, part)
            path = data_dir / f"{arm}_{part}.jsonl"
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.unlink(missing_ok=True)
            append_jsonl(tmp, recs)
            tmp.replace(path)  # readers never see a half-written file
            files[f"{arm}_{part}"] = str(path)
            stats[f"{arm}_{part}"] = arm_stats(arm, recs, prompt_tokens)

    meta = {
        "pool_run": pool_run, "pool_path": str(pool_path), "pool_sha256": sha256_file(pool_path),
        "export_version": EXPORT_VERSION, "arms": arms, "limit": limit, "export_cfg": ex,
        "n_questions": {k: len(v) for k, v in parts.items()},
        "holdout_db_ids": held,
        "n_pool_questions": int(pool["qid"].nunique()),
        "inference_tag": inference_tag(targets, parts["train"], tc["inference_quantile"], tc["clip"], tc["decimals"]),
        "files": files, "stats": stats,
    }
    write_json(run_dir / "export_meta.json", meta)
    save_config(cfg, run_dir / "train_config.yaml")
    write_run_meta(run_dir, extra={"command": "train-export", "pool_run": pool_run})
    log.info("exported %s to %s (%d train / %d val questions, %d held-out databases)", ",".join(arms), data_dir,
             len(parts["train"]), len(parts["val"]), len(held))
    return meta


def read_export(run_dir: Path, arm: str, part: str) -> list[dict[str, Any]]:
    """Records of one arm and part, checked for the current export version."""
    recs = list(iter_jsonl(run_dir / "data" / f"{arm}_{part}.jsonl"))
    old = [r for r in recs if r.get("export_version") != EXPORT_VERSION]
    if old:
        raise RuntimeError(f"{len(old)} records in {arm}_{part} come from an older export; rerun train-export")
    return recs
