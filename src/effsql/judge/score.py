"""`python -m effsql.judge score`: the judge's p and c for every candidate of a split (GPU, resumable).

Reads runs/<cand_run>/candidates_<split>.jsonl, writes runs/<cand_run>/scores_<judge>_<split>.jsonl:
{qid, p: [...], c: [...]} in the order of the candidates (p = sigmoid of the yes-minus-no logit).
Nothing is executed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from effsql.config import load_config
from effsql.io import append_jsonl, done_ids, latest_by
from effsql.judge.candidates import data_split, read_candidates
from effsql.judge.model import build_questions, forward, make_groups, sigmoid
from effsql.paths import Paths, load_paths

log = logging.getLogger(__name__)


def scores_path(paths: Paths, cand_run: str, judge_run: str, split: str):
    return paths.run_dir(cand_run) / f"scores_{judge_run}_{split}.jsonl"


def score_split(judge_run: str, cand_run: str, split: str, judge: tuple[Any, Any, Any, tuple[int, int]], *,
                paths: Paths | None = None, jcfg: dict[str, Any] | None = None, cfg: dict[str, Any] | None = None,
                device: str = "cuda") -> dict[str, Any]:
    import torch

    paths = paths or load_paths()
    jcfg = jcfg or load_config("judge.yaml")
    cfg = cfg or load_config("train.yaml")
    model, head, tok, yes_no = judge
    out = scores_path(paths, cand_run, judge_run, split)
    done = done_ids(out)
    recs = [r for r in read_candidates(paths, cand_run, split) if r["qid"] not in done]
    questions, stats = build_questions(tok, recs, data_split(split), paths, cfg["prompt"]["max_prompt_tokens"])
    groups = make_groups(questions, list(range(len(questions))), jcfg["score"]["questions_per_seq"],
                         int(jcfg["train"].get("max_seq_tokens", 16000)), np.random.default_rng(0))
    log.info("score %s: %d questions in %d sequences (%d done before)", split, len(questions), len(groups), len(done))
    with torch.no_grad():
        for k, g in enumerate(groups, 1):
            group = [questions[i] for i in g]
            lp, c = forward(model, head, group, device, yes_no)
            lp, c = lp.tolist(), c.tolist()
            rows, j = [], 0
            for q in group:
                n = len(q.ids)
                rows.append({"qid": q.qid, "lp": lp[j:j + n], "p": [sigmoid(x) for x in lp[j:j + n]], "c": c[j:j + n]})
                j += n
            append_jsonl(out, rows)
            if k % 50 == 0:
                log.info("score %s: %d/%d sequences", split, k, len(groups))
    # questions with no candidate (or too long) get no scores: the selection falls back to greedy
    return {"split": split, "n_scored_now": len(questions), "n_done_before": len(done), **stats}


def read_scores(paths: Paths, cand_run: str, judge_run: str, split: str) -> dict[str, dict[str, Any]]:
    path = scores_path(paths, cand_run, judge_run, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m effsql.judge score` first")
    return latest_by(path)
