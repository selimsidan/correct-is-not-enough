"""The judge model: prompts, shared-prefix packing, and the forward pass that gives p and c per candidate.

Prompt (configs/prompts/judge_*.txt): the generator's prompt layout (schema first, then evidence and
question) followed by the candidate query and "Is the candidate query correct? Answer yes or no.", in the
chat template, ending where the assistant's answer begins. At that last position the model's hidden state
h gives:
  correctness  logit of "yes" minus logit of "no" from the language-model head (p = sigmoid of it);
  cost         c = w . h + b, a linear head (hidden_size + 1 numbers) trained on relative log cost.

Packing: questions of one database share the system prompt and schema, and the candidates of a question
share everything up to the candidate SQL. One sequence holds [shared prefix][question 1 text][candidate
1][candidate 2]...[question 2 text]...; a candidate attends to the shared prefix, its question's text and
itself, with the position ids it has when alone, so every candidate's outputs equal those of its own
sequence. The shared parts are computed once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from effsql.gen.prompts import SHRINK_LADDER, fill, fit_messages, render_schema

ASK = "Candidate query:\n```sql\n"


@dataclass
class JQuestion:
    qid: str
    key: str  # database and schema shrink level: questions with the same key share the prompt prefix
    ids: list[list[int]]  # full prompt token ids per candidate
    labels: list[int | None] = field(default_factory=list)
    costs: list[float | None] = field(default_factory=list)


# Prompts ----------------------------------------------------------------------------------------
def judge_messages(templates: dict[str, str], example: Any, schema_text: str, sql: str) -> list[dict[str, str]]:
    user = fill(templates["judge_user"], {"db_id": example.db_id, "schema": schema_text, "question": example.question,
                                          "evidence": example.evidence or "(none)", "sql": sql.strip()})
    return [{"role": "system", "content": templates["judge_system"]}, {"role": "user", "content": user}]


def question_prompts(tok: Any, templates: dict[str, str], example: Any, schema: dict[str, Any], sqls: list[str],
                     budget: int) -> tuple[list[list[int]], str, int]:
    """Token ids of every candidate's prompt with one schema level for the whole question (the level at
    which the longest candidate fits). Returns (ids per candidate, level, longest prompt length)."""
    from effsql.train.sft import make_counter, prompt_text

    longest = max(sqls, key=len) if sqls else ""
    _, level, n = fit_messages(lambda s: judge_messages(templates, example, s, longest), schema, budget,
                               make_counter(tok))
    schema_text = render_schema(schema, next(x for x in SHRINK_LADDER if x["name"] == level))
    texts = [prompt_text(tok, judge_messages(templates, example, schema_text, s)) for s in sqls]
    if not texts:
        return [], level, n
    # The text before the SQL is the same for every candidate: tokenized once, checked once against the
    # whole prompt (token boundaries), else every prompt is tokenized whole.
    k = texts[0].rfind(ASK) + len(ASK)
    head = tok(texts[0][:k], add_special_tokens=False)["input_ids"]
    ids = [head + tok(t[k:], add_special_tokens=False)["input_ids"] for t in texts]
    if ids[0] != tok(texts[0], add_special_tokens=False)["input_ids"]:
        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    return ids, level, n


def answer_token_ids(tok: Any) -> tuple[int, int]:
    """First token of "yes" and of "no" (single tokens in Qwen2.5's vocabulary)."""
    y = tok("yes", add_special_tokens=False)["input_ids"][0]
    n = tok("no", add_special_tokens=False)["input_ids"][0]
    if y == n:
        raise ValueError("the tokenizer does not tell yes from no by their first token")
    return int(y), int(n)


# Packing ----------------------------------------------------------------------------------------
def common_prefix(seqs: list[list[int]], cap: int) -> int:
    n = max(0, cap)
    first = np.asarray(seqs[0][:n])
    for s in seqs[1:]:
        diff = np.nonzero(np.asarray(s[:n]) != first[:n])[0]
        if len(diff):
            n = int(diff[0])
    return n


def pack(group: list[JQuestion]) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """(token ids, position ids, question segment, candidate segment, readout index per candidate).
    Segment 0 is shared; every candidate keeps at least its last token as its own."""
    all_ids = [c for q in group for c in q.ids]
    p = common_prefix(all_ids, min(len(x) for x in all_ids) - 1)
    ids, pos, sq, sc, read = list(all_ids[0][:p]), list(range(p)), [0] * p, [0] * p, []
    for qi, q in enumerate(group, start=1):
        qp = max(p, common_prefix(q.ids, min(len(x) for x in q.ids) - 1))
        ids += q.ids[0][p:qp]
        pos += range(p, qp)
        sq += [qi] * (qp - p)
        sc += [0] * (qp - p)
        for ci, c in enumerate(q.ids, start=1):
            ids += c[qp:]
            pos += range(qp, len(c))
            sq += [qi] * (len(c) - qp)
            sc += [ci] * (len(c) - qp)
            read.append(len(ids) - 1)
    return ids, pos, sq, sc, read


def packed_len(group: list[JQuestion]) -> int:
    return len(pack(group)[0])


def make_groups(questions: list[JQuestion], idx: list[int], per_seq: int, max_tokens: int,
                rng: np.random.Generator) -> list[list[int]]:
    """Per key, a random order filled greedily into groups of up to per_seq questions and max_tokens."""
    by_key: dict[str, list[int]] = {}
    for i in idx:
        by_key.setdefault(questions[i].key, []).append(i)
    groups: list[list[int]] = []
    for k in sorted(by_key):
        members = [by_key[k][j] for j in rng.permutation(len(by_key[k]))]
        cur: list[int] = []
        for i in members:
            if cur and (len(cur) >= per_seq or packed_len([questions[j] for j in cur + [i]]) > max_tokens):
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
    return groups


def plan_steps(questions: list[JQuestion], per_seq: int, per_step: int, max_tokens: int, seed: int
               ) -> list[list[list[int]]]:
    """Optimizer steps of about per_step questions, each a list of groups, in a seeded random order; the
    step holding the longest sequence goes first, so a memory problem shows at once."""
    rng = np.random.default_rng(seed)
    groups = make_groups(questions, list(range(len(questions))), per_seq, max_tokens, rng)
    groups = [groups[j] for j in rng.permutation(len(groups))]
    steps, cur, n = [], [], 0
    for g in groups:
        cur.append(g)
        n += len(g)
        if n >= per_step:
            steps.append(cur)
            cur, n = [], 0
    if cur:
        steps.append(cur)
    if steps:
        top = max(range(len(steps)), key=lambda s: max(packed_len([questions[i] for i in g]) for g in steps[s]))
        steps = [steps[top]] + steps[:top] + steps[top + 1:]
    return steps


# Model ------------------------------------------------------------------------------------------
def new_cost_head(hidden_size: int, device: str) -> Any:
    import torch

    head = torch.nn.Linear(hidden_size, 1).to(device=device, dtype=torch.float32)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def save_cost_head(head: Any, path: Path) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save({"state": head.state_dict(), "hidden_size": head.in_features}, tmp)
    tmp.replace(path)


def load_cost_head(path: Path, device: str) -> Any:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    head = new_cost_head(d["hidden_size"], device)
    head.load_state_dict(d["state"])
    return head


def forward(model: Any, head: Any, group: list[JQuestion], device: str, yes_no: tuple[int, int]) -> tuple[Any, Any]:
    """(yes-minus-no logit, cost) per candidate of the group, in order."""
    import torch

    ids, pos, sq, sc, read = pack(group)
    sq_t = torch.tensor(sq, device=device)
    sc_t = torch.tensor(sc, device=device)
    L = len(ids)
    causal = torch.ones(L, L, dtype=torch.bool, device=device).tril_()
    same_q = sq_t[None, :] == sq_t[:, None]
    own = (sc_t[None, :] == 0) | (sc_t[None, :] == sc_t[:, None])
    visible = (sq_t[None, :] == 0) | (same_q & own)
    mask = (causal & visible)[None, None]
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = inner.model(input_ids=torch.tensor([ids], device=device), position_ids=torch.tensor([pos], device=device),
                      attention_mask=mask, use_cache=False)
    h = out.last_hidden_state[0, torch.tensor(read, device=device)]
    logits = inner.lm_head(h)
    lp = (logits[:, yes_no[0]] - logits[:, yes_no[1]]).float()
    cost = head(h.float())[:, 0]
    return lp, cost


# Metrics ----------------------------------------------------------------------------------------
def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def judge_metrics(rows: list[dict[str, Any]], min_gap: float = 0.5) -> dict[str, Any]:
    """rows: per question {lp: [...], c: [...], labels: [...], costs: [...]} (labels and costs may hold None).
    Correctness: log loss, accuracy at p = 0.5, AUC, and top-1 (questions with a correct candidate whose
    highest-p candidate is correct). Cost: squared error on labelled candidates and pairwise order accuracy
    over pairs of labelled candidates of one question whose labels differ by at least min_gap (log2)."""
    lp, y, se, agree, n_pairs, top = [], [], [], 0, 0, []
    for r in rows:
        lab = [(a, b) for a, b in zip(r["lp"], r["labels"]) if b is not None]
        lp += [a for a, _ in lab]
        y += [b for _, b in lab]
        if any(b == 1 for b in r["labels"]):
            top.append(r["labels"][int(np.argmax(r["lp"]))] == 1)
        cs = [(c, t) for c, t in zip(r["c"], r["costs"]) if t is not None]
        se += [(c - t) ** 2 for c, t in cs]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if abs(cs[i][1] - cs[j][1]) >= min_gap:
                    n_pairs += 1
                    agree += (cs[i][0] - cs[j][0]) * (cs[i][1] - cs[j][1]) > 0
    lp_a, y_a = np.asarray(lp, dtype=float), np.asarray(y, dtype=float)
    logloss = float(np.mean(np.logaddexp(0, -lp_a) * y_a + np.logaddexp(0, lp_a) * (1 - y_a))) if len(y_a) else None
    return {"n_questions": len(rows), "n_candidates": len(y), "logloss": logloss,
            "accuracy": float(np.mean((lp_a > 0) == (y_a == 1))) if len(y_a) else None,
            "auc": auc(lp_a, y_a) if len(y_a) else None,
            "top1": float(np.mean(top)) if top else None, "n_top1": len(top),
            "cost_mse": float(np.mean(se)) if se else None, "n_cost": len(se),
            "cost_pair_acc": agree / n_pairs if n_pairs else None, "n_cost_pairs": n_pairs}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def build_questions(tok: Any, recs: list[dict[str, Any]], split: str, paths: Any, budget: int
                    ) -> tuple[list[JQuestion], dict[str, Any]]:
    """JQuestions from records {qid, cands: [{sql, label?, cost?}]} of one data split (train, dev, minidev)."""
    from effsql.data.bird import load_examples
    from effsql.gen.prompts import load_templates
    from effsql.gen.sample import schemas_for

    examples = {e.qid: e for e in load_examples(split, paths)}
    recs = [r for r in recs if r["cands"]]
    schemas = schemas_for(split, {examples[r["qid"]].db_id for r in recs}, paths)
    templates = load_templates()
    out, too_long, levels = [], 0, {}
    for r in recs:
        e = examples[r["qid"]]
        ids, level, n = question_prompts(tok, templates, e, schemas[e.db_id], [c["sql"] for c in r["cands"]], budget)
        if n > budget:
            too_long += 1
            continue
        levels[level] = levels.get(level, 0) + 1
        out.append(JQuestion(r["qid"], f"{e.db_id}|{level}", ids, [c.get("label") for c in r["cands"]],
                             [c.get("cost") for c in r["cands"]]))
    lens = [len(x) for q in out for x in q.ids]
    return out, {"n_questions": len(out), "n_candidates": len(lens), "n_skipped_too_long": too_long,
                 "prompt_levels": levels, "tokens_max": max(lens) if lens else 0}
