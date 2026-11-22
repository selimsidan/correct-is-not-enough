"""Answer-time execution, result voting and the latency helpers (judge/vote.py, judge/latency.py)."""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3

import pytest
from test_judge import judge_setup  # noqa: F401 (fixture)

from effsql.config import load_config
from effsql.exec.engine import compiles, preview_rows
from effsql.io import append_jsonl, read_json, read_jsonl
from effsql.judge.candidates import label_candidates
from effsql.judge.latency import judge_p, summarize
from effsql.judge.vote import exec_candidates, method_index, read_exec, vote


def cand(sql, n=1, greedy=False):
    return {"sql": sql, "n": n, "sources": ["greedy"] if greedy else ["base"]}


# Per question: (candidate, p). Expected picks (correct?) per method are worked out in the comments.
CASES = {
    "val": {"train_1": [  # books: 5 rows
        (cand("SELECT count(*) FROM books", 2, greedy=True), 0.3),   # correct, written twice
        (cand("SELECT count(*) FROM books WHERE id > 2"), 0.5),      # wrong
        (cand("SELECT count(*) FROM bookz"), 0.9)]},                 # does not compile
    "dev": {
        "dev_0": [
            (cand("SELECT count(*) FROM customers", greedy=True), 0.3),         # correct (10)
            (cand("SELECT count(id) FROM customers", 2), 0.35),                 # correct, same result
            (cand("SELECT count(*) FROM customers WHERE id > 1", 2), 0.6),      # wrong (8)
            (cand("SELECT count(*) FROM customerz"), 0.9)],                     # does not compile
        "dev_1": [(cand("SELECT sum(amount) FROM orders", greedy=True), 0.8)],
        "dev_2": [
            (cand("SELECT count(*) FROM studentz", greedy=True), 0.9),          # does not compile
            (cand("SELECT count(*) FROM students"), 0.1)],                      # correct
    },
}
DB = {"train_1": "books", "dev_0": "shop", "dev_1": "shop", "dev_2": "school"}


def write_inputs(p, run, judge):
    for split, qs in CASES.items():
        lines = [json.dumps({"qid": q, "db_id": DB[q], "cands": [c for c, _ in cs]}) for q, cs in qs.items()]
        (p.run_dir(run) / f"candidates_{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        append_jsonl(p.run_dir(run) / f"scores_{judge}_{split}.jsonl",
                     [{"qid": q, "lp": [0.0] * len(cs), "p": [x for _, x in cs], "c": [0.0] * len(cs)}
                      for q, cs in qs.items()])


def test_method_index_rules_and_fallbacks():
    cs = [cand("a", 1, greedy=True), cand("b", 2), cand("c", 2), cand("d")]
    p = [0.3, 0.35, 0.6, 0.9]
    res = [{"compiles": True, "ok": True, "result_hash": "x"}, {"compiles": True, "ok": True, "result_hash": "x"},
           {"compiles": True, "ok": True, "result_hash": "y"}, {"compiles": False, "ok": False}]
    assert method_index("judge", cs, p, res) == 3
    assert method_index("judge_compile", cs, p, res) == 2
    assert method_index("result_vote", cs, p, res) == 1           # x: 3 copies against 2; best p in x
    assert method_index("weighted_vote", cs, p, res) == 2         # x: 0.3 + 2 x 0.35 = 1.0 < 2 x 0.6
    assert method_index("judge", cs, None, res) == 0              # no scores: the generator's own answer
    assert method_index("result_vote", cs, p, None) == 3          # no execution: falls back to the judge
    assert method_index("result_vote", cs, p, [{"compiles": True, "ok": False}] * 4) == 3
    assert method_index("judge", [], p, res) is None
    # an empty result does not vote when another candidate returned rows (and still wins when none did)
    empty = [{"compiles": True, "ok": True, "result_hash": "e", "n_rows": 0}] * 3 + [
        {"compiles": True, "ok": True, "result_hash": "r", "n_rows": 2}]
    assert method_index("result_vote", cs, p, empty) == 2
    assert method_index("result_vote_nonempty", cs, p, empty) == 3
    assert method_index("weighted_vote_nonempty", cs, p, [dict(r, n_rows=0) for r in empty]) == 2
    with pytest.raises(ValueError):
        method_index("nope", cs, p, res)


def test_exec_and_vote(judge_setup):  # noqa: F811
    p, jcfg = judge_setup
    write_inputs(p, "cand", "J")
    exec_cfg = load_config("evaluate.yaml")["exec"]
    for split in CASES:
        label_candidates("cand", split, paths=p, jcfg=jcfg, exec_cfg={**exec_cfg, "workers": 1})
        exec_candidates("cand", split, paths=p, jcfg=jcfg, workers=1)
        exec_candidates("cand", split, paths=p, jcfg=jcfg, workers=1)  # resumed: nothing left
    ex = read_exec(p, "cand", "dev")
    assert len(read_jsonl(p.run_dir("cand") / "exec_dev.jsonl")) == 3
    r0 = ex["dev_0"]["results"]
    assert r0[0]["ok"] and r0[0]["compiles"] and r0[0]["preview"] == [[10]]
    assert r0[0]["result_hash"] == r0[1]["result_hash"] != r0[2]["result_hash"]
    assert not r0[3]["compiles"] and r0[3]["error_type"] == "syntax"

    res = vote("cand", "J", "V", paths=p, jcfg=jcfg, splits=("val", "dev"))
    assert res["chosen"] == "result_vote"  # val: judge 0, judge_compile 0, result_vote 100, weighted_vote 100
    dev = res["splits"]["dev"]
    assert dev["greedy"]["ex"] == pytest.approx(200 / 3)
    assert {m: round(o["ex"]) for m, o in dev["methods"].items()} == {
        "judge": 33, "judge_compile": 67, "result_vote": 100, "weighted_vote": 67,
        "result_vote_nonempty": 100, "weighted_vote_nonempty": 67}
    assert dev["oracle"]["ex"] == pytest.approx(100.0)
    preds = {r["qid"]: r["sql"] for r in read_jsonl(p.run_dir("V") / "predictions_dev.jsonl")}
    assert preds["dev_0"] == "SELECT count(id) FROM customers" and preds["dev_2"] == "SELECT count(*) FROM students"
    for run in ("V", "V_judge", "V_judge_compile", "V_result_vote", "V_weighted_vote"):
        assert read_json(p.run_dir(run) / "adapter_meta.json")["complete"]
        assert len(read_jsonl(p.run_dir(run) / "predictions_val.jsonl")) == 1
    text = (p.run_dir("V") / "selection.txt").read_text(encoding="utf-8")
    assert "method kept on validation: result_vote" in text and "answer-time execution" in text


def test_compile_check_and_preview(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    assert compiles(conn, "SELECT count(*) FROM t") and compiles(conn, "WITH x AS (SELECT id FROM t) SELECT * FROM x")
    assert not compiles(conn, "SELECT nope FROM t") and not compiles(conn, "SELEC 1") and not compiles(conn, "")
    assert not compiles(conn, "SELECT 1; SELECT 2")
    assert preview_rows([(b"\x01", "x" * 70, 1.5), (2, None, 0)], 1) == [["01", "x" * 60 + "...", 1.5]]


def test_latency_helpers():
    pr, miss = judge_p({9693: -0.1, 2152: -2.5, 5: -3.0}, (9693, 2152))
    assert pr == pytest.approx(1 / (1 + math.exp(-2.4))) and not miss
    pr, miss = judge_p({9693: -0.1, 5: -3.0}, (9693, 2152))  # "no" missing: counted at the lowest listed value
    assert pr == pytest.approx(1 / (1 + math.exp(-2.9))) and miss
    rows = [{"seconds": {s: float(i) for s in ("generate", "judge", "run_wait", "choose", "total")}, "n_candidates": 4}
            for i in range(10)]
    s = summarize(rows, warmup=2)
    assert s["n_questions"] == 8 and s["total"]["median"] == pytest.approx(5.5) and s["pct_over_3s"] == pytest.approx(75.0)


def test_result_text_and_prompts(judge_setup):  # noqa: F811
    from effsql.data.bird import load_examples
    from effsql.gen.prompts import load_templates
    from effsql.judge.model import judge_messages, result_text

    assert result_text({"compiles": False, "ok": False, "error_type": "syntax"}, 1.0).startswith("error: the query does not")
    assert result_text({"compiles": True, "ok": False, "error_type": "timeout"}, 1.0) == "no result: still running after 1 s"
    assert result_text({"compiles": True, "ok": True, "n_rows": 0, "preview": []}, 1.0) == "0 rows (empty result)"
    assert result_text({"compiles": True, "ok": True, "n_rows": 1, "preview": [[10]]}, 1.0) == "1 row:\n10"
    assert result_text({"ok": True, "n_rows": 7, "preview": [["a", None], ["b", 2.5]]}, 1.0) == \
        "7 rows, first 2:\na | NULL\nb | 2.5"
    assert result_text(None, 1.0) == "not run"
    p, _ = judge_setup
    e = load_examples("dev", p)[0]
    t = load_templates()
    with_res = judge_messages(t, e, "SCHEMA", "SELECT 1", "1 row:\n1")
    assert "Result when run: 1 row:\n1" in with_res[1]["content"] and "what the query returned" in with_res[0]["content"]
    assert "Result when run" not in judge_messages(t, e, "SCHEMA", "SELECT 1")[1]["content"]


def test_exec_judge_data(judge_setup):  # noqa: F811
    from effsql.judge.data import read_part
    from effsql.judge.vote import exec_judge_data

    p, jcfg = judge_setup
    exec_judge_data("jd", paths=p, jcfg=jcfg, workers=1)
    exec_judge_data("jd", paths=p, jcfg=jcfg, workers=1)  # resumed: nothing left
    for part in ("train", "val"):
        ex = read_exec(p, "jd", part)
        recs = read_part(p.run_dir("jd"), part)
        assert set(ex) == {r["qid"] for r in recs}
        assert all(len(ex[r["qid"]]["results"]) == len(r["cands"]) for r in recs)
    assert read_exec(p, "jd", "val")["train_1"]["results"][0]["preview"] == [[5]]


@pytest.mark.torch
@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft")),
                    reason="training libraries not installed")
def test_judge_with_results_trains_and_scores(judge_setup, tmp_path):  # noqa: F811
    from fakes import FakeBackend
    from test_sft import make_tiny_model

    from effsql.judge.candidates import generate_candidates
    from effsql.judge.score import read_scores, score_split
    from effsql.judge.train import load_judge, train_judge
    from effsql.judge.vote import exec_judge_data

    p, jcfg = judge_setup
    cfg = load_config("train.yaml")
    cfg["sft"].update(base_revision=None, lora={**cfg["sft"]["lora"], "r": 4, "alpha": 8})
    gen_cfg = load_config("generate.yaml")
    gen_cfg["model"] = {"id": str(make_tiny_model(tmp_path / "tiny")), "revision": None}
    kw = {"paths": p, "jcfg": jcfg, "cfg": cfg, "gen_cfg": gen_cfg, "device": "cpu"}
    with pytest.raises(FileNotFoundError):
        train_judge("JR", "jd", with_results=True, **kw)  # results not run yet
    exec_judge_data("jd", paths=p, jcfg=jcfg, workers=1)
    meta = train_judge("JR", "jd", with_results=True, **kw)
    assert meta["complete"] and meta["with_results"] and meta["timeout_s"] == jcfg["vote"]["timeout_s"]

    generate_candidates("cand", "dev", FakeBackend(), paths=p, jcfg=jcfg)
    model, head, tok, yes_no, jmeta = load_judge(p, "JR", "cpu", cfg=cfg, gen_cfg=gen_cfg)
    with pytest.raises(FileNotFoundError):
        score_split("JR", "cand", "dev", (model, head, tok, yes_no), paths=p, jcfg=jcfg, cfg=cfg, device="cpu",
                    with_results=True)
    exec_candidates("cand", "dev", paths=p, jcfg=jcfg, workers=1)
    score_split("JR", "cand", "dev", (model, head, tok, yes_no), paths=p, jcfg=jcfg, cfg=cfg, device="cpu",
                with_results=jmeta["with_results"])
    s = read_scores(p, "cand", "JR", "dev")
    assert set(s) == {"dev_0", "dev_1", "dev_2"} and all(0 <= x <= 1 for r in s.values() for x in r["p"])
