"""The ablation tables (effsql/ablate.py) on the fixture: choices on subsets of the samples, the sample curve's
end points, the verifier table, and the zero-shot validation run."""

import pytest
from fakes import block
from test_judge import judge_setup  # noqa: F401 (fixture)
from test_vote import CASES, write_inputs

from effsql.config import load_config
from effsql.io import append_jsonl, read_json, read_jsonl, write_json
from effsql.judge.candidates import label_candidates
from effsql.judge.vote import exec_candidates

# Per question: the sampled answers of each prompt (candidate positions in CASES; the greedy answer is
# written once by the greedy prompt), so that the counts n of CASES are what the samples give.
SLOTS = {"train_1": {"base": [0, 1], "efficient": [2]},
         "dev_0": {"base": [1, 2, 3], "efficient": [1, 2]},
         "dev_1": {"base": [], "efficient": []},
         "dev_2": {"base": [1], "efficient": []}}


@pytest.fixture
def abl(judge_setup):  # noqa: F811
    from effsql.ablate import Split

    p, jcfg = judge_setup
    for judge in ("judge_v1", "judge_v2"):
        write_inputs(p, "cand", judge)
    exec_cfg = load_config("evaluate.yaml")["exec"]
    exec_cfg["workers"] = 1
    for split, qs in CASES.items():
        exec_candidates("cand", split, paths=p, jcfg=jcfg, workers=1)
        label_candidates("cand", split, paths=p, jcfg=jcfg, exec_cfg=exec_cfg)
        raw = []
        for q, cs in qs.items():
            for v, idx in SLOTS[q].items():
                comps = [{"text": block(cs[i][0]["sql"]), "finish_reason": "stop"} for i in idx]
                raw.append({"key": f"{q}|{v}", "qid": q, "variant": v, "completions": comps, "gen_version": 1})
        append_jsonl(p.run_dir("cand") / f"gen_raw_{split}.jsonl", raw)
    data = {s: Split(p, "cand", s, ["judge_v1", "judge_v2", "judge_missing"]) for s in CASES}
    return p, data


def test_counts_from_samples_equal_the_candidates(abl):
    from effsql.ablate import sample_slots

    p, data = abl
    for d in data.values():
        slots = sample_slots(p, d)
        for q in d.qids:
            n = [int("greedy" in c["sources"]) for c in d.cands[q]]
            for _, i in slots[q]:
                n[i] += 1
            assert n == [c["n"] for c in d.cands[q]]


def test_choices_curve_and_tables(abl):
    from effsql.ablate import (
        CURVE_METHODS,
        cap,
        cost_spread,
        learned,
        prompt_sources,
        sample_curve,
        verifiers,
    )

    p, data = abl
    d = data["dev"]
    assert set(d.p) == {"judge_v1", "judge_v2"}  # a judge without scores is left out
    # dev_0: greedy correct; most written: greedy (n 1) loses to the n 2 ones, the first is correct
    assert d.ok("greedy") == [True, True, False]
    assert d.ok("oracle") == [True, True, True]
    assert d.ok("judge", "judge_v2") == [False, True, False]  # highest p: a query that does not compile
    assert d.ok("judge_compile", "judge_v2") == [False, True, True]
    assert d.ok("exec_vote") == [True, True, True]  # dev_0: 1 + 2 copies of the correct result beat 2
    # the same choice on an explicit subset with the full counts
    q = "dev_0"
    assert d.pick("exec_vote", None, q, [0, 1, 2], [1, 2, 2]) == d.pick("exec_vote", None, q)

    curve = sample_curve(p, data, draws=3)
    for split, s in curve.items():
        dd = data[split]
        for m, j, label in CURVE_METHODS:
            if j is None or j in dd.p:
                assert s["ex"][label][0] == pytest.approx(100 * sum(dd.ok("greedy")) / len(dd.qids))
                assert s["ex"][label][16] == pytest.approx(100 * sum(dd.ok(m, j)) / len(dd.qids))

    ver = verifiers(data)
    rows = {r["label"]: r for r in ver["dev"]["rows"]}
    assert rows["verifier v2 (reads results)"]["auc"] is not None and "vote_vs_final" in rows["verifier v1 (SQL only)"]
    assert "untrained verifier (B5)" not in rows

    pr = prompt_sources(p, data, draws=2)
    assert set(pr["dev"]) == {"normal prompt, 8 samples", "efficient prompt, 8 samples", "both prompts, 4 + 4 samples",
                              "both prompts, 8 + 8 samples (final)"}
    le = learned(data)["dev"]["judge_v2"]
    assert le["by_result"]["does not compile"]["n"] == 2 and sum(b["n"] for b in le["calibration"]) == 7
    assert cap(data)["dev"]["n_lost"] == 0
    steps = [s for s in data["dev"].steps["dev_0"] if s]
    assert cost_spread(data)["dev"]["n_two_or_more_correct"] == int(len(steps) >= 2)


def test_ladder_without_evaluations_and_report_text(abl):
    from effsql.ablate import format_report, ladder, verifiers

    p, data = abl
    lad = ladder(p, data, {})
    labels = [r["label"] for r in lad["dev"]["rows"]]
    assert labels[-1] == "+ weighted vote (final system)" and "vs_previous" in lad["dev"]["rows"][1]
    rep = {"ladder": lad, "verifiers": verifiers(data), "samples": {}, "prompts": {}, "difficulty": {}, "learned": {},
           "cap": {}, "cost": {}, "examples": []}
    assert "COMPONENT LADDER" in format_report(rep)


def test_zero_shot_val(judge_setup):  # noqa: F811
    from effsql.ablate import zero_shot_val

    p, _ = judge_setup
    write_json(p.run_dir("sft_x") / "export_meta.json", {"val_qids": ["train_1"]})
    append_jsonl(p.run_dir("pool_x") / "generations_train.jsonl", [
        {"key": "train_1|base|greedy", "completions": [{"text": block("SELECT count(*) FROM books"), "finish_reason": "stop"}]},
        {"key": "train_1|efficient|greedy", "completions": [{"text": block("SELECT 1"), "finish_reason": "stop"}]}])
    meta = zero_shot_val("zs", paths=p, pool_run="pool_x", export_run="sft_x")
    assert meta["complete"] and read_json(p.run_dir("zs") / "adapter_meta.json")["method"] == "zero-shot-base"
    preds = read_jsonl(p.run_dir("zs") / "predictions_val.jsonl")
    assert len(preds) == 1 and preds[0]["sql"] == "SELECT count(*) FROM books"
    append_jsonl(p.run_dir("pool_empty") / "generations_train.jsonl", [{"key": "train_0|base|greedy", "completions": []}])
    with pytest.raises(RuntimeError):
        zero_shot_val("zs2", paths=p, pool_run="pool_empty", export_run="sft_x")
