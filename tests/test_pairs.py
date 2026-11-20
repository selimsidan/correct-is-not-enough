"""Preference pairs of "cost first, then accuracy": chosen = the cheap target, rejected = the model's own
costlier or wrong answers (most frequent first), validation from the held-out databases, the ablations."""

import json

import pytest
from test_search import pool  # noqa: F401  (a fixture)

from effsql.io import append_jsonl, read_json, read_jsonl, write_json
from effsql.train.pairs import build_pairs
from effsql.train.select import format_cost_table, select_cost

CFG = {"efficiency_pairs": 3, "correctness_pairs": 1, "min_ratio": 1.1, "min_abs_steps": 50, "seed": 0}
MOVIES = "SELECT count(*) FROM movies"


def target(q, db, sql, steps, gold, source, pool_sql, pool_steps, pool_source):
    return {"qid": q, "db_id": db, "part": "x", "sql": sql, "vm_steps": steps, "gold_vm_steps": gold,
            "rel_cost": steps / gold, "source": source, "pool_sql": pool_sql, "pool_vm_steps": pool_steps,
            "pool_rel_cost": pool_steps / gold, "pool_source": pool_source, "search_version": 2}


@pytest.fixture
def runs(pool):  # noqa: F811
    p = pool
    rows = [target("train_0", "movies", "SELECT max(id) + 1 FROM movies", 100, 400, "search:r1", MOVIES, 400, "gold"),
            target("train_1", "books", "SELECT count(*) FROM books", 300, 300, "gold", "SELECT count(*) FROM books",
                   300, "gold"),
            target("train_2", "movies", "SELECT max(id) FROM movies", 200, 200, "gold", "SELECT max(id) FROM movies",
                   200, "gold")]
    (p.run_dir("T") / "cheap_targets_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    own = [  # the model's own answers (status, steps), as `rl --settings own_answers` scores them
        {"qid": "train_0", "sql": [MOVIES, MOVIES, MOVIES, "SELECT count(id) FROM movies",
                                   "SELECT count(*) FROM movies WHERE id > 3", None, "SELECT max(id) + 1 FROM movies",
                                   "SELECT count(*) FROM movies LIMIT 1"],
         "status": ["correct"] * 4 + ["wrong_result", "no_sql", "correct", "first_row_pick"],
         "steps": [400, 400, 400, 160, None, None, 100, None]},
        {"qid": "train_1", "sql": ["SELECT count(*) FROM books", "SELECT count(*) FROM books b JOIN books c ON b.id = c.id"],
         "status": ["correct", "correct"], "steps": [300, 900]},
        {"qid": "train_2", "sql": ["SELECT min(id) FROM movies"], "status": ["wrong_result"], "steps": [None]},
    ]
    append_jsonl(p.run_dir("O") / "iters" / "000" / "rewards.jsonl", own)
    write_json(p.run_dir("E") / "export_meta.json", {"pool_run": "base", "pool_sha256": "abc",
                                                     "holdout_db_ids": ["books"], "val_qids": ["train_1"]})
    return p


def test_pairs_from_own_answers(runs):
    p = runs
    m = build_pairs("T", "P", own_run="O", sft_export="E", pool_run="base", paths=p, cfg=CFG)
    tr = read_jsonl(p.run_dir("P") / "data" / "dpo_train.jsonl")
    eff = [r for r in tr if r["pair_type"] == "efficiency"]
    # the most frequent costlier answer first; the chosen query itself and the unparsed answer never rejected
    assert [r["rejected_sql"] for r in eff] == [MOVIES, "SELECT count(id) FROM movies"]
    assert [r["rejected_count"] for r in eff] == [3, 1] and eff[0]["cost_ratio"] == pytest.approx(4.0)
    assert all(r["chosen_sql"] == "SELECT max(id) + 1 FROM movies" and r["chosen_source"] == "search:r1" for r in tr)
    corr = [r for r in tr if r["pair_type"] == "correctness"]
    assert len(corr) == 1 and corr[0]["rejected_status"] in ("wrong_result", "first_row_pick")
    assert len({r["chosen_cand"] for r in tr}) == 1 and all(r["chosen_cand"] != r["rejected_cand"] for r in tr)
    s = m["stats"]["dpo_train"]
    assert s["questions"] == {"pairs": 1, "only_wrong": 1} and s["chosen_source"] == {"search": 1}

    val = read_jsonl(p.run_dir("P") / "data" / "dpo_val.jsonl")   # the held-out database
    assert [(r["qid"], r["pair_type"], r["cost_ratio"]) for r in val] == [("train_1", "efficiency", 3.0)]
    em = read_json(p.run_dir("P") / "export_meta.json")
    assert em["val_qids"] == ["train_1"] and em["pool_sha256"] == "abc" and em["own_run"] == "O"


def test_pair_ablations(runs):
    p = runs
    # chosen = gold: the model's count(id) is cheaper than gold, count(*) is gold: nothing to prefer
    m = build_pairs("T", "G", own_run="O", chosen="gold", sft_export="E", pool_run="base", paths=p, cfg=CFG)
    assert m["stats"]["dpo_train"]["n_examples"] == 0
    assert m["stats"]["dpo_train"]["questions"]["writes_cheap_form"] == 1
    m = build_pairs("T", "PO", own_run="O", chosen="pool", sft_export="E", pool_run="base", paths=p, cfg=CFG)
    assert m["stats"]["dpo_train"]["n_examples"] == 0          # the pool's best for movies is gold here
    # rejected = the pool's candidates (other models' queries), chosen = the search's target
    m = build_pairs("T", "OFF", rejected="pool", sft_export="E", pool_run="base", paths=p, cfg=CFG)
    recs = read_jsonl(p.run_dir("OFF") / "data" / "dpo_train.jsonl")
    assert recs and all(r["rejected_source"] == "pool" for r in recs)
    assert all(r["rejected_steps"] is None or r["rejected_steps"] >= 150 for r in recs if r["pair_type"] == "efficiency")
    with pytest.raises(ValueError, match="own-run"):
        build_pairs("T", "X", sft_export="E", paths=p, cfg=CFG)


def test_cost_selection_rule(tmp_path):
    from effsql.evaluate import EVAL_VERSION

    def write(name, correct, cost):
        append_jsonl(tmp_path / name / "eval_train.jsonl",
                     [{"qid": f"q{i}", "eval_version": EVAL_VERSION, "correct": c, "rel_cost": r if c else None}
                      for i, (c, r) in enumerate(zip(correct, cost))])
        return tmp_path / name

    n = 200
    ref = write("ref", [i < 150 for i in range(n)], [1.0] * n)
    cheap = write("cheap", [i < 146 for i in range(n)], [0.8] * n)                 # 4 fewer correct: allowed
    cheaper_wrong = write("cw", [i < 120 for i in range(n)], [0.5] * n)            # significantly less accurate
    pricier = write("pricier", [i < 155 for i in range(n)], [1.2] * n)
    res = select_cost("ref", {"ref": ref, "cheap": cheap, "cw": cheaper_wrong, "pricier": pricier})
    assert res["winner"] == "cheap" and not res["methods"]["cw"]["eligible"]
    assert res["methods"]["cheap"]["vs_ref"]["cost_ratio"] == pytest.approx(0.8)
    assert res["methods"]["cheap"]["vs_ref"]["cheaper"] == 146
    assert "winner: cheap" in format_cost_table(res)
    assert select_cost("ref", {"ref": ref, "pricier": pricier})["winner"] == "ref"  # none cheaper: keep the model
