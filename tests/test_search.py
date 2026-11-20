"""Cost-guided search: rounds, feedback in the prompt, verification of answers, resume."""

import pytest
from fakes import FakeBackend, block

import effsql.gen.search as search_mod
from effsql.config import load_config
from effsql.data.schema import build_schema_cache
from effsql.data.stage import stage
from effsql.gen.parse import parse_run
from effsql.gen.sample import generate
from effsql.gen.search import attempt_note, read_targets, search
from effsql.io import read_json, read_jsonl
from effsql.pool.build import build_pool
from effsql.pool.diversity import diversity
from effsql.pool.quality import foreign_literals, quality
from effsql.pool.screen import screen

# Answers to the search prompt, by question: round 1, then round 2.
ANSWERS = {
    "How many movies?": [
        ["SELECT count(*) FROM movies WHERE id > 3",                          # wrong result
         "SELECT count(*) FROM movies WHERE id >= 0",                         # correct, but a guessed constant
         "SELECT count(*) FROM movies a JOIN movies b ON a.id = b.id",        # correct, pricier
         "SELECT max(id) + 1 FROM movies"],                                   # correct and cheaper: usable
        ["SELECT max(id) + 1 FROM movies", "SELECT count(*) FROM movies LIMIT 1",  # a first-row pick: never usable
         "SELEC nonsense", "SELECT 20"],
    ],
    "How many books?": [["SELECT count(id) FROM books"] * 4, ["SELECT count(*) FROM books"] * 4],
    "Max id?": [["SELECT id FROM movies ORDER BY id DESC LIMIT 1"] * 4, ["SELECT max(id) FROM movies"] * 4],
}


class SearchBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, batch, params, *_):
        out, rest = [], []
        for msgs, p in zip(batch, params):
            user = msgs[-1]["content"]
            if "needs fewer steps" not in user:
                rest.append((msgs, p))
                continue
            self.prompts.append(user)
            q = user.rsplit("Question: ", 1)[1].split("\n", 1)[0].strip()
            r = 2 if "Earlier attempts" in user else 1
            out.append([self._c(block(s)) for s in ANSWERS[q][r - 1][: p.n]])
        assert not rest, "only search prompts expected"
        return out

    @staticmethod
    def _c(text):
        from effsql.gen.backend import Completion

        return Completion(text, "stop", len(text) // 4)


@pytest.fixture
def pool(fake_bird):
    p = fake_bird
    stage("train", paths=p, data_cfg={"stage": {"disk_reserve_gb": 0}})
    build_schema_cache("train", paths=p, data_cfg={"stage": {"disk_reserve_gb": 0}})
    generate("train", "base", FakeBackend(), paths=p)
    parse_run("train", "base", paths=p)
    ec = load_config("evaluate.yaml")["exec"] | {"workers": 1}
    screen("train", "base", paths=p, exec_cfg=ec)
    diversity("train", "base", 0, paths=p)
    quality("train", "base", paths=p, exec_cfg=ec, workers=1)
    build_pool("train", "base", paths=p)
    return p


def test_foreign_literals():
    gold = "SELECT a FROM t JOIN m ON t.m = m.id WHERE m.medal_name = 'Gold'"
    assert foreign_literals("SELECT a FROM t WHERE t.medal_id = 1", gold, "Who won gold?", "") == ["1"]
    assert foreign_literals("SELECT a FROM t WHERE x > -3.5 LIMIT 1", gold, "above -3.5?", None) == []
    assert foreign_literals("SELECT a + 1 FROM t WHERE name LIKE 'gold'", gold, "q", "") == []      # in gold
    rng = "SELECT x FROM w WHERE d BETWEEN '2012-01-01' AND '2012-01-31'"
    assert foreign_literals(rng, "SELECT x FROM w WHERE d LIKE '%2012-01%'", "January 2012", "") == []
    assert foreign_literals("SELEC (((", gold, "q", "") == []
    assert foreign_literals("SELECT 20", gold, "How many?", "") == ["(reads no table)"]


def test_search_rounds(pool, monkeypatch):
    p = pool
    # The fixture's tables are tiny, so gold is about as cheap as anything: start the movies question
    # from the pricier self-join of the base pool, so that the search has something to improve.
    real, start = search_mod.base_state, {}

    def start_pricier(paths, base_run):
        best, known = real(paths, base_run)
        best["train_0"] = next(k for k in known["train_0"].values() if "JOIN" in k["sql"])
        assert best["train_0"]["usable"] and best["train_0"]["rel_cost"] > 1
        start["train_0"] = best["train_0"]["rel_cost"]
        return best, known

    monkeypatch.setattr(search_mod, "base_state", start_pricier)
    cfg = load_config("search.yaml") | {"base_run": "base", "rounds": 2, "stop_rel_cost": 0.0, "chunk_size": 2}
    fb = SearchBackend()
    summ = search("srch", fb, cfg=cfg, paths=p, workers=1)
    assert summ["n_questions"] == 3 and [r["round"] for r in summ["rounds"]] == [1, 2]
    assert summ["start"]["geomean_rel_cost"] > 1

    ex = {r["sql"]: r for r in read_jsonl(p.run_dir("srch") / "search_exec_train.jsonl")}
    assert not ex["SELECT count(*) FROM movies WHERE id > 3"]["correct"]
    guessed = ex["SELECT count(*) FROM movies WHERE id >= 0"]
    assert guessed["correct"] and guessed["foreign_literals"] == ["0"] and not guessed["usable"]
    cheap = ex["SELECT max(id) + 1 FROM movies"]
    assert cheap["usable"] and cheap["rel_cost"] < start["train_0"] and cheap["round"] == 1
    assert ex["SELECT count(*) FROM books"]["known"]                   # gold itself: taken from the pool, not run
    assert ex["SELEC nonsense"]["error_type"] == "syntax"
    hard = ex["SELECT 20"]
    assert hard["correct"] and hard["foreign_literals"] == ["(reads no table)"] and not hard["usable"]
    assert summ["rounds"][0]["n_questions_improved"] >= 1
    assert summ["rounds"][-1]["geomean_rel_cost"] < summ["start"]["geomean_rel_cost"]
    pick = ex["SELECT count(*) FROM movies LIMIT 1"]
    assert pick["correct"] and "unordered_limit" in pick["quality_flags"] and not pick["usable"]

    # The cheap targets: per question the cheapest usable query (here the search's) and the pool's own.
    t = read_targets(p, "srch")
    assert set(t) == {"train_0", "train_1", "train_2"}
    assert t["train_0"]["sql"] == "SELECT max(id) + 1 FROM movies" and t["train_0"]["source"] == "search:r1"
    assert "JOIN" in t["train_0"]["pool_sql"] and t["train_0"]["pool_rel_cost"] == start["train_0"]
    assert all(r["part"] == "train" for r in t.values())                # no SFT export here: no validation
    assert summ["final"]["train"]["by_source"]["search"] >= 1

    # Round 2 prompt for movies: the cheaper query is the reference, earlier attempts with reasons.
    r2 = [u for u in fb.prompts if "How many movies?" in u and "Earlier attempts" in u]
    assert len(r2) == 1
    u = r2[0]
    assert "SELECT max(id) + 1 FROM movies" in u.split("Reference query", 1)[1].split("```", 2)[1]
    assert "returns a different result" in u and "constants that are not in the question" in u
    assert "not cheaper than the reference" in u

    # Resume: nothing is generated or executed again.
    n_calls, n_exec = len(fb.calls), len(read_jsonl(p.run_dir("srch") / "search_exec_train.jsonl"))
    search("srch", fb, cfg=cfg, paths=p, workers=1)
    assert len(fb.calls) == n_calls
    assert len(read_jsonl(p.run_dir("srch") / "search_exec_train.jsonl")) == n_exec


def test_pilot_targets_fixed(pool):
    p = pool
    cfg = load_config("search.yaml") | {"base_run": "base", "rounds": 1}
    search("pilot", SearchBackend(), cfg=cfg, paths=p, limit=2, workers=1)
    t = read_json(p.run_dir("pilot") / "search_targets_train.json")
    assert t["n"] == 2 and t["excluded_dbs"] == []
    with pytest.raises(RuntimeError, match="another --run"):
        search("pilot", SearchBackend(), cfg=cfg, paths=p, limit=3, workers=1)


def test_attempt_note():
    assert "fails (syntax)" in attempt_note({"correct": False, "error_type": "syntax", "error_msg": "near x"}, 10)
    assert "arbitrary row" in attempt_note({"correct": True, "usable": False, "foreign_literals": [], "vm_steps": 5}, 10)
    assert "far more expensive" in attempt_note({"correct": True, "usable": True, "vm_steps": None}, 10)
