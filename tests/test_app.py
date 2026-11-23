"""The answer path (effsql/answer.py) and the demo (effsql/app.py) on the fixture, with a scripted engine:
generate, run, judge (reading results), choose; the BIRD download check; the page's data."""

import hashlib
import importlib.util
import zipfile

import pytest
from fakes import block

from effsql.answer import judge_p

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))
torch_only = [pytest.mark.torch, pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")]

# Question -> (greedy answer, samples). The greedy answer does not compile; most samples are right.
SCRIPT = {
    "How many customers?": ("SELECT count(*) FROM customerz",
                            ["SELECT count(*) FROM customers", "SELECT count(id) FROM customers",
                             "SELECT count(*) FROM customers WHERE id > 1"]),
    "Which customers are there?": ("SELECT name FROM customers", ["SELECT name FROM customers"]),
}


class FakeEngine:
    """Generation from SCRIPT; the judge says yes to a query that returned rows, no otherwise."""

    def __init__(self, tok):
        from effsql.judge.model import answer_token_ids

        self.tok = tok
        self.yes, self.no = answer_token_ids(tok)
        self.judged: list[str] = []

    def generate(self, prompts, params):
        out = []
        for text, p in zip(prompts, params):
            q = text.rsplit("Question: ", 1)[1].split("\n", 1)[0].split("<|im_end|>")[0].strip()
            greedy, samples = SCRIPT[q]
            sqls = [greedy] if p["n"] == 1 else (samples * p["n"])[: p["n"]]
            out.append([(block(s), "stop", 10) for s in sqls])
        return out

    def judge(self, ids):
        res = []
        for x in ids:
            text = self.tok.decode(x)
            self.judged.append(text)
            ok = "Result when run: " in text and " row" in text.rsplit("Result when run: ", 1)[1] \
                and "0 rows" not in text.rsplit("Result when run: ", 1)[1]
            res.append({self.yes: 0.0 if ok else -3.0, self.no: -1.0 if ok else 0.0})
        return res


def tiny_tokenizer(path):
    from test_sft import END_OF_TURN, TEMPLATE
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|im_start|>", END_OF_TURN, "<|endoftext|>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={k: i for i, k in enumerate(alphabet + specials)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token=END_OF_TURN, pad_token="<|endoftext|>",
                                   additional_special_tokens=specials, chat_template=TEMPLATE)


def test_judge_p():
    p, miss = judge_p({1: -0.1, 2: -2.0, 3: -5.0}, (1, 2))
    assert 0.5 < p < 1 and not miss
    p, miss = judge_p({1: -0.1, 3: -5.0}, (1, 2))  # "no" missing: it gets the lowest listed value
    assert miss and p > 0.99


def zip_bird_dev(src_dev, out, folder="dev_test"):
    with zipfile.ZipFile(out, "w") as z:
        for f in sorted(src_dev.iterdir()):
            z.write(f, f"{folder}/{f.name}")
        z.writestr(f"{folder}/._dev.json", b"junk")
    return out


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_get_bird_dev_checks_the_archive(fake_bird, tmp_path):
    from effsql.app import get_bird_dev

    archive = zip_bird_dev(fake_bird.drive_root / "dev", tmp_path / "bird.zip")
    bc = {"url": "unused", "sha256": "0" * 64, "folder": "dev_test", "files_sha256": {}}
    with pytest.raises(ValueError):
        get_bird_dev(bc, tmp_path / "a", archive)
    bc["sha256"] = sha(archive)
    bc["files_sha256"] = {"dev.json": sha(fake_bird.drive_root / "dev" / "dev.json")}
    dev = get_bird_dev(bc, tmp_path / "b", archive)
    assert (dev / "dev_databases.zip").exists() and not (dev / "._dev.json").exists()
    assert get_bird_dev({**bc, "sha256": "bad"}, tmp_path / "b", archive) == dev  # checked once, kept


@pytest.fixture
def demo(fake_bird, tmp_path, monkeypatch):
    from effsql.app import prepare
    from effsql.config import load_config

    archive = zip_bird_dev(fake_bird.drive_root / "dev", tmp_path / "bird.zip")
    cfg = load_config("app.yaml")
    cfg["bird"] = {"url": "unused", "sha256": sha(archive), "folder": "dev_test", "files_sha256": {}}
    runs = tmp_path / "runs"
    for name in (cfg["adapters"]["drive_generator"], cfg["adapters"]["drive_judge"]):
        (runs / name / "adapter").mkdir(parents=True)
        (runs / name / "adapter" / "adapter_config.json").write_text("{}")
    tok = tiny_tokenizer(tmp_path / "tok")
    engines = []

    def make_engine():
        engines.append(FakeEngine(tok))
        return engines[-1]

    app = prepare(cfg=cfg, root=tmp_path / "demo", adapters="drive", drive_runs=runs, bird_archive=archive,
                  make_engine=make_engine, workers=1)
    return app, engines[0]


@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")
def test_answer_reads_results_and_votes(demo):
    from effsql.judge.vote import method_index

    app, engine = demo
    assert app.db_ids == ["school", "shop"] and app.mode == "test engine"
    e = app.by_qid["dev_0"]
    a = app.system.answer(e, app.schema("shop"), "dev", method="weighted_vote", methods=["judge"])
    sqls = [c["sql"] for c in a.candidates]
    assert sqls[0] == "SELECT count(*) FROM customerz" and "greedy" in a.candidates[0]["sources"]
    assert len(sqls) == 4 and sum(c["n"] for c in a.candidates) == 17  # 1 greedy + 8 + 8
    assert len(a.results) == len(a.p) == 4 and not a.results[0]["compiles"]
    assert a.p[0] < 0.1 and a.p[1] > 0.5  # the judge read the error and the rows
    assert any("does not compile" in t for t in engine.judged) and all("Result when run: " in t for t in engine.judged)
    assert a.sql in ("SELECT count(*) FROM customers", "SELECT count(id) FROM customers")
    assert a.choice == method_index("weighted_vote", a.candidates, a.p, a.results) and "judge" in a.answers
    assert set(a.seconds) == {"generate", "judge", "run_wait", "choose", "total"}
    assert all(len(r.get("preview") or []) <= 5 for r in a.results)


@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="training libraries not installed")
def test_app_ask_with_and_without_reference(demo):
    from effsql.app import group_votes, schema_markdown

    app, _ = demo
    out = app.ask("shop", "How many customers?", "", "dev_0")
    assert out["reference"]["correct"] and out["result"]["ok"] and out["result"]["columns"]
    assert out["how"][0]["chosen"] == "yes" and out["n_written"] == 17 and out["n_distinct"] == 4
    own = app.ask("shop", "Which customers are there?", "", "dev_0")  # edited: no reference
    assert own["reference"] is None and own["qid"].startswith("demo_") and len(own["result"]["rows"]) >= 1
    with pytest.raises(ValueError):
        app.ask("shop", "   ")
    md = schema_markdown(app.schema("shop"))
    assert "**customers**" in md and "| " in md
    a = app.system.answer(app.by_qid["dev_0"], app.schema("shop"), "dev")
    v = group_votes(a)
    assert v[0] is None and v[1] is not None and v[1] == pytest.approx(v[2])  # same result, same group


def test_run_for_display_caps_and_errors(fake_bird):
    from effsql.app import run_for_display
    from effsql.data.stage import stage

    stage("dev", paths=fake_bird, data_cfg={"stage": {"disk_reserve_gb": 0}})
    db = fake_bird.local_db_path("dev", "shop")
    ok = run_for_display(db, "SELECT name FROM customers ORDER BY name", fake_bird, 5)
    assert ok["ok"] and ok["columns"] == ["name"] and ok["hash"]
    assert not run_for_display(db, "SELECT nope FROM customers", fake_bird, 5)["ok"]
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    r = run_for_display(db, slow, fake_bird, 0.2)
    assert not r["ok"] and "still running" in r["error"]
    assert run_for_display(db, "SELECT * FROM customers", fake_bird, 5, max_rows=1)["error"] == "too many rows"
