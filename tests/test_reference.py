"""Reference models: OmniSQL-style schema and prompts, Arctic's prefilled answer, predictions that
`effsql evaluate` scores, resume."""

import pytest

from effsql.config import load_config
from effsql.data.bird import load_examples
from effsql.data.schema import build_schema_cache, load_schema
from effsql.data.stage import stage
from effsql.evaluate import evaluate
from effsql.gen.backend import Completion
from effsql.gen.prompts import estimate_tokens, load_templates
from effsql.gen.reference import (
    ARCTIC_PREFILL,
    ident,
    omnisql_question,
    omnisql_schema,
    predict_reference,
    reference_prompt,
)
from effsql.io import read_json, read_jsonl

CFG = {"stage": {"disk_reserve_gb": 0}}
# Question -> final SQL. The reasoning holds a draft first: the last ```sql block is the answer.
FINAL = {"How many customers?": "SELECT count(*) FROM customers",
         "Total order amount?": "SELECT sum(amount) FROM orders",
         "How many students?": "SELECT count(*) FROM students",
         "Count the customers.": "SELECT count(*) FROM customers"}  # mini-dev


class RefBackend:
    model_id = "fake/reference"
    model_revision = "cafe"

    def __init__(self):
        self.prompts = []

    def count_tokens(self, messages):
        return estimate_tokens(messages)

    def generate(self, batch, params, *_):
        out = []
        for msgs in batch:
            self.prompts.append(msgs)
            q = msgs[-1]["content"].split("Question:\n", 1)[1].split("\n\nInstructions:", 1)[0].splitlines()[-1]
            text = (f"First try:\n```sql\nSELECT 1\n```\nBetter:</think>\n<answer>Count them.\n"
                    f"```sql\n{FINAL[q]}\n```\n</answer>")
            out.append([Completion(text, "stop", 40)])
        return out


@pytest.fixture
def dev(fake_bird):
    for split in ("dev", "minidev"):
        stage(split, paths=fake_bird, data_cfg=CFG)
        build_schema_cache(split, paths=fake_bird, data_cfg=CFG)
    return fake_bird


def test_schema_and_prompts(dev):
    p = dev
    assert ident("id") == "id" and ident("date") == "`date`" and ident("School Name") == "`School Name`"
    e = next(x for x in load_examples("dev", p) if x.question == "Total order amount?")
    assert omnisql_question(e) == "amount is in EUR\nTotal order amount?"
    schema = load_schema("dev", "shop", p)
    text = omnisql_schema(schema)
    assert text.startswith("CREATE TABLE ") and text.rstrip().endswith(");")
    assert "PRIMARY KEY (" in text and "example: [" in text and ",\n);" not in text
    assert "example: [" not in omnisql_schema(schema, values=False)

    t = load_templates()
    msgs, level, _ = reference_prompt("omnisql", t, e, schema, 10_000, estimate_tokens)
    assert len(msgs) == 1 and msgs[0]["content"].startswith("Task Overview:\nYou are a data science expert.")
    assert "Take a deep breath" in msgs[0]["content"] and level == "values"
    msgs, _, _ = reference_prompt("arctic", t, e, schema, 10_000, estimate_tokens)
    assert [m["role"] for m in msgs] == ["system", "user"] and "`<answer>` tags" in msgs[1]["content"]
    _, level, _ = reference_prompt("arctic", t, e, schema, 10, estimate_tokens)
    assert level == "no_values"                                   # too long: the values go first
    msgs, _, _ = reference_prompt("effsql", t, e, schema, 10_000, estimate_tokens)
    assert msgs[0]["content"] == t["system_base"]                 # our zero-shot prompt
    assert ARCTIC_PREFILL.endswith("<think>")


def test_predict_and_evaluate(dev):
    p = dev
    cfg = load_config("reference.yaml")
    fb = RefBackend()
    s = predict_reference("arctic7b", "dev", "ref_a", fb, paths=p, cfg=cfg)
    assert s["n_questions"] == 3 and s["style"] == "arctic" and s["n_cut_off"] == 0
    preds = {r["qid"]: r for r in read_jsonl(p.run_dir("ref_a") / "predictions_dev.jsonl")}
    assert {r["sql"] for r in preds.values()} == set(FINAL.values())   # the answer, not the draft
    assert all(r["extraction"] == "sql_block" and r["model"] == "fake/reference" for r in preds.values())

    n = len(fb.prompts)
    predict_reference("arctic7b", "dev", "ref_a", fb, paths=p, cfg=cfg)
    assert len(fb.prompts) == n                                         # resume: nothing is generated again

    ec = load_config("evaluate.yaml")
    ec["exec"]["workers"] = 1
    m = evaluate(str(p.run_dir("ref_a") / "predictions_dev.jsonl"), "dev", "ref_a_dev", paths=p, cfg=ec)
    assert m["overall"]["n"] == 3 and m["overall"]["ex"] == pytest.approx(100.0)

    predict_reference("arctic7b", "minidev", "ref_a", fb, paths=p, cfg=cfg)
    assert read_json(p.run_dir("ref_a") / "predict_stats_minidev.json")["n_questions"] == 3
    with pytest.raises(ValueError, match="dev and minidev"):
        predict_reference("arctic7b", "train", "ref_a", fb, paths=p, cfg=cfg)
