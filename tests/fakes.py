"""A scripted stand-in for the vLLM backend, so the pipeline runs on CPU in tests."""

from __future__ import annotations

from effsql.gen.backend import Completion, GenParams
from effsql.gen.prompts import estimate_tokens


def block(sql: str) -> str:
    return f"Here is the query:\n```sql\n{sql}\n```"


# Question text -> (greedy answer, sampled answers). The fixture's train questions.
SCRIPT = {
    "How many movies?": (
        block("SELECT count(*) FROM movies"),
        [block("select COUNT(*) from movies;"),                                          # duplicate of greedy
         block("SELECT count(*) FROM movies a JOIN movies b ON a.title = b.title"),     # correct, far pricier
         block("SELECT count(*) FROM movies WHERE id > 3"),                             # wrong result
         "I am not sure."],                                                             # no SQL
    ),
    "How many books?": (block("SELECT count(*) FROM books"), [block("SELECT count(*) FROM books")] * 4),
    "Max id?": (block("SELECT max(id) FROM movies"), [block("SELECT max(id) FROM movies")] * 4),
    # dev questions (predict)
    "How many customers?": (block("SELECT count(*) FROM customers"), []),
    "Total order amount?": (block("SELECT sum(amount) FROM orders"), []),
    "How many students?": (block("SELECT count(*) FROM studentz"), []),
}
REWRITES = {
    "optimized": "SELECT count(id) FROM books",
    "cte": "WITH x AS (SELECT a.id FROM books a JOIN books b ON a.title <= b.title) SELECT count(DISTINCT id) FROM x",
}


class FakeBackend:
    model_id = "fake/model"
    model_revision = "deadbeef"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def count_tokens(self, messages):
        return estimate_tokens(messages)

    def generate(self, batch, params: list[GenParams]):
        self.calls.append(len(batch))
        out = []
        for msgs, p in zip(batch, params):
            user = msgs[-1]["content"]
            if "Rewrite the reference query" in user:
                style = user.rsplit("in this style. ", 1)[1].split(":", 1)[0]
                texts = [block(REWRITES.get(style, "SELECT 1"))]
            else:
                q = user.rsplit("Question: ", 1)[1].split("\n", 1)[0].strip()  # the tagged prompt adds a line after it
                greedy, samples = SCRIPT[q]
                texts = [greedy] if p.n == 1 else (samples + [greedy] * p.n)[: p.n]
            out.append([Completion(t, "stop", len(t) // 4) for t in texts])
        return out
