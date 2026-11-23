"""The demo (notebook 90): ask the final system your own question about a BIRD dev database.

`prepare` gets everything the system needs onto the machine:
  * BIRD dev from BIRD's own site, checked against the hashes in configs/app.yaml (the same files every
    experiment used); the databases are staged locally and their schemas cached, exactly as for evaluation,
    so the prompts are the ones the model was trained and tested with;
  * the two adapters, from the public Hugging Face repository (or from Drive, before publishing);
  * the engine: 16 bit on a GPU with at least runtime.full_min_gpu_gb (L4, A100), the reported system;
    4 bit on smaller GPUs (a free Colab T4), where answers can differ slightly.

`App.ask` answers one question through effsql.answer.System (the timed path of the latency measurement)
and returns what the page shows: the SQL, its result, how the answer was chosen (every candidate, how often
it was written, what it returned, the verifier's score, its result group's vote) and the time per stage.
For a BIRD example question it also runs the reference SQL and says whether the result matches, as the
evaluation does. `build_ui` is the Gradio page.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from effsql.config import load_config

log = logging.getLogger(__name__)
SPLIT = "dev"


# Preparation -------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_bird_dev(bc: dict[str, Any], data_root: Path, archive: Path | None = None) -> Path:
    """data_root/dev/ with BIRD dev's files, downloaded once and checked. archive: a local copy of the zip."""
    dev = data_root / "dev"
    if (dev / ".checked").exists():
        return dev
    data_root.mkdir(parents=True, exist_ok=True)
    zpath = archive or data_root / "bird_dev.zip"
    if archive is None and not (zpath.exists() and sha256_file(zpath) == bc["sha256"]):
        log.info("downloading BIRD dev (about 350 MB) from %s", bc["url"])
        tmp = zpath.with_suffix(".part")
        with urllib.request.urlopen(bc["url"]) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(zpath)
    if sha256_file(zpath) != bc["sha256"]:
        raise ValueError(f"{zpath} is not the BIRD dev archive this project used (sha256 differs)")
    dev.mkdir(parents=True, exist_ok=True)
    prefix = bc["folder"].rstrip("/") + "/"
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename
            if not name.startswith(prefix) or info.is_dir() or Path(name).name.startswith("._"):
                continue
            with z.open(info) as src, open(dev / name[len(prefix):], "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    for name, want in bc.get("files_sha256", {}).items():
        if sha256_file(dev / name) != want:
            raise ValueError(f"BIRD dev file {name} differs from the one this project used")
    (dev / ".checked").write_text("ok\n", encoding="utf-8")
    if archive is None:
        zpath.unlink()
    return dev


def gpu_mode(full_min_gb: float) -> tuple[str, bool]:
    """(description, quantize) for the GPU of this machine."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU: Runtime > Change runtime type > T4 GPU (free), L4 or A100, then run again.")
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 2**30
    if gb >= full_min_gb:
        return f"{props.name} ({gb:.0f} GB): the model in 16 bit, exactly the reported system", False
    return (f"{props.name} ({gb:.0f} GB): the base model in 4 bit to fit this GPU; answers can differ slightly "
            "from the reported system"), True


def get_adapters(ac: dict[str, Any], root: Path, source: str, drive_runs: Path | None = None) -> tuple[Path, Path]:
    """(generator, judge) adapter folders. source: hf (the public repository) or drive (runs on Drive)."""
    if source == "drive":
        if drive_runs is None:
            raise ValueError("drive_runs is needed to read the adapters from Drive")
        return drive_runs / ac["drive_generator"] / "adapter", drive_runs / ac["drive_judge"] / "adapter"
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(repo_id=ac["repo"], revision=ac.get("revision"), local_dir=root / "adapters"))
    return d / ac["generator"], d / ac["judge"]


# Answering ---------------------------------------------------------------------------------------
def run_for_display(db_path: Path, sql: str, paths: Any, timeout_s: float, max_rows: int = 100000
                    ) -> dict[str, Any]:
    """Columns, rows (up to max_rows) and the evaluation's result hash of one query, or its error."""
    from effsql.exec.engine import connect_readonly
    from effsql.exec.results import result_hash

    conn = connect_readonly(db_path, paths)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return {"ok": False, "error": "too many rows", "columns": cols, "rows": rows[:max_rows]}
        return {"ok": True, "columns": cols, "rows": rows, "hash": result_hash(rows)}
    except sqlite3.Error as e:
        msg = "stopped: still running" if time.monotonic() > deadline else str(e)
        return {"ok": False, "error": msg, "columns": [], "rows": []}
    finally:
        conn.close()


def group_votes(a: Any) -> list[float | None]:
    """Per candidate, the weighted vote of its result group (sum of times written x p over the candidates
    with the same result); None for a candidate that returned nothing."""
    tot: dict[str, float] = {}
    for c, r, p in zip(a.candidates, a.results, a.p):
        if r.get("ok"):
            tot[r["result_hash"]] = tot.get(r["result_hash"], 0.0) + c["n"] * p
    return [tot.get(r["result_hash"]) if r.get("ok") else None for r in a.results]


def one_line(text: str, n: int = 140) -> str:
    s = " / ".join(x.strip() for x in text.splitlines() if x.strip())
    return s if len(s) <= n else s[: n - 3] + "..."


@dataclass
class App:
    cfg: dict[str, Any]
    paths: Any
    system: Any
    mode: str
    examples: dict[str, list[Any]]      # db_id -> BIRD dev examples shown for it
    by_qid: dict[str, Any]

    @property
    def db_ids(self) -> list[str]:
        return sorted(self.examples)

    def schema(self, db_id: str) -> dict[str, Any]:
        from effsql.data.schema import load_schema

        return load_schema(SPLIT, db_id, self.paths)

    def ask(self, db_id: str, question: str, hint: str = "", qid: str | None = None) -> dict[str, Any]:
        from effsql.data.bird import Example
        from effsql.judge.model import result_text

        question, hint = question.strip(), (hint or "").strip()
        if not question:
            raise ValueError("Type a question first.")
        ex = self.by_qid.get(qid) if qid else None
        if ex is not None and (ex.db_id != db_id or ex.question != question or (ex.evidence or "") != hint):
            ex = None  # edited: no longer the BIRD question, so no reference answer
        e = ex or Example(qid="demo_" + hashlib.sha256(f"{db_id}|{question}|{hint}".encode()).hexdigest()[:12],
                          bird_question_id=None, split=SPLIT, db_id=db_id, question=question, evidence=hint,
                          gold_sql="", difficulty=None)
        a = self.system.answer(e, self.schema(db_id), SPLIT, method=self.cfg["method"])
        rc = self.cfg["runtime"]
        db_path = self.paths.local_db_path(SPLIT, db_id)
        shown = run_for_display(db_path, a.sql, self.paths, rc["display_timeout_s"]) if a.sql else \
            {"ok": False, "error": "no query", "columns": [], "rows": []}
        votes = group_votes(a)
        timeout = self.system.vc["timeout_s"]
        order = sorted(range(len(a.candidates)), key=lambda i: (i != a.choice, -(votes[i] or -1), -a.p[i]))
        how = [{"chosen": "yes" if i == a.choice else "", "verifier p": round(a.p[i], 3),
                "group vote": None if votes[i] is None else round(votes[i], 2), "written": a.candidates[i]["n"],
                "result": one_line(result_text(a.results[i], timeout)), "SQL": " ".join(a.candidates[i]["sql"].split())}
               for i in order]
        out: dict[str, Any] = {"qid": e.qid, "sql": a.sql, "result": shown, "how": how, "seconds": a.seconds,
                               "n_written": sum(c["n"] for c in a.candidates), "n_distinct": len(a.candidates),
                               "reference": None}
        if ex is not None:
            gold = run_for_display(db_path, ex.gold_sql, self.paths, rc["display_timeout_s"])
            out["reference"] = {"sql": ex.gold_sql, "result": gold,
                                "correct": bool(shown.get("ok") and gold.get("ok") and shown["hash"] == gold["hash"])}
        return out


def prepare(config: str = "app.yaml", *, root: str | Path | None = None, adapters: str = "hf",
            drive_runs: str | Path | None = None, bird_archive: str | Path | None = None,
            cfg: dict[str, Any] | None = None, make_engine: Any = None, workers: int | None = None) -> App:
    """Everything for the demo; see the module docstring. adapters: hf (the public repository) or drive
    (drive_runs: the runs folder on Drive). bird_archive: a local copy of BIRD's dev zip, instead of
    downloading it. cfg, make_engine and workers are for tests (a fake engine, no worker processes)."""
    from effsql.answer import System, VLLMEngine
    from effsql.data.bird import load_examples
    from effsql.data.schema import build_schema_cache
    from effsql.data.stage import stage
    from effsql.paths import load_paths
    from effsql.train.sft import base_model_cfg

    cfg = cfg or load_config(config)
    rc = cfg["runtime"]
    root = Path(root or rc["root"])
    data_root = root / "data"
    get_bird_dev(cfg["bird"], data_root, Path(bird_archive) if bird_archive else None)
    os.environ["EFFSQL_DRIVE_ROOT"] = str(data_root.resolve())
    os.environ["EFFSQL_LOCAL_ROOT"] = str((root / "local").resolve())
    paths = load_paths()
    stage(SPLIT, paths=paths)
    build_schema_cache(SPLIT, paths=paths)
    gen_dir, judge_dir = get_adapters(cfg["adapters"], root, adapters, Path(drive_runs) if drive_runs else None)
    for d in (gen_dir, judge_dir):
        if not (d / "adapter_config.json").exists():
            raise FileNotFoundError(f"no adapter in {d}")

    if make_engine is None:
        mode, quantize = gpu_mode(rc["full_min_gpu_gb"])
        model_cfg = base_model_cfg(load_config("train.yaml"), load_config("generate.yaml"))

        def make_engine() -> Any:
            return VLLMEngine(model_cfg, gen_dir, judge_dir, gpu_memory_utilization=rc["gpu_memory_utilization"],
                              quantize=quantize, max_model_len=rc["quantized_max_model_len"] if quantize else None)
    else:
        mode = "test engine"
    gp = cfg["generator_prompt"]
    system = System(make_engine, paths=paths, arm=gp["arm"], tag=gp.get("tag"),
                    reads_results=cfg["judge_reads_results"], workers=workers)
    examples: dict[str, list[Any]] = {}
    all_ex = load_examples(SPLIT, paths)
    rng = np.random.default_rng(0)
    for db in sorted({e.db_id for e in all_ex}):
        mine = [e for e in all_ex if e.db_id == db]
        examples[db] = [mine[i] for i in sorted(rng.permutation(len(mine))[: rc["examples_per_db"]])]
    return App(cfg, paths, system, mode, examples, {e.qid: e for v in examples.values() for e in v})


# The page ----------------------------------------------------------------------------------------
INTRO = """### Ask a question about a database

The system is one 7B model (Qwen2.5-Coder-7B-Instruct) with two small adapters. For your question it
1. **writes 17 queries**: its best guess plus 16 samples from two prompts (duplicates are merged);
2. **runs each one** on the database, stopping any that is still running after 1 second;
3. **checks each one** with a second adapter, the *verifier*, which reads the query **and what it returned**
   and scores how likely it is to be correct (p);
4. **answers** with the result that has the highest total score: each query adds p for every time it was
   written (the *group vote*).

Pick a database, try an example question or type your own. A hint is optional: BIRD questions come with one
(for example *"Italian refers to nationality = 'Italian'"*), and it helps when the question uses words that
differ from the values stored in the database."""


def schema_markdown(schema: dict[str, Any], max_cols: int = 30, max_chars: int = 40) -> str:
    """Every table with its size, columns (and types) and its first example rows, from the cached schema."""
    def cell(v: Any) -> str:
        x = "NULL" if v is None else str(v).replace("|", "/").replace("\n", " ")
        return x if len(x) <= max_chars else x[: max_chars - 3] + "..."

    lines = []
    for t in schema.get("tables", []):
        cols = t.get("columns", [])
        n = t.get("row_count")
        lines.append(f"**{t.get('name')}**" + (f" ({n:,} rows)" if isinstance(n, int) else ""))
        names = [c["name"] for c in cols][:max_cols]
        types = {c["name"]: (c.get("type") or "") for c in cols}
        lines.append("| " + " | ".join(f"{c} ({types[c]})" if types[c] else c for c in names) + " |")
        lines.append("|" + "---|" * len(names))
        idx = {c: i for i, c in enumerate(t.get("sample_columns") or [])}
        for row in t.get("sample_rows") or []:
            lines.append("| " + " | ".join(cell(row[idx[c]]) if c in idx else "" for c in names) + " |")
        if len(cols) > max_cols:
            lines.append(f"... and {len(cols) - max_cols} more columns")
        lines.append("")
    return "\n".join(lines)


def build_ui(app: App) -> Any:
    import gradio as gr
    import pandas as pd

    def table(res: dict[str, Any], n: int) -> pd.DataFrame:
        rows = res.get("rows") or []
        cols = res.get("columns") or [f"column {i + 1}" for i in range(len(rows[0]) if rows else 0)]
        return pd.DataFrame([list(r) for r in rows[:n]], columns=cols)

    def pick_db(db: str):
        ex = app.examples[db]
        return (gr.update(choices=[e.question for e in ex], value=None), schema_markdown(app.schema(db)), "", "", "")

    def pick_example(db: str, q: str | None):
        for e in app.examples[db]:
            if e.question == q:
                return e.question, e.evidence or "", e.qid
        return gr.update(), gr.update(), ""

    def ask(db: str, q: str, hint: str, qid: str):
        try:
            out = app.ask(db, q, hint, qid or None)
        except ValueError as err:
            raise gr.Error(str(err)) from err
        n = app.cfg["runtime"]["display_rows"]
        res = out["result"]
        status = (f"{len(res['rows'])} row(s)" + (f", first {n} shown" if len(res["rows"]) > n else "")) if res["ok"] \
            else f"no result: {res['error']}"
        s = out["seconds"]
        info = (f"**Answer:** {status}.  \n{out['n_written']} queries written, {out['n_distinct']} different.  \n"
                f"Time: {s['total']:.1f} s in all (writing {s['generate']:.1f} s, running {s['run_wait']:.1f} s, "
                f"checking {s['judge']:.1f} s).")
        ref = out["reference"]
        if ref:
            info += ("  \n**Reference (BIRD):** the answer is " + ("**correct**" if ref["correct"] else "**not correct**")
                     + " (same rows as the reference query, in any order).")
        ref_sql = ref["sql"] if ref else ""
        ref_table = table(ref["result"], n) if ref else pd.DataFrame()
        return out["sql"], table(res, n), info, pd.DataFrame(out["how"]), ref_sql, ref_table

    with gr.Blocks(title="Generate, then judge: text to SQL") as page:
        gr.Markdown(INTRO)
        gr.Markdown(f"*Running on {app.mode}.*")
        with gr.Row():
            with gr.Column(scale=2):
                db = gr.Dropdown(app.db_ids, value=app.db_ids[0], label="Database")
                example = gr.Dropdown([e.question for e in app.examples[app.db_ids[0]]], label="Example questions (BIRD dev)")
                question = gr.Textbox(label="Your question", lines=2)
                hint = gr.Textbox(label="Hint (optional)", lines=1)
                qid = gr.State("")
                go = gr.Button("Answer", variant="primary")
            with gr.Column(scale=3), gr.Accordion("Tables and columns", open=False):
                schema = gr.Markdown(schema_markdown(app.schema(app.db_ids[0])))
        sql = gr.Code(label="SQL chosen", language="sql")
        info = gr.Markdown()
        result = gr.Dataframe(label="Result", wrap=True)
        with gr.Accordion("How it chose: every query it wrote", open=True):
            how = gr.Dataframe(wrap=True)
        with gr.Accordion("Reference answer (example questions only)", open=False):
            ref_sql = gr.Code(language="sql")
            ref_table = gr.Dataframe(wrap=True)
        db.change(pick_db, db, [example, schema, question, hint, qid])
        example.change(pick_example, [db, example], [question, hint, qid])
        question.input(lambda: "", None, qid)
        hint.input(lambda: "", None, qid)
        go.click(ask, [db, question, hint, qid], [sql, result, info, how, ref_sql, ref_table])
    return page


def launch(app: App, share: bool = True) -> Any:
    page = build_ui(app)
    page.queue(default_concurrency_limit=1)  # one question at a time: one engine
    return page.launch(share=share)
