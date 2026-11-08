from pathlib import Path

from effsql.config import REPO_ROOT
from effsql.io import append_jsonl, done_ids, read_jsonl


def test_resume_helpers_tolerate_truncated_line(tmp_path):
    p = tmp_path / "out.jsonl"
    append_jsonl(p, [{"qid": "a"}, {"qid": "b"}])
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"qid": "c", "trunc')  # simulated crash mid-write
    assert done_ids(p) == {"a", "b"}
    append_jsonl(p, [{"qid": "d"}])  # must start on a fresh line, not glue onto the broken one
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b", "d"]
    assert done_ids(tmp_path / "missing.jsonl") == set()


def test_no_en_or_em_dashes_in_repo():
    """Project rule: no en or em dashes anywhere in code, docs, configs or notebooks."""
    bad_chars = (chr(0x2013), chr(0x2014))
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info")
               for part in path.parts) or not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".ipynb", ".cfg", ""}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(c in text for c in bad_chars):
            offenders.append(str(Path(path).relative_to(REPO_ROOT)))
    assert not offenders, offenders


def test_notebooks_share_setup_cell_and_have_no_outputs():
    import json

    setup = (REPO_ROOT / "notebooks" / "setup_cell.py").read_text(encoding="utf-8").strip()
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        first = "".join(code_cells[0]["source"]).strip().replace("GPU = True", "GPU = False")
        assert first == setup, f"{nb_path.name}: setup cell differs from notebooks/setup_cell.py"
        assert all(not c.get("outputs") for c in code_cells), f"{nb_path.name} has outputs"
