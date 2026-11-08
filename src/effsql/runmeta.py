"""Run metadata: git commit, machine, package versions and data file hashes.

Written into every run directory, because timings are only comparable on the same runtime type
and results are only reproducible with the same code and data.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from effsql.config import REPO_ROOT
from effsql.io import write_json

_PACKAGES = ("effsql", "numpy", "pandas", "pyarrow", "scipy", "sqlglot", "pyyaml", "vllm", "torch")


def git_commit(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def cpu_info() -> dict[str, Any]:
    """CPU model and core count (Linux reads /proc/cpuinfo; elsewhere falls back to platform)."""
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    if model is None and sys.platform == "darwin":
        try:
            model = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip() or None
        except FileNotFoundError:
            pass
    return {
        "cpu_model": model or platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run_meta(data_files: list[Path] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect metadata; data_files are hashed so qids like train_<index> stay well defined."""
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "machine": cpu_info(),
        "packages": package_versions(),
        "argv": sys.argv,
    }
    if data_files:
        meta["data_sha256"] = {str(p): sha256_file(p) for p in data_files if p.exists()}
    if extra:
        meta.update(extra)
    return meta


def write_run_meta(run_dir: Path, **kwargs: Any) -> dict[str, Any]:
    meta = run_meta(**kwargs)
    write_json(run_dir / "run_meta.json", meta)
    return meta
