"""Config loading. Configs are plain YAML files under `configs/` in the repo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_NAME = "correct-is-not-enough"
REPO_URL = f"https://github.com/selimsidan/{REPO_NAME}"

# src/effsql/config.py -> repo root is two levels above the package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Directory holding the YAML configs (env EFFSQL_CONFIG_DIR overrides the repo default)."""
    return Path(os.environ.get("EFFSQL_CONFIG_DIR", REPO_ROOT / "configs"))


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by file name (looked up in config_dir()) or by explicit path."""
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = config_dir() / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict[str, Any], path: Path) -> None:
    """Write a config dict as YAML (used to snapshot configs into a run directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
