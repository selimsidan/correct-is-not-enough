"""Path resolution for Drive data, local staged databases and outputs.

Every path comes from configs/paths.yaml plus two env vars, so library code never hard-codes
a location:
  EFFSQL_DRIVE_ROOT  root of the BIRD data on Drive (default /content/drive/MyDrive/586)
  EFFSQL_LOCAL_ROOT  local staging root on the Colab disk (default /content/bird_local)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effsql.config import load_config

SPLITS = ("train", "dev", "minidev")
JUNK_PARTS = ("__MACOSX",)
JUNK_NAMES = (".DS_Store",)
_SAFE_DB_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def is_junk(member: str) -> bool:
    """True for macOS artefacts that must be ignored everywhere (__MACOSX, ._*, .DS_Store)."""
    parts = member.replace("\\", "/").split("/")
    if any(p in JUNK_PARTS for p in parts):
        return True
    name = parts[-1]
    return name in JUNK_NAMES or name.startswith("._")


def check_db_id(db_id: str) -> str:
    """Reject db_ids that could escape the staging directory when used in a path."""
    if not _SAFE_DB_ID.match(db_id):
        raise ValueError(f"Unexpected db_id {db_id!r}")
    return db_id


@dataclass
class Paths:
    """Resolved locations for one session."""

    drive_root: Path
    local_root: Path
    out_root: Path
    splits: dict[str, dict[str, str]]
    forbidden: list[Path] = field(default_factory=list)
    bird_dev_release: str = ""

    # Drive inputs -----------------------------------------------------------------------------
    def split_file(self, split: str, key: str) -> Path:
        """Absolute Drive path of one configured file of a split (questions, gold, tables, ...)."""
        spec = self._spec(split)
        if key not in spec:
            raise KeyError(f"Split {split!r} has no {key!r} entry in paths.yaml")
        path = self.drive_root / spec[key]
        self.assert_not_forbidden(path)
        return path

    def qid_mode(self, split: str) -> str:
        """How qids are formed: `question_id` (<split>_<question_id>) or `index` (<split>_<position>)."""
        mode = self._spec(split).get("qid", "question_id")
        if mode not in ("question_id", "index"):
            raise ValueError(f"Split {split!r}: qid must be question_id or index, got {mode!r}")
        return mode

    def db_split(self, split: str) -> str:
        """Split whose databases a split runs on (minidev -> dev)."""
        return self._spec(split).get("db_split", split)

    def db_zip(self, split: str) -> Path:
        """Zip holding the databases used by `split` (minidev resolves to the dev zip)."""
        return self.split_file(self.db_split(split), "db_zip")

    # Local staged databases -------------------------------------------------------------------
    def local_split_dir(self, split: str) -> Path:
        return self.local_root / self.db_split(split)

    def local_db_dir(self, split: str, db_id: str) -> Path:
        return self.local_split_dir(split) / check_db_id(db_id)

    def local_db_path(self, split: str, db_id: str) -> Path:
        return self.local_db_dir(split, db_id) / f"{db_id}.sqlite"

    def manifest_path(self, split: str) -> Path:
        return self.local_split_dir(split) / "manifest.json"

    # Outputs ----------------------------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.out_root / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.out_root / "runs"

    @property
    def results_csv(self) -> Path:
        return self.out_root / "results" / "results.csv"

    def schema_cache_path(self, split: str, db_id: str) -> Path:
        """Schema cache is per database, so minidev shares the dev cache."""
        return self.cache_dir / "schemas" / self.db_split(split) / f"{check_db_id(db_id)}.json"

    def run_dir(self, name: str, create: bool = True) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"Bad run name {name!r}")
        path = self.runs_dir / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # Guards -----------------------------------------------------------------------------------
    def assert_not_forbidden(self, path: Path) -> None:
        resolved = Path(os.path.abspath(path))
        for bad in self.forbidden:
            if resolved == bad or bad in resolved.parents:
                raise PermissionError(f"Refusing to read {path}: under forbidden path {bad}")

    def assert_local(self, path: Path) -> Path:
        """Refuse any database path that is not under local_root (never execute SQL on Drive)."""
        resolved = Path(os.path.abspath(path))
        drive_mount = Path("/content/drive")
        for remote in (self.drive_root, drive_mount):
            if resolved == remote or remote in resolved.parents:
                raise PermissionError(f"Refusing to open {path}: databases on Drive are never queried")
        if not (resolved == self.local_root or self.local_root in resolved.parents):
            raise PermissionError(f"Refusing to open {path}: not under local root {self.local_root}")
        return resolved

    def _spec(self, split: str) -> dict[str, str]:
        if split not in self.splits:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(self.splits)}")
        return self.splits[split]


def load_paths(cfg: dict[str, Any] | None = None) -> Paths:
    """Build Paths from configs/paths.yaml with env var overrides."""
    cfg = cfg if cfg is not None else load_config("paths.yaml")
    drive_root = Path(os.path.abspath(os.environ.get("EFFSQL_DRIVE_ROOT", cfg["drive_root"])))
    local_root = Path(os.path.abspath(os.environ.get("EFFSQL_LOCAL_ROOT", cfg["local_root"])))
    return Paths(
        drive_root=drive_root,
        local_root=local_root,
        out_root=drive_root / cfg.get("out_subdir", "effsql"),
        splits=cfg["splits"],
        forbidden=[drive_root / p for p in cfg.get("forbidden", [])],
        bird_dev_release=cfg.get("bird_dev_release", ""),
    )
