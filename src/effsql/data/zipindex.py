"""Locate each database's members inside a BIRD databases zip without extracting anything.

The prefix inside the zip (expected `dev_databases/` or `train_databases/`) is not assumed:
a `.sqlite` member whose parent folder has the same name as the file stem defines a database,
and `.csv` files under `<db_id>/database_description/` are its column descriptions.
macOS junk (__MACOSX, ._*, .DS_Store) is ignored.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from effsql.paths import is_junk


@dataclass
class DbMembers:
    db_id: str
    sqlite: zipfile.ZipInfo | None = None
    descriptions: list[zipfile.ZipInfo] = field(default_factory=list)

    @property
    def members(self) -> list[zipfile.ZipInfo]:
        return ([self.sqlite] if self.sqlite else []) + self.descriptions

    @property
    def uncompressed_bytes(self) -> int:
        return sum(m.file_size for m in self.members)


@dataclass
class ZipIndex:
    dbs: dict[str, DbMembers]
    n_entries: int
    n_junk: int
    top_level: Counter
    nested_zips: list[str]
    other: list[str]


def index_zip(zf: zipfile.ZipFile) -> ZipIndex:
    dbs: dict[str, DbMembers] = {}
    n_junk = 0
    top_level: Counter = Counter()
    nested: list[str] = []
    other: list[str] = []
    infos = zf.infolist()
    for info in infos:
        name = info.filename
        if is_junk(name):
            n_junk += 1
            continue
        top_level[name.split("/", 1)[0]] += 1
        if info.is_dir():
            continue
        p = PurePosixPath(name)
        parts = p.parts
        if p.suffix.lower() == ".zip":
            nested.append(name)
        elif p.suffix.lower() == ".sqlite" and len(parts) >= 2 and parts[-2] == p.stem:
            entry = dbs.setdefault(p.stem, DbMembers(p.stem))
            if entry.sqlite is not None:
                raise ValueError(f"Two .sqlite members for db {p.stem!r}: {entry.sqlite.filename}, {name}")
            entry.sqlite = info
        elif (
            p.suffix.lower() == ".csv"
            and len(parts) >= 3
            and parts[-2] == "database_description"
        ):
            dbs.setdefault(parts[-3], DbMembers(parts[-3])).descriptions.append(info)
        else:
            other.append(name)
    return ZipIndex(dbs, len(infos), n_junk, top_level, nested, other)
