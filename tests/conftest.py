"""Fixtures: a fake Drive tree that mirrors the real BIRD layout, built from tiny SQLite dbs.

Nothing here needs Drive or BIRD. Layout (relative to EFFSQL_DRIVE_ROOT):
  dev/dev.json, dev.sql, dev_tables.json, dev_tied_append.json, dev_databases.zip
  train/train/train.json, train_gold.sql, train_tables.json, train_databases.zip
  train/train/train_databases/   partial leftover (forbidden)
  minidev/MINIDEV/mini_dev_sqlite.json, mini_dev_sqlite_gold.sql, dev_tables.json, dev_databases/
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from effsql.paths import Paths, load_paths

LONG_NOTE = "x" * 500


def make_shop_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, "home city" TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            amount REAL,
            note TEXT
        );
        """
    )
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", [(i, f"c{i}", f"city{i % 3}") for i in range(10)])
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)", [(i, i % 10, i * 1.5, LONG_NOTE) for i in range(50)]
    )
    conn.commit()
    conn.close()


def make_simple_db(path: Path, table: str, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, title TEXT, data BLOB)")
    conn.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)", [(i, f"t{i}", b"\x00\x01") for i in range(n)])
    conn.commit()
    conn.close()


CUSTOMERS_CSV = (
    "original_column_name,column_name,column_description,data_format,value_description\r\n"
    "id,customer id,unique id,integer,\r\n"
    "name,,customer name (caf\xe9),text,\r\n"
    "home city,,city of residence,text,\r\n"
).encode("cp1252")  # not valid utf-8, exercises the encoding fallback
ORDERS_CSV = b"original_column_name,column_name,column_description\nid,,order id\namount,,total in EUR\n"


def write_db_zip(zip_path: Path, prefix: str, dbs: dict[str, Path], descriptions: dict[str, dict[str, bytes]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}/", "")
        for db_id, src in dbs.items():
            zf.write(src, f"{prefix}/{db_id}/{db_id}.sqlite")
            zf.writestr(f"{prefix}/{db_id}/._{db_id}.sqlite", b"junk")
            zf.writestr(f"__MACOSX/{prefix}/{db_id}/._{db_id}.sqlite", b"junk")
            for name, content in descriptions.get(db_id, {}).items():
                zf.writestr(f"{prefix}/{db_id}/database_description/{name}", content)
        zf.writestr(f"{prefix}/.DS_Store", b"junk")


def tables_entry(db_id: str) -> dict:
    if db_id == "shop":
        return {
            "db_id": "shop",
            "table_names_original": ["customers", "orders"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "home city"],
                                      [1, "id"], [1, "customer_id"], [1, "amount"], [1, "note"]],
            "primary_keys": [1, 4],
            "foreign_keys": [[5, 1]],
        }
    return {"db_id": db_id, "table_names_original": [], "column_names_original": [[-1, "*"]],
            "primary_keys": [], "foreign_keys": []}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def write_gold(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{sql}\t{db}\n" for sql, db in pairs) + "\n", encoding="utf-8")


DEV_RECORDS = [
    {"question_id": 0, "db_id": "shop", "question": "How many customers?", "evidence": "",
     "SQL": "SELECT count(*) FROM customers", "difficulty": "simple"},
    {"question_id": 1, "db_id": "shop", "question": "Total order amount?", "evidence": "amount is in EUR",
     "SQL": "SELECT sum(amount) FROM orders", "difficulty": "moderate"},
    {"question_id": 2, "db_id": "school", "question": "How many students?", "evidence": "",
     "SQL": "SELECT count(*) FROM students", "difficulty": "challenging"},
]


@pytest.fixture
def fake_bird(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Paths:
    root = tmp_path / "drive" / "586"
    src = tmp_path / "src_dbs"
    make_shop_db(src / "shop.sqlite")
    make_simple_db(src / "school.sqlite", "students", 7)
    make_simple_db(src / "movies.sqlite", "movies", 20)
    make_simple_db(src / "books.sqlite", "books", 5)

    # dev
    dev = root / "dev"
    write_json(dev / "dev.json", DEV_RECORDS)
    gold = [(r["SQL"], r["db_id"]) for r in DEV_RECORDS]
    gold[1] = ("SELECT SUM(amount) FROM orders", "shop")  # gold file differs from the JSON SQL field
    write_gold(dev / "dev.sql", gold)
    write_json(dev / "dev_tables.json", [tables_entry("shop"), tables_entry("school")])
    write_json(dev / "dev_tied_append.json", [dict(DEV_RECORDS[0], SQL="SELECT count(id) FROM customers")])
    write_db_zip(
        dev / "dev_databases.zip", "dev_databases",
        {"shop": src / "shop.sqlite", "school": src / "school.sqlite"},
        {"shop": {"customers.csv": CUSTOMERS_CSV, "orders.csv": ORDERS_CSV}},
    )

    # train (no question_id, no difficulty)
    tr = root / "train" / "train"
    train_records = [
        {"db_id": "movies", "question": "How many movies?", "evidence": "", "SQL": "SELECT count(*) FROM movies"},
        {"db_id": "books", "question": "How many books?", "evidence": "", "SQL": "SELECT count(*) FROM books"},
        {"db_id": "movies", "question": "Max id?", "evidence": "", "SQL": "SELECT max(id) FROM movies"},
    ]
    write_json(tr / "train.json", train_records)
    write_gold(tr / "train_gold.sql", [(r["SQL"], r["db_id"]) for r in train_records])
    write_json(tr / "train_tables.json", [tables_entry("movies"), tables_entry("books")])
    write_db_zip(tr / "train_databases.zip", "train_databases",
                 {"movies": src / "movies.sqlite", "books": src / "books.sqlite"}, {})
    make_simple_db(tr / "train_databases" / "movies" / "movies.sqlite", "partial", 1)  # forbidden leftover
    (root / "train" / "__MACOSX").mkdir(parents=True)

    # minidev: subset of dev, different order; one description file differs from the dev zip
    md = root / "minidev" / "MINIDEV"
    mini = [DEV_RECORDS[2], DEV_RECORDS[0]]
    write_json(md / "mini_dev_sqlite.json", mini)
    write_gold(md / "mini_dev_sqlite_gold.sql", [(r["SQL"], r["db_id"]) for r in mini])
    write_json(md / "dev_tables.json", [tables_entry("shop"), tables_entry("school")])
    desc = md / "dev_databases" / "shop" / "database_description"
    desc.mkdir(parents=True)
    (desc / "customers.csv").write_bytes(CUSTOMERS_CSV)
    (desc / "orders.csv").write_bytes(ORDERS_CSV + b"note,,free text\n")

    monkeypatch.setenv("EFFSQL_DRIVE_ROOT", str(root))
    monkeypatch.setenv("EFFSQL_LOCAL_ROOT", str(tmp_path / "local"))
    return load_paths()
