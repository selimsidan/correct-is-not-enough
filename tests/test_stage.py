import os
import shutil
from collections import namedtuple

import pytest

from effsql.data import stage as stage_mod
from effsql.data.stage import parse_shard, select_db_ids, stage
from effsql.io import read_json

CFG = {"stage": {"disk_reserve_gb": 0, "copy_buffer_mb": 1}}


def test_stage_dev_extracts_dbs_and_descriptions_without_junk(fake_bird):
    manifest = stage("dev", paths=fake_bird, data_cfg=CFG)
    assert sorted(manifest["dbs"]) == ["school", "shop"]
    shop = fake_bird.local_db_dir("dev", "shop")
    assert (shop / "shop.sqlite").exists()
    assert sorted(p.name for p in (shop / "database_description").iterdir()) == ["customers.csv", "orders.csv"]
    all_files = [p.name for p in fake_bird.local_root.rglob("*")]
    assert not [n for n in all_files if n.startswith("._") or n in (".DS_Store", "__MACOSX")]
    entry = manifest["dbs"]["shop"]
    assert entry["n_tables"] == 2 and entry["size_bytes"] > 0 and entry["source_zip"].endswith("dev_databases.zip")
    assert read_json(fake_bird.manifest_path("dev")) == manifest


def test_stage_is_idempotent(fake_bird, caplog):
    stage("dev", paths=fake_bird, data_cfg=CFG)
    db = fake_bird.local_db_path("dev", "shop")
    mtime = os.stat(db).st_mtime_ns
    caplog.set_level("INFO")
    stage("dev", paths=fake_bird, data_cfg=CFG)
    assert os.stat(db).st_mtime_ns == mtime
    assert "2 already staged, 0 to extract" in caplog.text


def test_stage_restages_truncated_file(fake_bird):
    stage("dev", paths=fake_bird, data_cfg=CFG)
    db = fake_bird.local_db_path("dev", "shop")
    size = db.stat().st_size
    with open(db, "r+b") as f:
        f.truncate(100)
    stage("dev", paths=fake_bird, data_cfg=CFG)
    assert db.stat().st_size == size


def test_stage_train_subset(fake_bird):
    manifest = stage("train", db_ids=["books"], paths=fake_bird, data_cfg=CFG)
    assert list(manifest["dbs"]) == ["books"]
    assert not fake_bird.local_db_dir("train", "movies").exists()
    # the partial Drive folder is never used as a source
    assert manifest["dbs"]["books"]["source_zip"].endswith("train_databases.zip")


def test_stage_unknown_db_id(fake_bird):
    with pytest.raises(ValueError, match="Unknown db_ids"):
        stage("train", db_ids=["nope"], paths=fake_bird, data_cfg=CFG)


def test_minidev_stages_into_dev_folder(fake_bird):
    manifest = stage("minidev", paths=fake_bird, data_cfg=CFG)
    assert sorted(manifest["dbs"]) == ["school", "shop"]
    assert fake_bird.manifest_path("minidev") == fake_bird.manifest_path("dev")
    assert fake_bird.local_db_path("minidev", "shop") == fake_bird.local_db_path("dev", "shop")


def test_disk_check_refuses(fake_bird, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(stage_mod.shutil, "disk_usage", lambda p: usage(10, 10, 10))
    with pytest.raises(RuntimeError, match="Not enough local disk"):
        stage("dev", paths=fake_bird, data_cfg=CFG)


def test_corrupt_database_fails_loudly(fake_bird, monkeypatch):
    stage("dev", paths=fake_bird, data_cfg=CFG)
    db = fake_bird.local_db_path("dev", "school")
    db.write_bytes(b"not a database" * 100)
    # make the size check pass so the corrupt file is not re-extracted
    monkeypatch.setattr(stage_mod, "_is_staged", lambda *a: True)
    with pytest.raises(RuntimeError, match="school"):
        stage("dev", paths=fake_bird, data_cfg=CFG)


def test_select_and_shard():
    ids = ["e", "a", "d", "b", "c"]
    assert select_db_ids(ids) == ["a", "b", "c", "d", "e"]
    shards = [select_db_ids(ids, shard=f"{i}/2") for i in range(2)]
    assert shards == [["a", "c", "e"], ["b", "d"]]
    assert select_db_ids(ids, db_ids=["c", "a"], limit=1) == ["a"]
    with pytest.raises(ValueError):
        parse_shard("2/2")
