import pytest

from effsql.exec.engine import connect_readonly
from effsql.paths import is_junk


def test_env_overrides(fake_bird, tmp_path):
    assert fake_bird.drive_root == tmp_path / "drive" / "586"
    assert fake_bird.out_root == fake_bird.drive_root / "effsql"
    assert fake_bird.split_file("train", "questions").name == "train.json"
    assert fake_bird.db_zip("minidev") == fake_bird.split_file("dev", "db_zip")


def test_never_open_database_on_drive(fake_bird):
    on_drive = fake_bird.drive_root / "train" / "train" / "train_databases" / "movies" / "movies.sqlite"
    assert on_drive.exists()
    with pytest.raises(PermissionError):
        connect_readonly(on_drive, fake_bird)


def test_refuse_paths_outside_local_root(fake_bird, tmp_path):
    with pytest.raises(PermissionError):
        connect_readonly(tmp_path / "elsewhere.sqlite", fake_bird)


def test_forbidden_partial_folder(fake_bird):
    fake_bird.splits["train"]["questions"] = "train/train/train_databases/x.json"
    with pytest.raises(PermissionError):
        fake_bird.split_file("train", "questions")


def test_readonly_connection(fake_bird):
    from effsql.data.stage import stage

    stage("dev", paths=fake_bird, data_cfg={"stage": {"disk_reserve_gb": 0}})
    conn = connect_readonly(fake_bird.local_db_path("dev", "shop"), fake_bird)
    assert conn.execute("SELECT count(*) FROM customers").fetchone()[0] == 10
    with pytest.raises(Exception, match="readonly"):
        conn.execute("DELETE FROM customers")


@pytest.mark.parametrize(
    "name,junk",
    [("__MACOSX/a/b.sqlite", True), ("a/._b.sqlite", True), ("a/.DS_Store", True), ("a/b/b.sqlite", False)],
)
def test_is_junk(name, junk):
    assert is_junk(name) is junk


def test_bad_db_id_rejected(fake_bird):
    with pytest.raises(ValueError):
        fake_bird.local_db_path("dev", "../etc")
