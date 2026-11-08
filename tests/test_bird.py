import pytest

from effsql.data.bird import load_examples, load_tied_gold, read_gold_file


def test_dev_examples(fake_bird):
    ex = load_examples("dev", fake_bird)
    assert [e.qid for e in ex] == ["dev_0", "dev_1", "dev_2"]
    assert ex[0].bird_question_id == 0 and ex[0].difficulty == "simple"
    # gold comes from the .sql file; the differing JSON SQL is kept for reference
    assert ex[1].gold_sql == "SELECT SUM(amount) FROM orders"
    assert ex[1].json_sql == "SELECT sum(amount) FROM orders"
    assert ex[0].json_sql is None


def test_train_examples_use_index_qids(fake_bird):
    ex = load_examples("train", fake_bird)
    assert [e.qid for e in ex] == ["train_0", "train_1", "train_2"]
    assert all(e.difficulty is None and e.bird_question_id is None for e in ex)


def test_minidev_keeps_file_order_and_dev_question_ids(fake_bird):
    ex = load_examples("minidev", fake_bird)
    # qids by position because question_ids repeat in mini-dev
    assert [e.qid for e in ex] == ["minidev_0", "minidev_1", "minidev_2"]
    assert [e.bird_question_id for e in ex] == [2, 0, 0]


def test_gold_count_mismatch_raises(fake_bird):
    gold = fake_bird.split_file("dev", "gold")
    gold.write_text("SELECT 1\tshop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold lines"):
        load_examples("dev", fake_bird)


def test_gold_db_order_mismatch_raises(fake_bird):
    gold = fake_bird.split_file("dev", "gold")
    lines = gold.read_text(encoding="utf-8").strip().split("\n")
    lines[0], lines[2] = lines[2], lines[0]
    gold.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="db_id"):
        load_examples("dev", fake_bird)


def test_gold_without_tab_raises(tmp_path):
    p = tmp_path / "g.sql"
    p.write_text("SELECT 1 shop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tab"):
        read_gold_file(p)


def test_tied_gold(fake_bird):
    tied = load_tied_gold(fake_bird)
    assert tied[0].sql == "SELECT count(id) FROM customers"
    assert 1 not in tied  # reworded question is excluded


def test_question_id_mode_requires_ids(fake_bird):
    fake_bird.splits["train"]["qid"] = "question_id"
    with pytest.raises(ValueError, match="no question_id"):
        load_examples("train", fake_bird)
