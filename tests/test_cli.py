import sys

from effsql.cli import main


def test_cli_inspect_stage_schema(fake_bird, capsys, monkeypatch):
    monkeypatch.setattr("effsql.data.stage._free_bytes", lambda p: 10**15)
    main(["inspect"])
    out = capsys.readouterr().out
    assert "=== dev.zip ===" in out and "Saved report to" in out
    assert list((fake_bird.cache_dir / "inspect").glob("inspect_*.json"))

    main(["stage", "--split", "dev"])
    main(["schema-cache", "--split", "dev", "--db-ids", "shop"])
    assert fake_bird.schema_cache_path("dev", "shop").exists()
    assert not fake_bird.schema_cache_path("dev", "school").exists()


def test_cpu_commands_do_not_import_gpu_stack(fake_bird, monkeypatch):
    monkeypatch.setattr("effsql.data.stage._free_bytes", lambda p: 10**15)
    main(["stage", "--split", "train", "--shard", "0/2"])
    assert "vllm" not in sys.modules and "torch" not in sys.modules
