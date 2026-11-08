# correct-is-not-enough

Efficiency-aware text-to-SQL with small open LLMs on [BIRD](https://bird-bench.github.io/).
CMPE 586 term project. The `effsql` package generates candidate SQL, measures correctness and cost,
builds a candidate pool for later SFT and DPO, and evaluates predictions with EX, VES and R-VES.

Status: **stage 1** (skeleton, paths, inspection, staging, loader, schema cache).

## Where things live

| What | Where |
|---|---|
| Code | this repository (the only place code lives) |
| Execution | Google Colab, notebooks opened from GitHub |
| BIRD data | Google Drive, `MyDrive/586/` |
| Outputs | Google Drive, `MyDrive/586/effsql/` |
| Databases during a session | Colab local disk, `/content/bird_local/` (never queried on Drive) |

### Drive layout

Upload the official BIRD releases (`train.zip`, `dev.zip`, `minidev.zip`, original 2023 dev) like this.
Only the files named below are read.

```
MyDrive/586/
  dev/
    dev.json  dev.sql  dev_tables.json  dev_tied_append.json  dev_databases.zip
  train/train/
    train.json  train_gold.sql  train_tables.json  train_databases.zip
    train_databases/          (partial leftover: never read)
  minidev/MINIDEV/
    mini_dev_sqlite.json  mini_dev_sqlite_gold.sql  dev_tables.json
    dev_databases/<db_id>/database_description/*.csv   (only diffed by `inspect`)
  effsql/                     (created by the code)
    cache/schemas/<split>/<db_id>.json
    cache/gold_costs/<split>.jsonl
    cache/inspect/inspect_<timestamp>.json
    runs/<run_name>/
    results/results.csv
```

Mini-dev (SQLite) runs on the 11 dev databases, so it has no zip of its own.
Paths are set in [configs/paths.yaml](configs/paths.yaml) and two env vars:
`EFFSQL_DRIVE_ROOT` (default `/content/drive/MyDrive/586`) and `EFFSQL_LOCAL_ROOT` (default `/content/bird_local`).

## Notebooks

| Notebook | Runtime | Open |
|---|---|---|
| `00_setup_and_stage` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/00_setup_and_stage.ipynb) |

Later stages add `10_generate` (GPU), `20_screen_and_rewrite` (CPU, then GPU), `30_build_and_verify_pool` (CPU),
`40_evaluate` (CPU) and `50_analysis` (CPU). Only LLM generation uses a GPU.

Notebooks contain no logic: the setup cell, then `!effsql ...` calls.

### Shared setup cell

Every notebook starts with the contents of [notebooks/setup_cell.py](notebooks/setup_cell.py)
(a test keeps them identical). It mounts Drive, clones or updates this repo at `REF` (default `main`),
installs it (`[gen]` extra when `GPU = True`), sets the env vars and prints the git commit.
Every run also writes the commit, CPU model and package versions into `runs/<run>/run_meta.json`.

## Commands (stage 1)

```bash
effsql inspect                                   # verify the real BIRD files, save a report
effsql stage --split dev                         # extract the 11 dev databases locally
effsql stage --split minidev                     # checks mini-dev db_ids, reuses the dev databases
effsql stage --split train --shard 0/8           # train: one shard of the 69 databases at a time
effsql schema-cache --split dev                  # per-database schema JSON on Drive
effsql schema-cache --split train --shard 0/8
```

`--db-ids a,b`, `--shard i/n` and `--limit N` select databases for every command. Shards are round-robin
over sorted db_ids, so the same shard spec always selects the same databases. Staging is idempotent
and refuses to run if the selected databases will not fit on the local disk.

## Local development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # strips notebook outputs, blocks en and em dashes
pytest
```

Tests build tiny SQLite databases and fake BIRD zips, so they need neither Drive nor BIRD.

## Rules this repo follows

- Never commit data, databases, model outputs, tokens or notebook outputs.
- BIRD dev and mini-dev are for evaluation only; no training data is built from their gold SQL.
- SQL is only ever executed on the local copy of a database.
