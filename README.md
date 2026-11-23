# correct-is-not-enough

Efficiency-aware text-to-SQL with small open LLMs on [BIRD](https://bird-bench.github.io/).
CMPE 586 term project. The `effsql` package generates candidate SQL, measures correctness and cost,
builds a candidate pool for later SFT and DPO, and evaluates predictions with EX, VES and R-VES.

Status: **stages 1 to 5 built** (data, measurement, generation, screening, rewriting, candidate pool, verification); smoke run pending.

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
| `10_pipeline` | L4 GPU (Part A), then CPU (Part B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/10_pipeline.ipynb) |
| `11_search` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/11_search.ipynb) |
| `32_rl` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/32_rl.ipynb) |
| `33_rl_evaluate` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/33_rl_evaluate.ipynb) |
| `34_rl_eff` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/34_rl_eff.ipynb) |
| `35_rl_eff_evaluate` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/35_rl_eff_evaluate.ipynb) |
| `36_reference_models` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/36_reference_models.ipynb) |
| `37_reference_evaluate` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/37_reference_evaluate.ipynb) |
| `70_search_full` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/70_search_full.ipynb) |
| `71_own_answers_and_pairs` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/71_own_answers_and_pairs.ipynb) |
| `72_dpo` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/72_dpo.ipynb) |
| `73_dpo_check` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/73_dpo_check.ipynb) |
| `74_rl` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/74_rl.ipynb) |
| `75_choose_and_report` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/75_choose_and_report.ipynb) |
| `76_ablations` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/76_ablations.ipynb) |
| `77_ablations_report` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/77_ablations_report.ipynb) |
| `80_judge_train` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/80_judge_train.ipynb) |
| `81_judge_candidates` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/81_judge_candidates.ipynb) |
| `82_judge_report` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/82_judge_report.ipynb) |
| `83_vote` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/83_vote.ipynb) |
| `84_latency` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/84_latency.ipynb) |
| `85_judge_results` | A100 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/85_judge_results.ipynb) |
| `86_judge_results_report` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/86_judge_results_report.ipynb) |
| `40_evaluate` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/40_evaluate.ipynb) |
| `50_analysis` | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selimsidan/correct-is-not-enough/blob/main/notebooks/50_analysis.ipynb) |

`10_pipeline` runs the whole candidate-pool pipeline from one notebook: Part A on an L4 GPU (generate, screen,
rewrite, build the pool, predict dev and mini-dev), Part B on a CPU runtime (verify the pool against real timing,
evaluate, summary). `40_evaluate` scores any predictions file on its own; `50_analysis` draws the report figures.

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

## Commands (stage 2)

```bash
effsql gold-cache --split dev                    # gold result hash, vm_steps, plan per question (cached)
effsql evaluate --pred gold --split dev --run sanity_gold_dev               # gold vs gold, EX only
effsql evaluate --pred preds.jsonl --split dev --run my_model --model M --method base
effsql evaluate --pred preds.jsonl --split minidev --timing official --repeat 3 --run my_model_rves
effsql export-official --pred preds.jsonl --split minidev --out /content/official_check
```

Predictions: our JSONL (`{"qid": "dev_0", "sql": "..."}` per line) or the official JSON
(`{"0": "SQL\t----- bird -----\tdb_id"}`, aligned by position). `--pred gold` scores the gold SQL.

### Metrics

| Metric | Definition |
|---|---|
| EX | `set(pred_rows) == set(gold_rows)`, ported from BIRD mini-dev `evaluation_ex.py` |
| EX-tied (dev) | EX where a match with the alternative gold in `dev_tied_append.json` also counts (39 of 42 used) |
| R-VES | mini-dev `evaluation_ves.py`: 100 timed runs per correct pair, 3-sigma cleaning, reward buckets, `mean(sqrt(reward)) * 100` |
| VES | original BIRD formula `mean(sqrt(time_ratio)) * 100`, on the same timing loop |
| relative cost | `(steps_pred + c) / (steps_gold + c)` for correct predictions, c = progress handler period |
| faster than gold | share of correct predictions with relative cost below 0.9 |
| efficient accuracy at tau | share of questions that are correct with relative cost <= tau |
| R-VES (steps) | R-VES with speed-up = 1 / relative cost instead of timed seconds: deterministic, gold = 100, wrong = 0; the main metric for choosing between trained models |
| R-VES/EX | R-VES (steps) over the correct answers only: efficiency apart from accuracy (100 = as cheap as gold) |
| VES (steps) | `mean(sqrt(1 / relative cost)) * 100`, unbounded, so a few extreme speed-ups dominate it; reported, not used to choose |

Trained models are chosen on the validation databases only (`python -m effsql.train select`): among the
models whose EX is not significantly below the best (McNemar), the highest R-VES (steps) wins.
`python -m effsql.train report` prints every model on validation, dev and mini-dev from the saved
evaluations, with a bootstrap interval for R-VES and paired tests against a reference model.

`vm_steps` counts SQLite VM instructions through a progress handler (every N = 10 instructions). It is
deterministic: connections load the schema at open and do not cache prepared statements.
Counting slows queries 2 to 4x, so execution runs in two passes: correctness at native speed under the
official 30 s limit (deadline checked every 1000 instructions), then step counting only for ok golds and
correct predictions, with a 300 s limit.
Timing presets: `quick` (perf_counter, 1 warm-up, 5 interleaved runs, for development) and
`official` (exact port, for reported numbers). Timing is always sequential; only the correctness and
step-counting pass is parallel.

## Pipeline (stages 3 to 5)

```bash
# Part A, L4 GPU runtime
effsql generate --split train --run R [--limit 20]       # 2 prompts x (1 greedy + 4 samples) per question
effsql parse --split train --run R                       # extract SQL, canonical form, dedup; gold joins the pool
effsql screen --split train --run R                      # execute: correctness, vm_steps, plan
effsql diversity --split train --run R --round 0         # >= 2 correct candidates at different cost levels?
effsql rewrite --run R --round 1                         # gold rewrites for flagged questions (4 styles)
effsql parse ... ; effsql screen ... ; effsql diversity --split train --run R --round 1
effsql build-pool --split train --run R                  # runs/R/pool_train.parquet (+ .jsonl)
effsql predict --split dev --prompt base --run R         # greedy predictions for evaluate
# Part B, CPU runtime
effsql verify-pool --split train --run R                 # step ordering vs wall-clock timing
effsql evaluate --pred runs/R/predictions_dev_base.jsonl --split dev --run R_dev_base
effsql summary --split train --run R
effsql render-prompt --split dev --qid dev_0             # inspect a prompt on CPU
```

Configs: [generate.yaml](configs/generate.yaml) (model, sampling), [rewrite.yaml](configs/rewrite.yaml) (styles),
[pool.yaml](configs/pool.yaml) (efficiency levels, step cap, verification), prompt templates in
[configs/prompts/](configs/prompts/). Only the model's original SQL is ever executed; the sqlglot canonical form is
just the dedup key. Rewrites are kept only when their result matches gold.

### Cost-guided search (notebook 11)

The first pool has a query at least 2x cheaper than gold for only about 6% of train questions. The search gives a
model what an engineer would look at, over several rounds: the cheapest known correct query with its step count and
SQLite plan, and the earlier attempts with what went wrong. Answers are executed and kept only when they return
gold's result, pass the arbitrary-row check and use no constant that appears nowhere in the question, evidence or
gold. A pilot on 300 train questions compares Qwen2.5-Coder-32B (AWQ) with our 7B model in the same loop.

```bash
python -m effsql.gen.search --run search_pilot_32b --model 32b --limit 300   # A100; resumable
```

Settings in [search.yaml](configs/search.yaml), prompts `search_system.txt` and `search_user.txt`.

### Reinforcement learning with an execution reward (notebooks 32, 33)

The model writes 8 answers per train question; every answer is executed and scored (0 when wrong, failing,
cut off, an arbitrary-row pick that gold does not use, or using a constant found nowhere in the question,
evidence or gold; 0.1 to 1.9 when correct, by its cost relative to gold); the better answers of each
question are made more likely than its other answers (clipped policy gradient, GRPO-style, no KL term).
Generation (vLLM) and training alternate in separate processes; every iteration is saved.

```bash
python -m effsql.train rl --run rl_v1_pilot --init-from sft_v1_best_e1 --pilot   # cost projection first
python -m effsql.train rl --run rl_v1 --init-from sft_v1_best_e1                 # A100; resumable
```

Settings in the `rl` section of [train.yaml](configs/train.yaml).

Efficiency phase (notebooks 34, 35): `rl --settings rl_eff` continues from the kept RL model on the
questions where efficiency can be learned (a strictly usable pool query at most 0.9x gold's cost, or correct
RL answers that differed in cost), with 16 answers per question and a cost reward saturating at 2x.

### Reference models (notebooks 36, 37)

Published models run in our harness, for comparison only (nothing is trained or chosen with them):
OmniSQL-7B, Arctic-Text2SQL-R1-7B, Qwen2.5-Coder-32B-Instruct (AWQ) and our base model with OmniSQL's prompt.
Each gets the prompt it was published with (OmniSQL's schema format, Arctic's `<think>` prompt; settings in
[reference.yaml](configs/reference.yaml)); predictions are scored by `effsql evaluate` like ours, so accuracy
and query cost are directly comparable.

```bash
python -m effsql.gen.reference --model arctic7b          # GPU; writes runs/ref_arctic7b, dev and mini-dev
```

### Cost first, then accuracy (notebooks 70 to 77)

Why: training for cheaper queries failed in two ways. RL cannot find cheaper forms (the model's own answers
to a question almost never differ in cost, so a cost reward has nothing to compare), and preference
training moved the cost but lost accuracy, partly because its "cheaper correct" queries included first-row
picks that match gold only by luck, and its rejected queries were ones the model would never write.

The method, in order:
1. **Cost-guided search** on every train question (`python -m effsql.gen.search --run search_v1 --model 7b`):
   starting from the first pool, the cheapest *usable* correct query known per question
   (`cheap_targets_train.jsonl`). Usable is strict: same result as gold, no first-row pick gold does not use,
   no constant found nowhere in the question, evidence or gold.
2. **The model's own answers**: the current model answers every train question 8 times, all executed
   (`python -m effsql.train rl --settings own_answers`: sampling and scoring only).
3. **Pairs** (`python -m effsql.train pairs`): chosen = the cheap target; rejected = the model's own correct
   answer costing at least 10% more (efficiency) or its own wrong answer (correctness), most frequent first.
4. **DPO** from the current model on these pairs (2 epochs; the model after epoch 1 is kept too).
5. **RL** from the DPO model, with the cheapest known query as the cost yardstick of the reward
   (`--settings rl_v2`) instead of gold.
6. **Choice on validation** (`python -m effsql.train select-cost`, fixed before training): a model is
   allowed unless its EX is significantly lower than the current model's (McNemar), then the lowest cost on
   the questions both answer correctly wins (or the current model is kept). Dev and mini-dev are scored after.

Ablations (DPO only): chosen = gold (no generated queries), chosen = the first pool only (no search),
rejected = other models' pool queries (off-policy, as in the first DPO).

### Generate, then judge (notebooks 80 to 82)

One model, two LoRA adapters on the same frozen Qwen2.5-Coder-7B. The generator (`rl_v1_it8`, unchanged)
writes up to 17 candidates per question (greedy, 8 samples with its prompt, 8 with the efficient prompt).
The judge (`configs/judge.yaml`, `src/effsql/judge/`) reads one candidate and gives p, the probability that it
is correct ("yes" against "no" from the language-model head, as in GradeSQL), and c, its cost relative to the
question's other correct answers (a linear head on the same hidden state). Answer: among the candidates with
p >= max p - delta, the lowest c. Nothing is executed at answer time; delta is chosen on validation.

```bash
python -m effsql.judge data --run judge_data_v1                    # CPU: own_v1 + pool_v1, already executed
python -m effsql.judge train --run judge_v1 --data judge_data_v1   # A100, resumable
python -m effsql.judge candidates --run judge_cand_v1              # A100 (vLLM): val, dev, mini-dev
python -m effsql.judge score --run judge_cand_v1 --judge judge_v1  # A100: p and c per candidate
python -m effsql.judge label --run judge_cand_v1                   # CPU: execution, for delta and analysis only
python -m effsql.judge select --run judge_cand_v1 --judge judge_v1 --out judge_v1_sel   # CPU
```

The judge starts as a copy of the generator's adapter and trains one pass on 55,904 candidates of 8,141 train
questions (the generator's own answers first, then the first pool; up to 8 per question), validated on the
held-out databases. Questions of one database share the schema in one sequence and a question's candidates
share its text (attention mask and position ids keep every candidate's result equal to scoring it alone).

### Choosing with execution at answer time (notebooks 83, 84)

The candidates hold a correct answer far more often than the judge's pick, so the answer may also use the
database: every distinct candidate is compiled against the schema (EXPLAIN, no data read) and run once with a
1 s cap, all in parallel. Methods (`src/effsql/judge/vote.py`): the judge alone, the judge among candidates
that compile, result voting (candidates grouped by result, the group written most often wins, the judge picks
within it) and judge-weighted result voting, each also with candidates that return no rows left out of the
vote when some candidate returned rows. The method with the highest validation EX is kept.

```bash
python -m effsql.judge exec --run judge_cand_v1                                   # CPU, resumable
python -m effsql.judge vote --run judge_cand_v1 --judge judge_v1 --out vote_v1    # CPU
python -m effsql.judge latency --judge judge_v1 --out latency_v1                  # A100: one engine, both adapters
```

`latency` answers 100 dev questions one at a time with the base model loaded once and both adapters (vLLM),
and reports the time per stage (generate, judge, waiting for the queries, choosing).

### A judge that reads the results (notebooks 85, 86)

`judge_v2` replaces `judge_v1` (it is not trained on top of it): the same data, settings and starting point
(a copy of the generator's adapter), but each prompt also shows what the candidate returned when run with the
answer-time cap (the number of rows and the first 5 rows, or the error). The two judges differ only in what
they read, so the comparison shows whether a learned judge can take the place of hand-written rules such as
"rows first".

```bash
python -m effsql.judge exec-data --run judge_data_v1                              # CPU, train databases
python -m effsql.judge train --run judge_v2 --data judge_data_v1 --with-results   # A100, resumable
python -m effsql.judge score --run judge_cand_v1 --judge judge_v2                 # reads exec_<split>.jsonl
python -m effsql.judge vote --run judge_cand_v1 --judge judge_v2 --out vote_v2    # CPU
```

### Ablations (notebooks 87, 88)

Each run removes or changes one part of the final system; every comparison is on the same questions, with an
exact McNemar test. Notebook 87 (A100, about 8 hours, one "Run all" that resumes after a disconnect):

| Run | What changes |
|---|---|
| B3 | candidates from the SFT generator (before RL), chosen by `judge_v2` |
| B5 | `judge_untrained`: the base model asked the judge's question with the result, no training |
| B1 | `judge_v2_own`: trained on the generator's own answers only (no base-model pool) |
| B4 | `judge_v2_nocost`: trained without the cost loss |
| B2 | `judge_v2_base`: started from the plain base model instead of the generator's adapter |

Notebook 88 (CPU) recomputes every choice from saved files and writes `ablations_report/report.txt`: the
component ladder, the verifier ablations (including an execution majority vote with nothing learned), accuracy
against the number of samples, the two prompts' candidates, results by difficulty, what the verifier learned,
the cost spread of the model's own answers and worked examples.

```bash
python -m effsql.judge candidates --run judge_cand_sft --generator sft_v1_best_e1   # B3, GPU
python -m effsql.judge untrained --run judge_untrained                             # B5, GPU
python -m effsql.judge derive-data --src judge_data_v1 --run judge_data_own --sources own
python -m effsql.judge train --run judge_v2_own --data judge_data_own --with-results          # B1
python -m effsql.judge train --run judge_v2_nocost --data judge_data_v1 --with-results --cost-weight 0
python -m effsql.judge train --run judge_v2_base --data judge_data_v1 --with-results --init-from none
python -m effsql.ablate zero-shot-val --run zero_shot_v1                           # CPU
python -m effsql.ablate report --out ablations_report                              # CPU, nothing executed
```

### Demo (notebook 90)

[Open the demo in Colab](https://colab.research.google.com/github/selimsidan/generate-then-judge-sql/blob/main/notebooks/90_demo.ipynb):
pick a BIRD dev database, type a question (and optionally a hint), and see the SQL the system chose, its result,
and every query it wrote with what it returned and the verifier's score. *Run all* on a free T4 GPU (the base model
in 4 bit) or an L4 or A100 (16 bit, exactly the reported system). BIRD dev is downloaded from BIRD's site and
checked against the files every experiment used; the two adapters come from the Hugging Face repository.

`effsql.answer.System` is the whole answer path for one question (generate, run with the 1 s cap, judge, choose).
The demo (`effsql.app`) and the latency measurement (`python -m effsql.judge latency`) both answer through it.

### Compute and cost

* GPU (L4) only for `generate`, `rewrite` and `predict`; everything else is CPU work. Part A runs its CPU steps on
  the L4 machine to avoid switching runtimes; Part B (timing) must run on a CPU runtime.
* Smoke run first (`LIMIT = 20` in the notebook). Every command resumes after a disconnect.
* Disconnect and delete the GPU runtime as soon as Part A finishes: an idle L4 keeps using compute units.

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
