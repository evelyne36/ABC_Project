# Synthetic baseline-model benchmark

## Purpose

This suite evaluates LightGBM, CatBoost, and Random Forest on the exact
synthetic scenario definitions used by `exp/synthetic_crossgb`. It provides
tree-model reference results without changing the CrossGB experiment.

## Paper correspondence

This directory supplies the conventional tree baselines for the five-model
synthetic comparison. Appendix Section B.3 and Table 3 define their independent
300-candidate spaces; Appendix Tables 7--8 report the dimension-10 and
dimension-20 five-model results. Scenario definitions are specified in
Appendix Section C and Table 6.

## Environment

```bash
python3 -m pip install -r requirements.txt
```

`requirements-linux-cpu.txt` is retained only as a smaller CPU-only
model-family environment for servers where the complete repository environment
is not desired.

## Data and protocol

Data are generated from `exp/synthetic_crossgb/scenarios.py`. The smoke
protocol uses a reduced scenario/candidate budget; the full
`synthetic_baseline_random300_single_seed` protocol uses the frozen main
scenario grid, 5x4 nested CV, 300 candidates, and data seed 42.

Run all commands from the repository root.

## Smoke versus full execution

The smoke launcher reduces the scenario and candidate grid and is only an
environment/pipeline check. The full protocol runs every frozen scenario cell
with 5x4 nested CV and 300 independently sampled candidates per model.

### Smoke run

```bash
bash exp/synthetic_baselinemodel/smoke_linux.sh
```

### Full run

```bash
MODEL_NAMES="lightgbm catboost random_forest" \
TASK_WORKERS=15 N_JOBS=1 \
bash exp/synthetic_baselinemodel/run_linux_full.sh
```

## Outputs and resume

Run artifacts are stored in
`exp/synthetic_baselinemodel/results/<run-id>/`, with per-task JSON shards,
fold/candidate tables, statistics, audits, and metadata. The launcher stores
logs and a PID under `exp/synthetic_baselinemodel/logs/`. Resume the identical
run with:

```bash
RUN_ID=synthetic_baseline_random300_seed42 \
bash exp/synthetic_baselinemodel/run_linux_full.sh --resume
```

The implementation and protocol hashes prevent incompatible resumes.
Use the fold-level tables, not smoke summaries, when rebuilding Appendix
Tables 7--8.

## Limitations

Optional model packages and versions affect availability and runtime. Baseline
candidate spaces are model-specific, so conclusions should distinguish equal
candidate counts from identical hyperparameter semantics.
