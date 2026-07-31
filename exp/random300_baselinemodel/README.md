# Random300 baseline-model benchmark

## Purpose

This experiment supplies non-CrossGB reference models for the extra-tabular
Random300 protocol: linear, Extra Trees, Random Forest, LightGBM, CatBoost,
Torch MLP, RealMLP, and TabM.

## Paper correspondence

This directory reproduces the independently tuned strong-baseline panels in the
main paper. Appendix Section B.3 and Table 3 define the model-specific search
spaces; Appendix Table 16 reports the fixed ten-dataset regression and binary
panels. These baselines use equal candidate counts, not identical
hyperparameter semantics.

## Environment

Install the repository environment first. The local `requirements-*.txt` files
are model-family subsets for constrained Linux servers; they are not alternate
experiment protocols.

```bash
python3 -m pip install -r requirements.txt
```

## Data and protocol

- Data: the OpenML panel in `exp/tabular_extra_random300/data`, plus the
  TabArena regression fallback directory `dataset/TabArena_regression`.
- Full protocol: 5 outer folds, 4 inner folds, 300 candidates, and both
  `drop` and `onehot` categorical handling.
- Candidate spaces are defined in
  `exp/random300_baselinemodel/configs/default_search_space.json`.

Run all commands from the repository root.

```bash
python3 scripts/download_datasets.py download --group openml
python3 scripts/download_datasets.py verify --group openml
python3 scripts/download_datasets.py download --group tabarena
python3 scripts/download_datasets.py verify --group tabarena
```

The fixed ten-dataset regression schedule includes
`QSAR_fish_toxicity.csv`. The resolver first checks the extra-tabular directory
and then the TabArena regression directory, where the root downloader places
that file. The binary schedule likewise contains ten fixed public datasets.

## Smoke versus full execution

The smoke launcher uses a reduced dataset/model/candidate budget and checks
that every selected optional model is available. It does not reproduce the
paper panels. The full launcher uses 5x4 nested CV and 300 candidates per model.

### Smoke run

```bash
bash exp/random300_baselinemodel/smoke_linux.sh
```

### Paper-panel run

```bash
python3 exp/random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full --task-types regression binary \
  --models linear extra_trees random_forest lightgbm catboost torchmlp \
  --categorical-handlings drop onehot \
  --max-candidates 300 --outer-folds 5 --inner-folds 4 \
  --task-workers 8 --n-jobs 1 --run-id random300_baseline_table16
```

`run_linux_full.sh` additionally schedules multiclass, RealMLP, and TabM
experiments. Those are retained as extended analyses and are not part of
Appendix Table 16. The root environment installs all of them; the local
`requirements-*.txt` files provide smaller core, tree, or deep subsets only
for intentionally constrained runs.

The paper-panel command above uses the runner's frozen ten-regression and
ten-binary default lists, so QSAR is included without an additional flag.

## Outputs and restart behavior

Outputs are stored in `exp/random300_baselinemodel/results/<run-id>/` and logs
in `exp/random300_baselinemodel/logs/<run-id>/`. Review `summary.csv`,
`candidates.csv`, `metadata.json`, model availability/errors, and
`task_status.json`. This runner is not shard-resumable; rerun interrupted work
with a new run ID. A paper-panel run is valid only when all requested models
and datasets are present in `task_status.json`; unavailable models are recorded
rather than silently omitted.

## Limitations

Optional libraries can differ substantially in version, CPU behavior, and
preprocessing support. Failed optional models remain recorded rather than
silently removed. Compare only identical dataset, fold, candidate-budget, and
categorical-handling conditions.
