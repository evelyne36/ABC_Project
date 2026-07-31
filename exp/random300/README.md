# Random300 CrossGB benchmark

## Purpose

This experiment compares `crossgb`, `crossgb_pred0`, and CPU XGBoost on the
repository TabArena regression, binary, and multiclass datasets.

## Paper correspondence

This directory supplies the paired 300-candidate TabArena CrossGB--XGBoost
study discussed in the main paper. Appendix Sections B.1--B.2 define selection
and the shared search space; Appendix Tables 14--15 contain the complete
TabArena regression and classification panels. This submission package retains
only this paired core benchmark.

## Environment

```bash
python3 -m pip install -r requirements.txt
bash CrossGB/scripts/build_crossgb.sh
bash CrossGB/scripts/verify_crossgb.sh
```

## Data and protocol

- Data: `dataset/TabArena_regression`, `dataset/TabArena_binary`, and
  `dataset/TabArena_multiclass`.
- Nested CV: 5 outer folds and 4 inner folds for full runs.
- Search: 300 candidates per model in full mode; CrossGB balances ten
  predictive-feature fractions across three score aggregations.
- Default categorical handling: `onehot`. Pass another handling explicitly
  only when reproducing a separately reported preprocessing condition.
- Fairness: XGBoost and CrossGB share sampled tree backbones and mapped L2
  values. CPU threading should use `--task-workers N --n-jobs 1`.

Run all commands from the repository root.

```bash
python3 scripts/download_datasets.py download --group tabarena
python3 scripts/download_datasets.py verify --group tabarena
```

## Smoke versus full execution

The smoke command uses one regression dataset, 2x2 nested CV, one outer fold,
and 30 candidates. It validates the pipeline but does not reproduce a paper
cell. The full command uses the frozen 5x4 nested-CV, 300-candidate protocol.

### Smoke run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/random300/scripts/exp1_random300_all.py \
  --mode quick --task-types regression \
  --models xgb_cpu crossgb \
  --regression-datasets airfoil_self_noise.csv \
  --categorical-handling onehot \
  --max-candidates 30 --outer-folds 2 --inner-folds 2 \
  --outer-fold-index 0 --task-workers 1 --n-jobs 1 \
  --run-id smoke_random300
```

### Full run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/random300/scripts/run_random300_full.py \
  --categorical-handling onehot \
  --run-id random300_full
```

## Outputs and restart behavior

Each run writes `exp/random300/results/<run-id>/` and
`exp/random300/logs/<run-id>/`. Important files include `summary.csv`,
`candidates.csv`, `metadata.json`, `parameter_audit.json`, and
`candidate_configs.json`. This runner is not shard-resumable: after an
interruption, use a new run ID and retain the old directory for diagnosis.
The appendix tables use paired outer-fold rows from `summary.csv`; retain the
candidate and parameter audits with every reported run.

## Limitations

Categorical encoding is part of the experimental condition; compare only runs
with the same handling. Search results are seed- and fold-dependent, and the
local CrossGB build must match the source used for reported results.
