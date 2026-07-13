# Random300 Baseline Model Implementation Record

Date: 2026-07-07

This document records the current implementation of `random300_baselinemodel`
and the hyperparameter protocol needed to make every baseline model comparable
under a Random300 nested-CV setting.

## Goal

The goal is to add non-XGBoost baselines for the same tabular Random300
comparison surface used by the current MGB/XGBoost experiments.

The intended comparison is:

- same datasets as `tabular_extra_random300`;
- same categorical handling modes: `drop` and `onehot`;
- same nested-CV structure;
- same task granularity;
- model-by-model execution so cheap baselines finish first;
- default setting plus simple random hyperparameter search;
- no Transformer/foundation/in-context tabular models in this baseline folder.

## Current Scope

Implemented baseline models:

1. `linear`
2. `extra_trees`
3. `random_forest`
4. `lightgbm`
5. `catboost`
6. `torchmlp`
7. `realmlp`
8. `tabm`

Not included by design:

- XGBoost and MGB, because they already live in the existing Random300/MGB
  comparison surface.
- FT-Transformer, TabTransformer, SAINT, TabPFN, TabICL, TabDPT, TabSTAR, and
  similar Transformer/foundation/in-context approaches, because their
  comparison protocol is not aligned with this baseline goal.

## Directory Layout

`random300_baselinemodel/`

- `scripts/exp1_random300_all.py`: main CLI entry point.
- `scripts/run_random300_full.py`: compatibility wrapper for full runs.
- `utils/candidates.py`: model order and Random300 candidate generation.
- `utils/schedule.py`: task construction and light-to-heavy ordering.
- `utils/model_factory.py`: sklearn/LightGBM/CatBoost/PyTorch/pytabkit estimator
  construction.
- `utils/runner.py`: nested-CV execution and artifact writing.
- `configs/default_search_space.json`: current Random300 search space.
- `requirements-*.txt`: dependency sets by model family.
- `run_linux_full.sh`: full experiment launch script.
- `smoke_linux.sh`: minimal local smoke.
- `HYPERPARAMETER_NOTES.md`: shorter discussion notes.
- `IMPLEMENTATION_RECORD.md`: this detailed record.

## Task Unit

The unit of execution is:

```text
model * categorical_handling * task_type * dataset * outer_fold
```

Where:

- `model` is one of the implemented baselines.
- `categorical_handling` is `drop` or `onehot`.
- `task_type` is `regression`, `binary`, or `multiclass`.
- `dataset` is one staged CSV from `tabular_extra_random300/data`.
- `outer_fold` is one outer split index.

Each task runs the full inner-CV model selection for that model and then fits
the selected candidate on the outer-train split before evaluating outer-test.

## Execution Order

The scheduler is intentionally outcome-first:

1. model order;
2. categorical handling order;
3. dataset pressure order;
4. outer fold index.

Current model order:

```text
linear -> extra_trees -> random_forest -> lightgbm -> catboost -> torchmlp -> realmlp -> tabm
```

Current categorical order:

```text
drop -> onehot
```

The dataset pressure order is inherited from the tabular extra Random300 runner
and approximates cost with existing row/feature pressure metadata. This gives
usable partial results early: cheap models, cheap preprocessing, and small
datasets are completed before heavier combinations.

## Nested-CV Protocol

Defaults:

- full mode: 5 outer folds, 4 inner folds;
- quick mode: caller-controlled lightweight smoke;
- full mode runs all outer folds;
- quick mode runs one requested outer fold unless `--run-all-outer-folds` is
  passed.

Primary metrics match the existing tabular Random300 runner:

- regression: RMSE, minimize;
- binary classification: AUC, maximize;
- multiclass classification: logloss, minimize.

Model selection uses the mean inner-CV primary metric. Ties are broken by lower
inner train+predict time and then candidate index.

## Outputs

Each run writes:

- `results/<run_id>/candidate_configs.json`
- `results/<run_id>/parameter_audit.json`
- `results/<run_id>/candidates.csv`
- `results/<run_id>/summary.csv`
- `results/<run_id>/aggregate.csv`
- `results/<run_id>/run_metadata.json`
- `logs/<run_id>/task_status.json`
- `logs/<run_id>/tasks/*.log`

`candidates.csv` contains every evaluated candidate for every task.

`summary.csv` contains one selected result per task.

`aggregate.csv` aggregates by:

```text
model, categorical_handling, task_type, dataset
```

## Dependency Strategy

The implementation supports partial installation. Missing optional dependencies
do not crash the entire run; the affected task records a failed/no-valid status
with the dependency error in its task log.

Dependency files:

- `requirements-core.txt`: numpy, pandas, scipy, scikit-learn, joblib,
  threadpoolctl. Supports `linear`, `extra_trees`, `random_forest`.
- `requirements-tree.txt`: core plus LightGBM and CatBoost.
- `requirements-deep.txt`: core plus PyTorch, PyTorch Lightning stack, and
  pytabkit. Supports `torchmlp`, `realmlp`, `tabm`.
- `requirements-all.txt`: tree plus deep.
- `requirements-dev.txt`: all plus pytest.
- `requirements-linux-cpu.txt`: one-shot Linux CPU-only full-suite
  requirements, with key versions smoke-tested in conda env `llm`.

Install examples:

```bash
pip install -r random300_baselinemodel/requirements-core.txt
pip install -r random300_baselinemodel/requirements-tree.txt
pip install -r random300_baselinemodel/requirements-deep.txt

# Recommended on Linux experiment servers for CPU-only fairness:
pip install -r random300_baselinemodel/requirements-linux-cpu.txt
```

## Random300 Hyperparameter Contract

The important fairness rule is:

```text
Every selected baseline model receives exactly 300 candidates in full mode.
```

Candidate 1 is a default-like candidate. Candidates 2 through 300 are sampled
from the model's search space using a fixed seed. This gives each model:

- a recognizable default anchor;
- equal HPO candidate budget;
- reproducibility;
- library-specific parameters only where they materially define the model.

Full mode now defaults to `--max-candidates 300`. Quick mode defaults to 3
candidates only for smoke/testing.

## Why Random300 Is Reasonable Per Model

Random300 is reasonable here because the goal is not to exhaustively optimize
each library, but to give each baseline a nontrivial, equal, reproducible HPO
budget under the same nested-CV selection protocol.

The protocol is especially appropriate when:

- the model families differ and a shared grid would be artificial;
- the comparison target is performance under a bounded experiment budget;
- every model has a default candidate plus a simple model-specific search;
- final model choice is made only on inner folds.

The search spaces should be broad enough to contain plausible high-performing
settings, but not so broad that Random300 is mostly wasted on pathological
settings. The current spaces are therefore conservative, not maximal.

## Hyperparameter Alignment With MGB/XGBoost

The current MGB/XGBoost Random300 shared backbone uses:

- `learning_rate`: log-uniform from 0.005 to 0.1
- `max_depth`: log-integer from 4 to 10
- `subsample`: uniform from 0.6 to 1.0
- `colsample_bytree`: uniform from 0.6 to 1.0
- `n_estimators`: 200, 400, 600, or 800
- `lambda_l2`: log-uniform from `1e-4` to 5
- `max_bin`: fixed at 255

Baseline spaces now follow this contract where the model has an analogous
parameter:

- LightGBM directly aligns learning rate, estimators, depth, row sampling,
  column sampling, and L2 regularization.
- CatBoost directly aligns learning rate, iterations, depth, and L2 leaf
  regularization.
- RandomForest and ExtraTrees align tree count and feature subsampling, and use
  the same depth scale plus `None` as the default-unlimited tree baseline.
- Linear/Ridge uses the same L2 range for `alpha`; LogisticRegression uses the
  inverse regularization range `C = 1 / lambda_l2`, approximately 0.2 to 10000.
- Neural models do not share the numerical scale of GBDT learning rates or L2,
  so they keep model-appropriate ranges, but they use the same Random300
  candidate budget and expose analogous training budget, learning rate, and
  weight-decay controls.

## Hyperparameter Spaces

The active source of truth is:

```text
random300_baselinemodel/configs/default_search_space.json
```

### Linear / Ridge / Logistic

Regression:

- `alpha=0` is represented by the default candidate only if a caller explicitly
  sets it; current default uses Ridge with `alpha=1`.
- sampled regression candidates tune `alpha` log-uniformly.

Classification:

- Logistic regression tunes inverse regularization `C` log-uniformly.
- `max_iter` is fixed at 1000 for stability.

Current sampled knobs:

- `alpha`: log-uniform from `1e-4` to `5`;
- `C`: log-uniform from `0.2` to `10000`;
- `max_iter`: fixed at `1000`.

Reasoning:

- regularization dominates linear-model performance;
- this is cheap enough that Random300 is acceptable;
- tuning both `alpha` and `C` in shared candidates is harmless because only the
  relevant one is used for a given task type.

### Extra Trees

Current knobs:

- `n_estimators`: 200, 400, 600, or 800;
- `max_features`: uniform from 0.6 to 1.0;
- `min_samples_leaf`: log-integer from 1 to 20;
- `max_depth`: `None`, 4, 6, 8, or 10.

Reasoning:

- `ExtraTrees` is fast enough for Random300;
- `max_features` and leaf size cover the main bias/variance tradeoff;
- random thresholds already define the model family, so deeper library-specific
  tuning is not needed for a fair baseline.

### Random Forest

Current knobs:

- `n_estimators`: 200, 400, 600, or 800;
- `max_features`: uniform from 0.6 to 1.0;
- `min_samples_leaf`: log-integer from 1 to 20;
- `max_depth`: `None`, 4, 6, 8, or 10.

Reasoning:

- kept parallel to Extra Trees for comparability;
- RandomForest is heavier than ExtraTrees, so full Random300 is fair but
  expensive;
- common tree knobs are enough for the first rigorous baseline.

### LightGBM

Current knobs:

- `n_estimators`: 200, 400, 600, or 800;
- `learning_rate`: log-uniform from 0.005 to 0.1;
- `max_depth`: log-integer from 4 to 10;
- `num_leaves`: 15, 31, 63, or 127;
- `subsample`: uniform from 0.6 to 1.0;
- `colsample_bytree`: uniform from 0.6 to 1.0;
- `reg_lambda`: log-uniform from `1e-4` to 5.

Reasoning:

- this mirrors the public knobs usually responsible for GBDT performance:
  boosting length, shrinkage, leaf complexity, row/column sampling, and L2;
- other LightGBM-specific defaults remain at library defaults to avoid
  overfitting the protocol to LightGBM.

### CatBoost

Current knobs:

- `iterations`: 200, 400, 600, or 800;
- `learning_rate`: log-uniform from 0.005 to 0.1;
- `depth`: log-integer from 4 to 10;
- `l2_leaf_reg`: log-uniform from `1e-4` to 5.

Reasoning:

- these are the high-impact CatBoost controls analogous to the LightGBM space;
- categorical features are already externally handled by `drop`/`onehot`, so
  CatBoost native categorical handling is intentionally not used here;
- this keeps the comparison aligned with the same preprocessing contract used
  by MGB/XGBoost.

### TorchMLP

Current knobs:

- `hidden_layers`: `[128]`, `[256, 128]`, or `[512, 256]`;
- `learning_rate`: log-uniform from 0.0001 to 0.003;
- `batch_size`: 128, 256, or 512;
- `max_epochs`: 80, 120, or 200;
- `weight_decay`: log-uniform from `1e-4` to `1e-2`.

Reasoning:

- this is a conservative plain MLP baseline, not a tuned architecture search;
- learning rate, width/depth, batch size, and weight decay are the main knobs;
- fixed activation/loss/optimizer reduce protocol variance.

Current implementation details:

- standardizes input features;
- uses ReLU MLP;
- uses AdamW;
- uses MSE for regression and cross-entropy for classification;
- runs on CPU by default unless the estimator is extended.

### RealMLP

Current knobs:

- `max_epochs`: 80, 120, or 200;
- `learning_rate`: log-uniform from 0.0001 to 0.003;
- `batch_size`: 128, 256, or 512;
- `weight_decay`: log-uniform from `1e-4` to `1e-2`.

Reasoning:

- RealMLP is intended to be a strong MLP-family baseline;
- the current protocol gives it equal Random300 HPO budget while avoiding a
  large architecture search until the exact pytabkit version/API is finalized;
- final results should record the installed pytabkit version.

Implementation note:

- `model_factory.py` tries pytabkit-compatible sklearn estimator class names.
- If the installed pytabkit API differs, update the class-name mapping before
  the large run.

### TabM

Current knobs:

- `max_epochs`: 80, 120, or 200;
- `learning_rate`: log-uniform from 0.0001 to 0.003;
- `batch_size`: 128, 256, or 512;
- `k`: 16 or 32;
- `weight_decay`: log-uniform from `1e-4` to `1e-2`.

Reasoning:

- TabM is heavier and ensemble-like, so it is deliberately scheduled last;
- `k` is exposed because it directly controls the parameter-efficient ensemble
  size;
- other training knobs are aligned with TorchMLP/RealMLP for a fair neural
  baseline family comparison.

Implementation note:

- Like RealMLP, TabM currently depends on a pytabkit-compatible sklearn
  estimator.
- Before final large runs, confirm the exact class names and installed package
  version on the experiment server.

## Current Rigor Checks

Implemented tests:

- candidate generation is deterministic;
- requested models are covered;
- every model can generate 300 unique candidates;
- task ordering is model-first, categorical-first, and light-dataset-first.

Smoke checked:

- `linear + extra_trees` on `regression__Moneyball.csv`;
- `drop + onehot`;
- 2 outer folds, 2 inner folds;
- 4/4 tasks completed with status `ok`.

Optional dependency behavior checked:

- missing LightGBM records a task-level failure and does not crash the run.

## 2026-07-08 CPU Smoke Test In Conda Env `llm`

Environment:

- Python: 3.10.0
- numpy: 2.2.6
- pandas: 2.3.3
- scipy: 1.15.3
- scikit-learn: 1.7.2
- LightGBM: 4.6.0
- CatBoost: 1.2.10
- PyTorch: 2.12.1
- pytabkit: 1.7.3

CPU check:

```text
torch.version.cuda = None
torch.cuda.is_available() = False
```

Smoke tests run with:

- `--max-candidates 1`
- `--outer-folds 2`
- `--inner-folds 2`
- `--outer-fold-index 0`
- `--task-workers 1`
- `--n-jobs 1`

Results:

- all models, regression, `drop`, `regression__Moneyball.csv`: 8/8 ok
- all models, regression, `onehot`, `regression__Moneyball.csv`: 8/8 ok
- all models, binary, `drop`, `binary__haberman.csv`: 8/8 ok
- all models, multiclass, `drop`, `multiclass__balance-scale.csv`: 8/8 ok
- `pip check`: no broken requirements

During debugging, `realmlp` and `tabm` required pytabkit-specific constructor
names:

- RealMLP: `n_epochs`, `lr`, `wd`
- TabM: `n_epochs`, `lr`, `weight_decay`, `tabm_k`

The implementation now maps the shared config names to those pytabkit names and
forces `device="cpu"` for pytabkit models.

## Known Limitations

1. RealMLP and TabM class resolution is validated against pytabkit 1.7.3. If
   the server uses a different pytabkit version, re-run the smoke tests before
   launching full Random300.
2. TorchMLP is a local estimator, not a published benchmark implementation.
   This is acceptable for a plain MLP baseline, but should be stated clearly in
   experiment reporting.
3. Full Random300 for deep models is computationally heavy. The scheduler makes
   this manageable by running cheaper models first, but full completion may need
   model-by-model server launches.
4. The current protocol does not use early stopping. This keeps nested-CV
   behavior simple, but may be less efficient for neural/boosting models.
5. CatBoost native categorical handling is not used, because this folder
   intentionally compares the same `drop/onehot` preprocessing contract as the
   existing Random300 protocol.

## Recommended Run Strategy

Fast first pass:

```bash
python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --models linear extra_trees random_forest \
  --max-candidates 300 \
  --task-workers 8 \
  --n-jobs 1 \
  --run-id baseline_trees_light_random300
```

Then tree boosting:

```bash
python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --models lightgbm catboost \
  --max-candidates 300 \
  --task-workers 8 \
  --n-jobs 1 \
  --run-id baseline_boosting_random300
```

Then neural baselines:

```bash
python random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --models torchmlp realmlp tabm \
  --max-candidates 300 \
  --task-workers 4 \
  --n-jobs 1 \
  --run-id baseline_neural_random300
```

For the complete all-model run:

```bash
bash random300_baselinemodel/run_linux_full.sh
```

## References For Final Protocol Discussion

- TabArena: https://arxiv.org/abs/2506.16791
- Revisiting Deep Learning Models for Tabular Data:
  https://arxiv.org/abs/2106.11959
- RealMLP: https://arxiv.org/abs/2407.04491
- TabM: https://arxiv.org/abs/2410.24210
- PyTabKit: https://github.com/dholzmueller/pytabkit
