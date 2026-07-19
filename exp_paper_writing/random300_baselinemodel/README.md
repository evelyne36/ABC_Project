# Random300 Baseline Models

This folder adds non-XGBoost baselines for the same tabular extension protocol
used by `tabular_extra_random300`.

The task unit is:

`model * categorical_handling(drop/onehot) * dataset * outer_fold`

Tasks are scheduled model-first, then light categorical handling first, then
low-pressure datasets before high-pressure datasets, then fold index. This is
intended to produce usable partial results quickly.

Default model order:

1. `linear`
2. `extra_trees`
3. `random_forest`
4. `lightgbm`
5. `catboost`
6. `torchmlp`
7. `realmlp`
8. `tabm`

The runner writes the same style of artifacts as the existing Random300 runner:

- `results/<run_id>/candidate_configs.json`
- `results/<run_id>/parameter_audit.json`
- `results/<run_id>/candidates.csv`
- `results/<run_id>/summary.csv`
- `results/<run_id>/aggregate.csv`
- `results/<run_id>/run_metadata.json`
- `logs/<run_id>/task_status.json`
- `logs/<run_id>/tasks/*.log`

Quick smoke:

```bash
bash random300_baselinemodel/smoke_linux.sh
```

Full launch:

```bash
bash random300_baselinemodel/run_linux_full.sh
```

Optional dependencies:

- `linear`, `random_forest`, `extra_trees`: scikit-learn
- `lightgbm`: LightGBM
- `catboost`: CatBoost
- `torchmlp`: PyTorch
- `realmlp`, `tabm`: pytabkit-compatible sklearn estimators

Install sets:

```bash
# Fast first batch: linear + RF + ExtraTrees
pip install -r random300_baselinemodel/requirements-core.txt

# Add LightGBM + CatBoost
pip install -r random300_baselinemodel/requirements-tree.txt

# Add TorchMLP + RealMLP + TabM
pip install -r random300_baselinemodel/requirements-deep.txt

# Everything
pip install -r random300_baselinemodel/requirements-all.txt

# Linux CPU-only full suite, recommended for fair baseline runs
pip install -r random300_baselinemodel/requirements-linux-cpu.txt
```

If an optional dependency is missing, that task records a failed status instead
of crashing the whole run.

Hyperparameter note: this first framework uses each library's default-like
setting plus Random300 model-specific candidates in full mode. The
deep-learning ranges are intentionally conservative placeholders until the final
RealMLP/TabM/TorchMLP protocol is agreed. See `IMPLEMENTATION_RECORD.md` for the
detailed protocol record.
