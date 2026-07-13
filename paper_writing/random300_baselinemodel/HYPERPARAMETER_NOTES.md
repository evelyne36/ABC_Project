# Hyperparameter Notes

This is the initial discussion surface for the baseline-model protocol. The
framework is intentionally usable before the final hyperparameter policy is
locked.

## Current Framework Policy

- Every model starts with one default-like candidate.
- Full mode uses 300 candidates per model: one default-like candidate plus 299
  simple random-search samples from
  `configs/default_search_space.json`.
- Nested CV selection uses the same primary metrics as the existing tabular
  Random300 runner:
  - regression: RMSE, minimize
  - binary: AUC, maximize
  - multiclass: logloss, minimize
- Categorical handling is explicit and task-level:
  - `drop`
  - `onehot`

## Model-Specific Starting Points

## Alignment With MGB/XGBoost Random300

The baseline spaces intentionally reuse the MGB/XGBoost shared Random300 ranges
when a model has a comparable parameter:

- boosting learning rate: 0.005 to 0.1;
- boosting/tree count: 200, 400, 600, or 800;
- tree depth scale: 4 to 10;
- row/feature sampling: 0.6 to 1.0;
- L2-like regularization: `1e-4` to 5.

Model-specific parameters remain model-specific. For example, LightGBM also
tunes `num_leaves`, forests tune `min_samples_leaf`, and TabM tunes `k`.
Neural learning rates and weight decay use neural-network-appropriate numeric
scales, but still receive the same Random300 candidate budget.

### Linear / Ridge / Logistic

- Regression uses `LinearRegression` when `alpha=0`; otherwise `Ridge`.
- Classification uses `LogisticRegression`.
- Search is intentionally cheap over `alpha` and `C`.

### Random Forest / Extra Trees

- Public knobs are shared where possible:
  `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth`.
- Other implementation-specific defaults are left untouched.
- This keeps tree baselines cheap and avoids overfitting the comparison to
  non-public tuning details.

### LightGBM / CatBoost

- Current ranges cover conservative learning rate, boosting length, depth/leaf
  complexity, and L2 regularization.
- Non-shared library-specific defaults are left to the library.
- These should be revisited only if we decide to match a paper-specific or
  benchmark-specific HPO budget.

### TorchMLP

- Current implementation is a small PyTorch MLP estimator with standardization,
  AdamW, ReLU layers, and cross-entropy/MSE loss.
- Current search covers hidden size, learning rate, batch size, epochs, and
  weight decay.
- Before final large runs, decide whether this should remain a local Torch MLP
  or be replaced by a benchmark library implementation.

### RealMLP

- The framework expects a pytabkit-compatible sklearn estimator.
- Current search covers the high-impact training knobs only:
  epochs, learning rate, batch size, weight decay.
- RealMLP papers emphasize strong training protocol and regularization details;
  final settings should be aligned with the exact implementation/version used.

### TabM

- The framework expects a pytabkit-compatible sklearn estimator.
- Current search adds `k` to the shared neural knobs because TabM is based on
  parameter-efficient ensembling.
- TabM should run after cheaper baselines because it is heavier and ensemble-like.

## References To Use When Locking The Final Protocol

- TabArena: https://arxiv.org/abs/2506.16791
- Revisiting Deep Learning Models for Tabular Data: https://arxiv.org/abs/2106.11959
- RealMLP: https://arxiv.org/abs/2407.04491
- TabM: https://arxiv.org/abs/2410.24210

## Open Decisions

- Whether neural baselines get the same candidate count as tree baselines or a
  smaller time-budgeted search.
- Whether `torchmlp` should use this local PyTorch estimator or an external
  benchmark implementation.
- Which exact RealMLP/TabM package version and class names will be installed on
  the experiment server.
- Whether to cap epochs/time per dataset for TabM and RealMLP to keep the full
  run schedulable.
