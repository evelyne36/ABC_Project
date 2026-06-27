# MetaGB Preflight Adapter Plan

## Goal

Build a lightweight preflight scorer that predicts whether `current_metagb` is worth trying before running a full Random300/Random600 comparison against `xgb_cpu`.

The scorer should support two decisions:

1. Whether MetaGB is likely to improve quality enough to justify its runtime.
2. How much predictive-feature search budget is likely to be useful.

All work in this folder is isolated from the existing `random300`, `random600`, `datacollection0621`, and `dataset` directories. Those directories are read-only inputs.

## Working Hypothesis

MetaGB is useful when a dataset contains local regions where the response is better explained by one or a few predictive features than by a constant leaf value.

Concretely, the preflight signal should be high when:

- local leaf-level feature/target association is stronger than global association;
- a shallow conditional partition reveals predictive features different from the split features;
- the useful predictive feature set appears sparse enough that `pred_feature_fraction` can avoid wasting work;
- the task is mechanism-aligned with conditional/predictive decoupling.

The signal should be low when:

- ordinary constant leaves already explain most structure;
- local linear signal is weak or unstable;
- high-dimensional noise makes predictive-feature traversal expensive;
- feature scale dispersion suggests unstable `grad_x / hess_xx` statistics;
- the expected runtime overhead dominates likely quality gains.

## Phase 1: Post-hoc Label Table

Use completed result artifacts:

- `summary.csv`
- `candidates.csv`
- `pred_fraction_analysis.csv`

Generate a dataset-level label table:

- `mgb_relative_improvement`
- `mgb_win`
- `runtime_ratio`
- `mgb_worth_it`
- `selected_pred_feature_fraction_mean`
- `selected_metagb_score_agg_mode`
- `cv_test_same_direction`

The first implementation uses only completed paired `xgb_cpu` and `current_metagb` rows. Incomplete task groups stay out of the label table instead of being guessed from logs.

## Phase 2: PreAnalysis Features

For each dataset and preprocessing mode, compute cheap features from the dataset itself:

- raw and processed row/feature counts;
- numeric/categorical feature fraction;
- missing fraction;
- one-hot expansion ratio;
- zero/sparsity fraction after preprocessing;
- target/class imbalance;
- feature scale dispersion;
- global feature-target association;
- shallow-tree local association;
- conditional/predictive decoupling proxy.

The shallow-tree probe is intentionally model-agnostic and cheap. It is not a replacement for a MetaGB run; it is a diagnostic that asks whether local regions reveal stronger one-dimensional predictive responses.

## Phase 3: Rule Scorer

Start with an interpretable rule scorer instead of a black-box meta-model because the current completed dataset count is small.

Initial outputs:

- `USE_XGB`
- `TRY_MGB_LIGHT`
- `TRY_MGB_FULL`

The score combines:

- local linearity lift;
- local predictive strength;
- decoupling proxy;
- task-family prior;
- runtime-risk penalty;
- feature-scale penalty.

## Phase 4: Validation

Validate the rule scorer against existing labels:

- confusion table by recommendation;
- mean observed relative improvement by recommendation;
- retained-positive rate: how many known wins are kept by `TRY_MGB_*`;
- rejected-loss rate: how many known losses are sent to `USE_XGB`;
- runtime-risk analysis.

Because the initial label count is small, the first validation is descriptive. Leave-one-dataset-out or logistic meta-model validation should wait until more completed task groups are available.

## Phase 5: Micro Pre-experiment

After the table is stable, run a tiny controlled pre-experiment only on representative positive/negative datasets:

- small XGB pilot;
- small MetaGB pilot;
- pred whitelist or pred fraction sweep;
- compare whether the pilot decision predicts the full Random result.

This phase should be added as a separate script so the current read-only analysis remains reproducible and side-effect-light.

## Current Stop Condition

The first practical milestone is complete when this folder can generate:

- `outputs/preflight_dataset_table.csv`
- `outputs/preflight_recommendations.csv`
- `outputs/preflight_report.md`

from existing result CSVs and dataset files without modifying any existing experiment directory.
