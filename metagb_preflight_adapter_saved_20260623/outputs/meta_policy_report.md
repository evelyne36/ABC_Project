# Trainable MetaGB Preflight Policy Report

## Protocol

- Unit of evaluation: dataset.
- Split: grouped leave-one-dataset-out.
- Classifier: L2-regularized logistic regression when both classes are present; class-prior fallback otherwise.
- Regression heads: ridge regressors for relative gain and log runtime ratio.
- Decision target: `mgb_worth_it`, derived from quality/runtime utility.

## Configuration

- input: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/metagb_preflight_adapter/outputs/preflight_dataset_table.csv`
- rows: 11
- features: 24
- alpha_runtime_penalty: 0.002
- decision_threshold: 0.5
- classifier_c: 0.5
- ridge_alpha: 1.0

## Cross-Validated Classification

- worth-it positives: 2/11
- accuracy: 0.5455
- ROC-AUC: 0.1111
- average precision: 0.1556

## Utility Evaluation

- policy_mean_utility: -0.009471
- always_mgb_mean_utility: -0.031596
- oracle_mean_utility: 0.001112
- policy_regret_vs_oracle: 0.010583
- try_rate: 0.272727

## Threshold Tuning Diagnostic

- This sweep is computed on cross-validated predictions.
- It is useful for choosing candidate thresholds, but final threshold selection still needs historical-only validation.
- best_threshold_by_cv_utility: 0.65
- best_threshold_policy_mean_utility: 0.000000
- best_threshold_try_rate: 0.0000
- best_threshold_retained_positive_rate: 0.0000
- best_threshold_rejected_loss_rate: 1.0000

## Policy-vs-Observed Confusion

| policy_decision   |   False |   True |   All |
|:------------------|--------:|-------:|------:|
| TRY_MGB           |       3 |      0 |     3 |
| USE_XGB           |       6 |      2 |     8 |
| All               |       9 |      2 |    11 |

## Regression Heads

- relative gain MAE: 1.102087

## Top Final Logistic Coefficients

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| zero_fraction_processed      |      0.554214 |          0.554214 |
| projected_gradient_lift_mean |      0.357323 |          0.357323 |
| local_linearity_lift         |     -0.354565 |          0.354565 |
| local_abs_corr_max_mean      |     -0.349617 |          0.349617 |
| raw_categorical_fraction     |     -0.288046 |          0.288046 |
| raw_numeric_fraction         |      0.288046 |          0.288046 |
| onehot_expansion_ratio       |      0.288046 |          0.288046 |
| raw_features                 |     -0.223601 |          0.223601 |
| raw_rows                     |      0.192974 |          0.192974 |
| processed_rows               |      0.192974 |          0.192974 |
| feature_row_ratio            |      0.192066 |          0.192066 |
| tree_leaf_count              |      0.186348 |          0.186348 |

## Academic Caveats

- This report is a protocol scaffold, not final evidence: the current labeled set is small.
- The correct validation unit is a dataset/preprocessing group; candidate-level splits would leak information.
- Thresholds and runtime penalty must be selected on historical datasets only before evaluating new datasets.
- The projected-gradient features are theory-aligned proxies for MetaGB's second-order leaf score, not a full MetaGB training run.
