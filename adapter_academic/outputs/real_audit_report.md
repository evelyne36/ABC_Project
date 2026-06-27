# Real Benchmark Retrospective Audit

## Contract

- Real benchmark outcomes are not used to compute mechanism scores.
- The audit only checks whether intrinsic mechanism scores align with completed experiments.
- High score but no benchmark win should trigger mechanism-to-optimization case analysis, not immediate scorer rejection.

## Band Counts

- MODERATE_MECHANISM_MATCH: 3
- WEAK_MECHANISM_MATCH: 8

## Ranked Scores

| dataset                              | mechanism_band           |   mechanism_alignment_score |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_predictive_concentration |   component_signal_support |   audit_mgb_relative_improvement | audit_mgb_win   |
|:-------------------------------------|:-------------------------|----------------------------:|----------------------------------:|---------------------------------:|--------------------------------:|-------------------------------------:|---------------------------:|---------------------------------:|:----------------|
| airfoil_self_noise.csv               | MODERATE_MECHANISM_MATCH |                    0.63959  |                         1         |                        0.0833333 |                        0.628279 |                             0.500516 |                   0.930495 |                      -0.081196   | False           |
| miami_housing.csv                    | MODERATE_MECHANISM_MATCH |                    0.555864 |                         1         |                        0.5       |                        0        |                             0        |                   0.808639 |                       0.00460172 | True            |
| concrete_compressive_strength.csv    | MODERATE_MECHANISM_MATCH |                    0.415223 |                         0         |                        0.916667  |                        0.229378 |                             0.462432 |                   0.939377 |                      -0.00261404 | False           |
| physiochemical_protein.csv           | WEAK_MECHANISM_MATCH     |                    0.397543 |                         1         |                        0         |                        0        |                             0.313288 |                   0.162137 |                       0.00716682 | True            |
| QSAR_fish_toxicity.csv               | WEAK_MECHANISM_MATCH     |                    0.323612 |                         0         |                        0.916667  |                        0        |                             0.448561 |                   0.495888 |                       0.0130623  | True            |
| wine_quality.csv                     | WEAK_MECHANISM_MATCH     |                    0.276658 |                         0         |                        0.916667  |                        0        |                             0.2619   |                   0.213018 |                      -0.00863573 | False           |
| diamonds.csv                         | WEAK_MECHANISM_MATCH     |                    0.259407 |                         0.0510362 |                        0.708333  |                        0        |                             0.259972 |                   0.384638 |                      -0.0130364  | False           |
| healthcare_insurance_expenses.csv    | WEAK_MECHANISM_MATCH     |                    0.229623 |                         0.158361  |                        0.291667  |                        0        |                             0.969635 |                   0.043163 |                      -0.00873052 | False           |
| houses.csv                           | WEAK_MECHANISM_MATCH     |                    0.179107 |                         0         |                        0.291667  |                        0        |                             0.548958 |                   0.512942 |                      -0.203699   | False           |
| Another-Dataset-on-used-Fiat-500.csv | WEAK_MECHANISM_MATCH     |                    0.130225 |                         0         |                        0         |                        0        |                             0.796446 |                   0.505808 |                      -0.0139137  | False           |
| Food_Delivery_Time.csv               | WEAK_MECHANISM_MATCH     |                    0.092982 |                         0         |                        0.142857  |                        0        |                             0.572677 |                   0        |                      -0.00131953 | False           |

## Outcome Audit By Mechanism Band

| mechanism_band           |   count |       mean |      median |
|:-------------------------|--------:|-----------:|------------:|
| MODERATE_MECHANISM_MATCH |       3 | -0.0264028 | -0.00261404 |
| WEAK_MECHANISM_MATCH     |       8 | -0.0286382 | -0.00868313 |

## High-Score Non-Wins For Case Analysis

| dataset                           | mechanism_band           |   mechanism_alignment_score |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_predictive_concentration |   component_signal_support |   audit_mgb_relative_improvement | audit_mgb_win   |
|:----------------------------------|:-------------------------|----------------------------:|----------------------------------:|---------------------------------:|--------------------------------:|-------------------------------------:|---------------------------:|---------------------------------:|:----------------|
| airfoil_self_noise.csv            | MODERATE_MECHANISM_MATCH |                    0.63959  |                                 1 |                        0.0833333 |                        0.628279 |                             0.500516 |                   0.930495 |                      -0.081196   | False           |
| concrete_compressive_strength.csv | MODERATE_MECHANISM_MATCH |                    0.415223 |                                 0 |                        0.916667  |                        0.229378 |                             0.462432 |                   0.939377 |                      -0.00261404 | False           |

These cases may contain MetaGB-compatible structure that the current training protocol failed to convert into final benchmark gain.
