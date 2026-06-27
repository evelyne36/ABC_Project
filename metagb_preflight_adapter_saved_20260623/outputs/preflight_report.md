# MetaGB Preflight Report

Rows: 11

## Recommendation Counts

- TRY_MGB_LIGHT: 5
- USE_XGB: 6

## Observed Labels

- Labeled datasets: 11
- Observed wins: 3/11
- Mean relative improvement: -0.0280

## Relative Improvement By Recommendation

| recommendation   |   count |       mean |      median |
|:-----------------|--------:|-----------:|------------:|
| TRY_MGB_LIGHT    |       5 | -0.0158365 | -0.00261404 |
| USE_XGB          |       6 | -0.0381886 | -0.00868313 |

## Try-vs-Win Confusion

| try_mgb   |   False |   True |   All |
|:----------|--------:|-------:|------:|
| False     |       5 |      1 |     6 |
| True      |       3 |      2 |     5 |
| All       |       8 |      3 |    11 |

## Missed Wins

| dataset                    | recommendation   |   preflight_score |   mgb_relative_improvement |   local_linearity_lift |   decoupled_leaf_fraction |
|:---------------------------|:-----------------|------------------:|---------------------------:|-----------------------:|--------------------------:|
| physiochemical_protein.csv | USE_XGB          |           0.62994 |                 0.00716682 |               0.495559 |                     0.125 |

## False Try Recommendations

| dataset                           | recommendation   |   preflight_score |   mgb_relative_improvement |   local_linearity_lift |   decoupled_leaf_fraction |
|:----------------------------------|:-----------------|------------------:|---------------------------:|-----------------------:|--------------------------:|
| airfoil_self_noise.csv            | TRY_MGB_LIGHT    |           1.5788  |                -0.081196   |               1.44671  |                     0.25  |
| concrete_compressive_strength.csv | TRY_MGB_LIGHT    |           1.98863 |                -0.00261404 |               1.14434  |                     0.75  |
| diamonds.csv                      | TRY_MGB_LIGHT    |           1.33968 |                -0.0130364  |               0.317528 |                     0.625 |

## Caveats

- This is a first-pass descriptive scorer, not a validated meta-model.
- The shallow-tree probe is a mechanism proxy for local one-feature responses.
- More completed binary/multiclass/drop/onehot result groups are needed before strong thresholds are credible.
