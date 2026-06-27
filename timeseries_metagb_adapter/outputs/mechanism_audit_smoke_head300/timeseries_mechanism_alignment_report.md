# Time-Series MetaGB Mechanism Audit

## Contract

- Mechanism scores are computed from the first outer training block only.
- Smoke outcomes are joined afterwards as retrospective audit evidence.
- This report does not modify or import outputs into `metagb_preflight_adapter/`.

## Result Table

| dataset       | mechanism_band           |   mechanism_alignment_score |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_predictive_concentration |   component_signal_support |   mgb_relative_rmse_improvement | winner         |
|:--------------|:-------------------------|----------------------------:|----------------------------------:|---------------------------------:|--------------------------------:|-------------------------------------:|---------------------------:|--------------------------------:|:---------------|
| ETTh1         | WEAK_MECHANISM_MATCH     |                    0.356713 |                          0.265933 |                                1 |                               0 |                                    0 |                          1 |                       -0.216272 | xgb_cpu        |
| exchange_rate | MODERATE_MECHANISM_MATCH |                    0.55     |                          1        |                                1 |                               0 |                                    0 |                          1 |                        0.28351  | current_metagb |

## Interpretation

- exchange_rate: score=0.550, band=MODERATE_MECHANISM_MATCH, smoke winner=current_metagb, relative RMSE improvement=0.284.
- ETTh1: score=0.357, band=WEAK_MECHANISM_MATCH, smoke winner=xgb_cpu, relative RMSE improvement=-0.216.

## Caveats

- summary input: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_metagb_adapter/results/smoke_2datasets_2x2_30_head300_tinytrees_20260624/summary.csv`
- max_raw_rows: 300
- The completed smoke used tiny trees and two outer folds, so outcome evidence is weak for final ranking.
- Candidate test metrics are available only for selected candidates, so this cannot estimate an oracle search gap.
