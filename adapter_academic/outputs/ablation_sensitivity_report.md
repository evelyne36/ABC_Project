# Ablation And Sensitivity Report

## Ablation

| variant                         | removed_component                  |   auc |   mean_score |
|:--------------------------------|:-----------------------------------|------:|-------------:|
| full                            |                                    |     1 |     0.631367 |
| remove_linear_leaf_advantage    | component_linear_leaf_advantage    |     1 |     0.540677 |
| remove_cond_pred_decoupling     | component_cond_pred_decoupling     |     1 |     0.785477 |
| remove_local_heterogeneity      | component_local_heterogeneity      |     1 |     0.594764 |
| remove_predictive_concentration | component_predictive_concentration |     1 |     0.623015 |
| remove_signal_support           | component_signal_support           |     1 |     0.609328 |

## Sensitivity

|   weight_linear |   weight_decoupling |   auc |   mean_expected_positive_score |   mean_expected_negative_score |   score_gap |
|----------------:|--------------------:|------:|-------------------------------:|-------------------------------:|------------:|
|            0.2  |                0.15 |     1 |                       0.784132 |                       0.222506 |    0.561626 |
|            0.2  |                0.25 |     1 |                       0.693562 |                       0.279937 |    0.413625 |
|            0.2  |                0.35 |     1 |                       0.622059 |                       0.325276 |    0.296783 |
|            0.35 |                0.15 |     1 |                       0.82011  |                       0.201932 |    0.618178 |
|            0.35 |                0.25 |     1 |                       0.739528 |                       0.252805 |    0.486722 |
|            0.35 |                0.35 |     1 |                       0.673596 |                       0.294429 |    0.379168 |
|            0.5  |                0.15 |     1 |                       0.845809 |                       0.187236 |    0.658573 |
|            0.5  |                0.25 |     1 |                       0.773502 |                       0.232751 |    0.540751 |
|            0.5  |                0.35 |     1 |                       0.712765 |                       0.270984 |    0.44178  |

## Interpretation

- Ablation tests whether each mechanism component is necessary for synthetic mechanism recovery.
- Sensitivity checks whether the score is robust to plausible theory-weight changes.
- These checks address the NeurIPS-style concern that the score is an arbitrary heuristic.
