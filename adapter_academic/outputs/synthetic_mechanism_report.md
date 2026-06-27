# Synthetic Mechanism Recovery Report

## Contract

- Scores are computed without outcome labels.
- Synthetic labels indicate whether the data generator contains a MetaGB-aligned mechanism.
- Feature-hit rate uses known true predictive features.

## Scenario Summary

| scenario                          |   dimension |   expected_mechanism_match |   score_mean |   score_ci95_low |   score_ci95_high |   feature_hit_rate_mean |   feature_hit_ci95_low |   feature_hit_ci95_high |
|:----------------------------------|------------:|---------------------------:|-------------:|-----------------:|------------------:|------------------------:|-----------------------:|------------------------:|
| piecewise_constant                |          10 |                          0 |     0.185988 |         0.157929 |          0.21852  |               nan       |              nan       |               nan       |
| piecewise_constant                |          50 |                          0 |     0.319623 |         0.310085 |          0.331504 |               nan       |              nan       |               nan       |
| local_linear_response             |          10 |                          1 |     0.7625   |         0.75     |          0.7875   |                 0.91    |                0.8     |                 0.98    |
| local_linear_response             |          50 |                          1 |     0.761718 |         0.748566 |          0.786879 |                 0.92    |                0.8     |                 0.99    |
| conditional_predictive_decoupling |          10 |                          1 |     0.75     |         0.75     |          0.75     |                 0.89375 |                0.8625  |                 0.925   |
| conditional_predictive_decoupling |          50 |                          1 |     0.748181 |         0.746409 |          0.749664 |                 0.9     |                0.8625  |                 0.93125 |
| high_dim_sparse_noise             |          20 |                          1 |     0.75     |         0.75     |          0.75     |                 0.9     |                0.86875 |                 0.93125 |
| high_dim_sparse_noise             |         100 |                          1 |     0.726704 |         0.723685 |          0.729502 |                 0.925   |                0.89375 |                 0.95625 |
| high_dim_sparse_noise             |         300 |                          1 |     0.67759  |         0.675914 |          0.679035 |                 0.91875 |                0.88125 |                 0.95625 |

## Interpretation

- `piecewise_constant` should score low because constant leaves already match the generator.
- `local_linear_response` and `conditional_predictive_decoupling` should score high.
- `high_dim_sparse_noise` should remain high if the projected-gradient signal is robust to irrelevant features.
