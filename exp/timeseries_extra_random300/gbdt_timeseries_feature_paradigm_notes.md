# GBDT-Based Tabular Models for Time Series: Paradigm Notes

Date: 2026-06-30

## Core Position

For a GBDT-based time-series adapter, the right paradigm is not to copy the
TabPFN-TS timestamp-only protocol. TabPFN can use Transformer-style in-context
learning across historical rows and future query rows, although vanilla TabPFN
still needs time-aware features or position encoding to know temporal order.

GBDT does not have such row-context inference at prediction time. A tree model
sees one feature row at a time. Therefore temporal dependence and cross-channel
dependence must be explicitly converted into columns.

Recommended main paradigm:

```text
global pooled + direct multi-horizon GBDT
+ lag-window features
+ rolling/window summary features
+ calendar/Fourier/seasonal features
+ cross-channel lag summaries or selected cross-channel lags
+ horizon_id, variable_id, series_id/static features
```

In this paradigm:

```text
one row = one forecast origin / target variable / horizon task
columns = all information available at that forecast origin
target = y[target_variable, origin + horizon]
```

## Why Lag-Window Features Are Necessary for GBDT

GBDT cannot attend over a separate historical context the way TabPFN does. It can
only split on current row columns. Thus sequence structure must be materialized:

- own-variable lags: `y_j[t-1], y_j[t-2], ...`
- seasonal lags: `y_j[t-s], y_j[t-2s], ...`
- differences: `y_j[t] - y_j[t-1]`, `y_j[t] - y_j[t-s]`
- rolling/window summaries: mean, std, min, max, slope, quantiles, EWMA
- cross-channel lags: `y_k[t-l] -> y_j[t+h]`
- cross-channel summaries: mean/std/top-k correlated variable summaries
- future-known features: calendar, holidays, known dynamic covariates

This is not a weak workaround. It is the standard supervised framing /
propositionalization route for using tabular models on time-series data.

## View on Mean / Rolling / Window Summary Features

Simple window means are valuable but should be treated as one member of a
structured feature family, not as arbitrary manual tricks.

Useful window summaries answer different questions:

- mean / median: local level
- std / IQR / min-max: local volatility and range
- slope / first-last difference: local trend
- EWMA: recency-weighted level
- seasonal-window mean: repeated seasonal state
- quantiles: robust distributional state
- missing-rate or mask summaries: data quality / availability state

For high-dimensional multichannel series, summaries also reduce dimension:

```text
raw wide lag features:       O(num_channels * lag_length)
summary per channel group:   O(num_groups * num_windows * num_stats)
selected top-k channels:     O(top_k * selected_lags)
factor/PCA summaries:        O(num_factors * selected_lags_or_stats)
```

The safest design is a mixed representation:

1. Keep exact recent lags for the target variable.
2. Keep exact seasonal lags for the target variable.
3. For other variables, avoid full flattening unless channel count is small.
4. Use cross-channel summaries, top-k correlated channels, or latent factors.
5. Add calendar/Fourier features for future timestamps.

## Automated / Less-Handcrafted Feature Engineering Literature

Relevant families:

1. Massive automated feature extraction:
   - hctsa extracts thousands of scientific time-series features.
   - catch22 distills hctsa-like features to 22 low-redundancy canonical
     features with much lower cost.
   - tsfresh extracts many time-series features and uses hypothesis tests for
     relevance filtering.

2. Random convolutional transforms:
   - ROCKET / MiniRocket generate many convolutional-kernel features with very
     little manual feature design.
   - The resulting feature matrix can be consumed by linear models, and the same
     idea can be adapted as a feature block for GBDT.

3. Forecasting meta-features:
   - FFORMA-style work uses time-series features and XGBoost-like meta-learning
     to choose or weight forecasting models.
   - This supports the claim that general time-series feature summaries carry
     useful forecasting signal.

4. Automated tabular feature engineering:
   - Deep Feature Synthesis / OneBM-style systems generate aggregates and
     transformations automatically.
   - Recent tabular work such as TabPrep argues that systematic feature
     generation can improve even strong models, including tree-based and
     foundation models.

## Experimental Implication

For this branch, the most defensible story is:

> TabPFN-TS demonstrates a timestamp-feature + Transformer-ICL route. Our GBDT
> route uses lag-window propositionalization plus systematic window summaries.
> The two methods differ in where temporal and cross-channel structure is
> represented: TabPFN leaves more to context attention, while GBDT requires the
> structure to be explicitly materialized as columns.

Suggested ablation ladder:

1. calendar/Fourier only
2. target lags only
3. target lags + rolling summaries
4. target lags + calendar/Fourier
5. target lags + cross-channel summaries
6. target lags + selected cross-channel exact lags
7. automated feature block: catch22/tsfresh/ROCKET-style summaries
8. full mixed representation

## Decision Record for the Next Implementation Pass

The implementation should not be described as sometimes adding features and
sometimes deleting features. The unified story is forecast-origin state
representation:

```text
time-series forecasting for GBDT =
  learn f(state at forecast origin, horizon, target identity) -> future value
```

For a historical window `W_t in R^(L x C)`, construct:

```text
phi(W_t) = [phi_fine(W_t), phi_stat(W_t), phi_rel(W_t)]
```

where:

- `phi_fine`: exact lag values that preserve local temporal order;
- `phi_stat`: rolling/window summaries that encode local level, volatility,
  trend, and distributional state;
- `phi_rel`: cross-channel summaries, selected related channels, or low-rank
  factors that encode variable relationships.

The central constraint is a feature budget:

```text
dim(phi(W_t)) <= B
```

When `L * C` is small enough, `phi_fine` can include all channels, so the
representation looks additive: all lags plus summaries. When `L * C` is too
large, fine-grained lag capacity is allocated to the target channel and a small
set of selected channels, while the remaining channels enter through
statistical and relational summaries. This is not a different method; it is the
same state-representation principle with a different resolution allocation.

We should implement a modest but meaningful first version, not a giant feature
factory. The first pass should keep the existing lag-window flattening behavior
as a baseline and add an explicit forecast-origin state mode that is:

- deterministic;
- leakage-safe, using only the input context window;
- bounded in dimensionality;
- identical for `current_metagb` and `xgb_cpu`;
- optional/configurable so the original flatten-only baseline remains runnable.

The target architecture for the first pass:

```text
feature_mode=origin_state:
  target-channel exact lags
  selected exact lags from a bounded number of train-correlated channels
  per-selected-channel temporal summaries
  per-lag and global cross-channel relational summaries across all available channels
  optional calendar/Fourier timestamp features

feature_mode=window_summary:
  retained as the additive low-dimensional diagnostic mode

explicitly deferred:
  catch22/tsfresh/ROCKET feature blocks
  top-k correlation channel selection
  PCA/factor features
  horizon-as-feature protocol rewrite
```

This implementation is intentionally conservative. It tests the central claim
that systematic forecast-origin state construction can make temporal and
cross-variable structure legible to GBDT-style tabular learners under a
controlled feature budget. The more automated feature blocks can be added after
this baseline is stable.

### 2026-07-01 Adjustment

The first smoke with `origin_state_channel_budget=16` showed that traffic was
too aggressively compressed for `current_metagb`: exact lags covered only 16 of
862 channels, then MetaGB could additionally sub-sample the already compressed
state. The adjusted implementation expands the default budget to 32, selects
the exact-lag channels by train-only absolute correlation with the target, and
adds per-lag all-channel mean/std relational features. This keeps the same
state-representation narrative while increasing fine-grained and relational
resolution.

Representative smoke evidence (`max_raw_rows=160`, `seq_len=6`, `pred_len=2`,
10 XGB candidates and 10 MetaGB candidates with pred fractions 0.1..1.0):

| dataset | raw channels | input dim | MGB RMSE | XGB RMSE | note |
|---|---:|---:|---:|---:|---|
| ETTh1 | 7 | 129 | 0.8909 | 0.8712 | tiny smoke; expansion not clearly helpful |
| electricity | 321 | 500 | 0.4984 | 0.4680 | tiny smoke; expansion not clearly helpful |
| traffic | 862 | 500 | 0.5671 | 0.6159 | expansion substantially improved MGB |

This is not a quality claim because the smoke uses very small training windows
and tiny trees. It does indicate that high-channel datasets need enough
origin-state resolution, while low-dimensional datasets may prefer a smaller
budget. The next stable experiment should sweep budget values such as 16, 32,
and 64 instead of treating one budget as universally optimal.

## Sources To Cite

- TabPFN-TS: https://arxiv.org/abs/2501.02945
- ApolloPFN / time-aware PFN critique: https://arxiv.org/abs/2603.15802
- Global forecasting locality/globality: https://arxiv.org/abs/2008.00444
- Global models simulation with LightGBM: https://arxiv.org/abs/2012.12485
- Multi-step forecasting strategies: https://arxiv.org/abs/1108.3259
- catch22: https://arxiv.org/abs/1901.10200
- hctsa / automatic massive feature extraction: https://arxiv.org/abs/1612.05296
- feature-based time-series analysis: https://arxiv.org/abs/1709.08055
- ROCKET: https://arxiv.org/abs/1910.13051
- MiniRocket: https://arxiv.org/abs/2012.08791
- One Button Machine / automated feature engineering: https://arxiv.org/abs/1706.00327
- TabPrep / systematic feature generation for tabular benchmarks:
  https://arxiv.org/abs/2606.02384
