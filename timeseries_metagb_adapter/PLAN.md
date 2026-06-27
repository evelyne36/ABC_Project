# Time-Series MetaGB Adapter Plan

## Goal

Explore whether MetaGB remains useful after converting ordered time-series forecasting
data into tabular supervised windows, while preserving the evaluation constraints used
by long-term time-series forecasting benchmarks.

## References

- Informer, AAAI 2021: official implementation uses ETT data, zero-mean
  normalization, and ordered train/validation/test windows.
- Autoformer, NeurIPS 2021: long-term forecasting benchmark line covering energy,
  traffic, economics, weather, and disease datasets.
- PatchTST, ICLR 2023: supervised long-term forecasting implementation with
  patch/window construction and channel-independent forecasting.
- ETDataset: ETTh1, ETTh2, ETTm1, ETTm2 CSV datasets.
- LSTNet multivariate time-series data: exchange_rate, electricity, traffic, solar.

## Design Decisions

1. Keep this work isolated under `timeseries_metagb_adapter/`.
2. Reuse the repository's existing XGBoost/MetaGB estimator construction and paired
   candidate generation.
3. Replace shuffled KFold with ordered expanding-window nested CV.
4. Fit `StandardScaler` separately inside each fold using only that fold's training
   target period.
5. Use RMSE as the primary selection and test metric, with MAE/MSE also reported.

## Default Protocol

- Smoke: 2 datasets, 2 outer folds, 2 inner folds, 30 candidates.
- Full: defaults to 5 outer folds, 4 inner folds, 300 candidates.
- Default smoke datasets: `ETTh1`, `exchange_rate`.
- Registry keeps the union of follow-up datasets: `ETTh1`, `ETTh2`, `ETTm1`,
  `ETTm2`, `exchange_rate`, `electricity`, `traffic`, `solar`.

## Acceptance Criteria

- No split uses shuffled indices.
- Every validation/test block is strictly later than its training block.
- Every scaler fit ends at or before the training target period and before the
  validation/test target period.
- Smoke run writes `summary.csv`, `candidates.csv`, `aggregate.csv`,
  `split_manifest.json`, `parameter_audit.json`, and `smoke_report.md`.
- Smoke run completes with 8 summary rows and 240 candidate rows for
  2 datasets x 2 folds x 2 models x 30 candidates.

