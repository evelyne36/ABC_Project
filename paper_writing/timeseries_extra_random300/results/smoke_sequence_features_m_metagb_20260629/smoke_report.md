# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_sequence_features_m_metagb_20260629`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | current_metagb |          1.02265 |             nan |        0.877343 |                    0.102033 |                    0.00146588 |       1 |

## Winner By Dataset

- ETTh1: current_metagb (mean RMSE=1.02265, folds=1)

## Split Audit

- split records: 1
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
