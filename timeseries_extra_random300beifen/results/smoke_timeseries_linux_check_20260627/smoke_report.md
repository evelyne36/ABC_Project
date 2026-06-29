# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_timeseries_linux_check_20260627`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ILI       | current_metagb |          1.17579 |             nan |         1.05973 |                  0.0629404  |                   0.000619832 |       1 |
| ILI       | xgb_cpu        |          1.1892  |             nan |         1.0204  |                  0.00770133 |                   0.000494625 |       1 |

## Winner By Dataset

- ILI: current_metagb (mean RMSE=1.17579, folds=1)

## Split Audit

- split records: 2
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
