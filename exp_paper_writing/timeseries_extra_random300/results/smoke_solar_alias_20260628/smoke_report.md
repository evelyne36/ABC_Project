# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_solar_alias_20260628`

## Aggregate

| dataset   | model   |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:--------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| solar     | xgb_cpu |          7.26296 |             nan |         6.08158 |                   0.0156841 |                    0.00215537 |       1 |

## Winner By Dataset

- solar: xgb_cpu (mean RMSE=7.26296, folds=1)

## Split Audit

- split records: 1
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
