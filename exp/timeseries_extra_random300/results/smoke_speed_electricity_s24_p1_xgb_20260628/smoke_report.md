# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_speed_electricity_s24_p1_xgb_20260628`

## Aggregate

| dataset     | model   |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:------------|:--------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| electricity | xgb_cpu |         0.383441 |             nan |        0.299396 |                      13.252 |                     0.0080095 |       1 |

## Winner By Dataset

- electricity: xgb_cpu (mean RMSE=0.383441, folds=1)

## Split Audit

- split records: 1
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
