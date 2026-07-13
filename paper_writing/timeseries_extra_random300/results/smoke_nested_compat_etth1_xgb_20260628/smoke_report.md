# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_nested_compat_etth1_xgb_20260628`

## Aggregate

| dataset   | model   |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:--------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | xgb_cpu |         0.676275 |             nan |        0.487722 |                   0.0177316 |                   0.000743001 |       1 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.676275, folds=1)

## Split Audit

- split records: 1
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
