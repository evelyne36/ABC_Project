# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_fixed_protocol_etth1_xgb_20260628`

## Aggregate

| dataset   | model   |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:--------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | xgb_cpu |         0.738579 |        0.292705 |        0.509688 |                    0.128065 |                   0.000915011 |       8 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.738579, folds=8)

## Split Audit

- split records: 8
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
