# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_seed_batch_order_20260629`

## Aggregate

| dataset   | model   |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:--------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | xgb_cpu |         0.742483 |        0.192729 |        0.653762 |                   0.0364262 |                    0.00118915 |       2 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.742483, folds=2)

## Split Audit

- split records: 2
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
