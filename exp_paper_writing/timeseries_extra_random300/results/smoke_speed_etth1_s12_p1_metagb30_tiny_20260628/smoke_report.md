# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_speed_etth1_s12_p1_metagb30_tiny_20260628`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | current_metagb |         0.736777 |             nan |        0.584231 |                      3.5698 |                   0.000738376 |       1 |

## Winner By Dataset

- ETTh1: current_metagb (mean RMSE=0.736777, folds=1)

## Split Audit

- split records: 1
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
