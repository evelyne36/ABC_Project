# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_window_summary_xgb_mgb_0630`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | current_metagb |         0.753289 |             nan |        0.638512 |                   0.114363  |                    0.00136342 |       1 |
| ETTh1     | xgb_cpu        |         0.784846 |             nan |        0.677101 |                   0.0157473 |                    0.00123379 |       1 |

## Winner By Dataset

- ETTh1: current_metagb (mean RMSE=0.753289, folds=1)

## Split Audit

- split records: 2
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
