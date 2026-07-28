# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_timeseries_extra_etth1_20260627`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | xgb_cpu        |         0.92942  |             nan |        0.849218 |                  0.00580333 |                   0.000532333 |       1 |
| ETTh1     | current_metagb |         0.959157 |             nan |        0.870897 |                  0.264325   |                   0.0014225   |       1 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.92942, folds=1)

## Split Audit

- split records: 2
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
