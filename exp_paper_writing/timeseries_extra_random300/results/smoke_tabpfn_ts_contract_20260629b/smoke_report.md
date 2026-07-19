# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_tabpfn_ts_contract_20260629b`

## Aggregate

| dataset   | model   |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:--------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| solar     | xgb_cpu |          3.51791 |             nan |         3.08435 |                   0.0125568 |                    0.00116079 |       1 |

## Winner By Dataset

- solar: xgb_cpu (mean RMSE=3.51791, folds=1)

## Split Audit

- split records: 1
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
