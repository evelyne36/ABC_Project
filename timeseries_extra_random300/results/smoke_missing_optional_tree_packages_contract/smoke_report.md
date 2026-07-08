# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_missing_optional_tree_packages_contract`

## Aggregate

| dataset   | model        |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:-------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | catboost_cpu |              nan |             nan |             nan |                         nan |                           nan |       1 |
| ETTh1     | lgbm_cpu     |              nan |             nan |             nan |                         nan |                           nan |       1 |

## Winner By Dataset

- ETTh1: catboost_cpu (mean RMSE=nan, folds=1)

## Split Audit

- split records: 0
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
