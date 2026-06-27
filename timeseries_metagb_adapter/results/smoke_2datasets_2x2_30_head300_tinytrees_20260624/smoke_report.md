# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_metagb_adapter/results/smoke_2datasets_2x2_30_head300_tinytrees_20260624`

## Aggregate

| dataset       | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:--------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1         | xgb_cpu        |         0.575888 |        0.260893 |        0.456153 |                   0.0151921 |                   0.000766916 |       2 |
| ETTh1         | current_metagb |         0.700436 |        0.224019 |        0.574729 |                   0.785743  |                   0.000917167 |       2 |
| exchange_rate | current_metagb |         1.3345   |        1.15677  |        1.26793  |                   0.661534  |                   0.000969957 |       2 |
| exchange_rate | xgb_cpu        |         1.86255  |        2.10244  |        1.71078  |                   0.018579  |                   0.000734166 |       2 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.575888, folds=2)
- exchange_rate: current_metagb (mean RMSE=1.3345, folds=2)

## Split Audit

- split records: 8
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
