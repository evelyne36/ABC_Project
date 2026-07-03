# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_origin_state_expanded_3datasets_xgb_mgb10_0701`

## Aggregate

| dataset     | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1       | xgb_cpu        |         0.87119  |             nan |        0.693553 |                   0.0100074 |                   0.000423668 |       1 |
| ETTh1       | current_metagb |         0.890948 |             nan |        0.702309 |                   0.365706  |                   0.000586416 |       1 |
| electricity | xgb_cpu        |         0.467953 |             nan |        0.368153 |                   0.0334248 |                   0.000647376 |       1 |
| electricity | current_metagb |         0.498418 |             nan |        0.421307 |                   5.63345   |                   0.000687458 |       1 |
| traffic     | current_metagb |         0.567136 |             nan |        0.458327 |                   4.66113   |                   0.000679208 |       1 |
| traffic     | xgb_cpu        |         0.61586  |             nan |        0.532759 |                   0.0359658 |                   0.000490792 |       1 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.87119, folds=1)
- electricity: xgb_cpu (mean RMSE=0.467953, folds=1)
- traffic: current_metagb (mean RMSE=0.567136, folds=1)

## Split Audit

- split records: 6
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
