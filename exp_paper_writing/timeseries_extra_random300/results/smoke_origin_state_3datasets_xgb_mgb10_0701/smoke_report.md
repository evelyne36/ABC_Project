# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_origin_state_3datasets_xgb_mgb10_0701`

## Aggregate

| dataset     | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1       | current_metagb |         0.79521  |             nan |        0.643158 |                  0.297055   |                   0.000658959 |       1 |
| ETTh1       | xgb_cpu        |         0.858778 |             nan |        0.673467 |                  0.00824446 |                   0.000382417 |       1 |
| electricity | xgb_cpu        |         0.405083 |             nan |        0.312722 |                  0.017322   |                   0.000520583 |       1 |
| electricity | current_metagb |         0.410277 |             nan |        0.327236 |                  1.34288    |                   0.000572541 |       1 |
| traffic     | xgb_cpu        |         0.605479 |             nan |        0.530299 |                  0.0186488  |                   0.000425542 |       1 |
| traffic     | current_metagb |         0.862231 |             nan |        0.771772 |                  0.18887    |                   0.000567542 |       1 |

## Winner By Dataset

- ETTh1: current_metagb (mean RMSE=0.79521, folds=1)
- electricity: xgb_cpu (mean RMSE=0.405083, folds=1)
- traffic: xgb_cpu (mean RMSE=0.605479, folds=1)

## Split Audit

- split records: 6
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
