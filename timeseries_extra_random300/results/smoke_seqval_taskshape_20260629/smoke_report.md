# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_seqval_taskshape_20260629`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | current_metagb |         0.657529 |       0.127738  |        0.479608 |                   1.40464   |                   0.000629188 |       2 |
| ETTh1     | xgb_cpu        |         0.662819 |       0.0513799 |        0.472396 |                   0.0499349 |                   0.000655166 |       2 |

## Winner By Dataset

- ETTh1: current_metagb (mean RMSE=0.657529, folds=2)

## Split Audit

- split records: 4
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
