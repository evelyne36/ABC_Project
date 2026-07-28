# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_sequence_window_summary_tree_contract`

## Aggregate

| dataset   | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:----------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1     | xgb_cpu        |         0.772006 |             nan |        0.579269 |                  0.00692717 |                    0.00047875 |       1 |
| ETTh1     | current_metagb |         0.838433 |             nan |        0.649065 |                  0.128187   |                    0.0006195  |       1 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.772006, folds=1)

## Split Audit

- split records: 2
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
