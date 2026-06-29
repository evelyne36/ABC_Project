# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_ts_parallel_workers2_20260628b`

## Aggregate

| dataset       | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:--------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ILI           | current_metagb |         0.538364 |             nan |        0.328682 |                   0.915718  |                   0.00091175  |       1 |
| ILI           | xgb_cpu        |         0.589339 |             nan |        0.421691 |                   0.0307196 |                   0.000637085 |       1 |
| exchange_rate | xgb_cpu        |         1.94102  |             nan |        1.75051  |                   0.0293965 |                   0.000599125 |       1 |
| exchange_rate | current_metagb |         2.1489   |             nan |        1.98025  |                   0.329082  |                   0.000630167 |       1 |

## Winner By Dataset

- ILI: current_metagb (mean RMSE=0.538364, folds=1)
- exchange_rate: xgb_cpu (mean RMSE=1.94102, folds=1)

## Split Audit

- split records: 4
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
