# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_seq_ltsf_3datasets_s6s12_p3p6_mgb10pf_20260629`

## Aggregate

| dataset       | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:--------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1         | xgb_cpu        |         0.788102 |       0.0103414 |        0.605155 |                    0.182668 |                    0.0031294  |       2 |
| ETTh1         | current_metagb |         0.793173 |       0.0299204 |        0.59648  |                   14.5337   |                    0.00580463 |       2 |
| Weather       | current_metagb |         1.34226  |       0.0875471 |        0.865624 |                    4.42722  |                    0.00395431 |       2 |
| Weather       | xgb_cpu        |         1.34238  |       0.0341738 |        1.02821  |                    0.43312  |                    0.00397312 |       2 |
| exchange_rate | current_metagb |         1.30864  |       0.0691392 |        1.07841  |                    3.65375  |                    0.00350571 |       2 |
| exchange_rate | xgb_cpu        |         1.6693   |       0.0352294 |        1.41931  |                    0.523499 |                    0.00419573 |       2 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.788102, folds=2)
- Weather: current_metagb (mean RMSE=1.34226, folds=2)
- exchange_rate: current_metagb (mean RMSE=1.30864, folds=2)

## Split Audit

- split records: 12
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
