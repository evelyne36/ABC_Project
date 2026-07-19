# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_seq_ltsf_remaining11_s6s12_p3p6_mgb10pf_20260629`

## Aggregate

| dataset     | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh2       | current_metagb |        1.36181   |      0.147552   |      0.987514   |                    3.40687  |                    0.00541775 |       2 |
| ETTh2       | xgb_cpu        |        1.62973   |      0.0292009  |      1.153      |                    0.610936 |                    0.00831623 |       2 |
| ETTm1       | xgb_cpu        |        1.05255   |      0.223439   |      0.850249   |                    0.324333 |                    0.00491052 |       2 |
| ETTm1       | current_metagb |        1.17482   |      0.467241   |      0.92042    |                    4.90212  |                    0.00844908 |       2 |
| ETTm2       | current_metagb |        0.585317  |      0.120807   |      0.490308   |                    5.11695  |                    0.00649844 |       2 |
| ETTm2       | xgb_cpu        |        0.617343  |      0.102739   |      0.508686   |                    0.197938 |                    0.00452856 |       2 |
| ILI         | xgb_cpu        |        0.398166  |      0.0540715  |      0.299493   |                    0.349687 |                    0.00545671 |       2 |
| ILI         | current_metagb |        0.495836  |      0.0493323  |      0.339684   |                    1.93742  |                    0.00612229 |       2 |
| PEMS03      | xgb_cpu        |        0.218218  |      0.00981388 |      0.178      |                    0.840231 |                    0.0087626  |       2 |
| PEMS03      | current_metagb |        0.219003  |      0.00116615 |      0.18085    |                    1.04381  |                    0.00612215 |       2 |
| PEMS04      | xgb_cpu        |        0.814085  |      0.102946   |      0.646188   |                    0.14548  |                    0.00564487 |       2 |
| PEMS04      | current_metagb |        0.826549  |      0.0809284  |      0.651768   |                    1.69484  |                    0.00791888 |       2 |
| PEMS07      | current_metagb |        0.345663  |      0.0165833  |      0.283274   |                    0.406352 |                    0.0055439  |       2 |
| PEMS07      | xgb_cpu        |        0.355238  |      0.0167276  |      0.293721   |                    0.236366 |                    0.00529444 |       2 |
| PEMS08      | current_metagb |        0.98297   |      0.0182183  |      0.67952    |                    0.443493 |                    0.00433048 |       2 |
| PEMS08      | xgb_cpu        |        1.11769   |      0.028108   |      0.737856   |                    0.259576 |                    0.00506785 |       2 |
| electricity | current_metagb |        0.751916  |      0.0535621  |      0.620355   |                    0.762204 |                    0.00670377 |       2 |
| electricity | xgb_cpu        |        0.758774  |      0.0692391  |      0.608769   |                    0.159919 |                    0.00511725 |       2 |
| solar       | xgb_cpu        |        0.0141233 |      0.0130875  |      0.00782391 |                    0.308102 |                    0.00541479 |       2 |
| solar       | current_metagb |        0.0146918 |      0.0134292  |      0.00809083 |                    5.00677  |                    0.00567623 |       2 |
| traffic     | xgb_cpu        |        0.844368  |      0.0689414  |      0.367329   |                    0.31836  |                    0.00595737 |       2 |
| traffic     | current_metagb |        0.951626  |      0.146743   |      0.419781   |                    0.439824 |                    0.00614958 |       2 |

## Winner By Dataset

- ETTh2: current_metagb (mean RMSE=1.36181, folds=2)
- ETTm1: xgb_cpu (mean RMSE=1.05255, folds=2)
- ETTm2: current_metagb (mean RMSE=0.585317, folds=2)
- ILI: xgb_cpu (mean RMSE=0.398166, folds=2)
- PEMS03: xgb_cpu (mean RMSE=0.218218, folds=2)
- PEMS04: xgb_cpu (mean RMSE=0.814085, folds=2)
- PEMS07: current_metagb (mean RMSE=0.345663, folds=2)
- PEMS08: current_metagb (mean RMSE=0.98297, folds=2)
- electricity: current_metagb (mean RMSE=0.751916, folds=2)
- solar: xgb_cpu (mean RMSE=0.0141233, folds=2)
- traffic: xgb_cpu (mean RMSE=0.844368, folds=2)

## Split Audit

- split records: 44
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
