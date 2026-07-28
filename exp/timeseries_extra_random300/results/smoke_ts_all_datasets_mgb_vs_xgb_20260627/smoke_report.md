# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_ts_all_datasets_mgb_vs_xgb_20260627`

## Aggregate

| dataset       | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:--------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1         | xgb_cpu        |         0.749937 |             nan |        0.591746 |                  0.00393375 |                   0.000526292 |       1 |
| ETTh1         | current_metagb |         0.98856  |             nan |        0.829593 |                  0.0356795  |                   0.000629332 |       1 |
| ETTh2         | current_metagb |         1.44838  |             nan |        1.12106  |                  0.0191488  |                   0.000575001 |       1 |
| ETTh2         | xgb_cpu        |         1.85733  |             nan |        1.35811  |                  0.00389804 |                   0.000488959 |       1 |
| ETTm1         | xgb_cpu        |         0.838664 |             nan |        0.612452 |                  0.00420446 |                   0.000609    |       1 |
| ETTm1         | current_metagb |         0.908471 |             nan |        0.655087 |                  0.0299353  |                   0.000647417 |       1 |
| ETTm2         | current_metagb |         0.226249 |             nan |        0.204702 |                  0.0192061  |                   0.000569916 |       1 |
| ETTm2         | xgb_cpu        |         0.230865 |             nan |        0.202402 |                  0.00457592 |                   0.000522126 |       1 |
| ILI           | current_metagb |         0.643296 |             nan |        0.520982 |                  0.021461   |                   0.000649084 |       1 |
| ILI           | xgb_cpu        |         0.669335 |             nan |        0.544687 |                  0.00468967 |                   0.000533958 |       1 |
| PEMS03        | current_metagb |         0.995462 |             nan |        0.960425 |                  0.261167   |                   0.000853666 |       1 |
| PEMS03        | xgb_cpu        |         1.08871  |             nan |        1.05207  |                  0.0106081  |                   0.000545    |       1 |
| PEMS04        | xgb_cpu        |         3.97649  |             nan |        3.70587  |                  0.00773596 |                   0.000494501 |       1 |
| PEMS04        | current_metagb |         4.44158  |             nan |        4.14672  |                  0.227966   |                   0.000759168 |       1 |
| PEMS07        | xgb_cpu        |         0.750988 |             nan |        0.679113 |                  0.008587   |                   0.000527375 |       1 |
| PEMS07        | current_metagb |         0.816517 |             nan |        0.745958 |                  0.264729   |                   0.000798751 |       1 |
| PEMS08        | xgb_cpu        |         2.75737  |             nan |        2.10583  |                  0.00615492 |                   0.000453083 |       1 |
| PEMS08        | current_metagb |         2.85354  |             nan |        2.17127  |                  0.27908    |                   0.000777417 |       1 |
| Weather       | current_metagb |         1.10053  |             nan |        0.88198  |                  0.128077   |                   0.000836292 |       1 |
| Weather       | xgb_cpu        |         1.38965  |             nan |        1.11539  |                  0.00881483 |                   0.000612792 |       1 |
| electricity   | xgb_cpu        |         0.404263 |             nan |        0.32175  |                  0.00875429 |                   0.000511874 |       1 |
| electricity   | current_metagb |         0.422354 |             nan |        0.338653 |                  0.431341   |                   0.000994375 |       1 |
| exchange_rate | current_metagb |         2.15707  |             nan |        1.99813  |                  0.0216236  |                   0.000794375 |       1 |
| exchange_rate | xgb_cpu        |         4.22759  |             nan |        3.8371   |                  0.00460488 |                   0.000473124 |       1 |
| solar         | current_metagb |         0.23203  |             nan |        0.23203  |                  0.264699   |                   0.000808959 |       1 |
| solar         | xgb_cpu        |         0.235246 |             nan |        0.235246 |                  0.00667596 |                   0.000522374 |       1 |
| traffic       | xgb_cpu        |         0.546004 |             nan |        0.467646 |                  0.00984767 |                   0.000545416 |       1 |
| traffic       | current_metagb |         0.645502 |             nan |        0.553522 |                  0.528669   |                   0.000847584 |       1 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.749937, folds=1)
- ETTh2: current_metagb (mean RMSE=1.44838, folds=1)
- ETTm1: xgb_cpu (mean RMSE=0.838664, folds=1)
- ETTm2: current_metagb (mean RMSE=0.226249, folds=1)
- ILI: current_metagb (mean RMSE=0.643296, folds=1)
- PEMS03: current_metagb (mean RMSE=0.995462, folds=1)
- PEMS04: xgb_cpu (mean RMSE=3.97649, folds=1)
- PEMS07: xgb_cpu (mean RMSE=0.750988, folds=1)
- PEMS08: xgb_cpu (mean RMSE=2.75737, folds=1)
- Weather: current_metagb (mean RMSE=1.10053, folds=1)
- electricity: xgb_cpu (mean RMSE=0.404263, folds=1)
- exchange_rate: current_metagb (mean RMSE=2.15707, folds=1)
- solar: current_metagb (mean RMSE=0.23203, folds=1)
- traffic: xgb_cpu (mean RMSE=0.546004, folds=1)

## Split Audit

- split records: 28
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
