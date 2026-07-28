# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_ts_all14_pressure_order_mgb_vs_xgb_20260628`

## Aggregate

| dataset       | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:--------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1         | xgb_cpu        |         0.749937 |             nan |        0.591746 |                  0.00398954 |                   0.000494292 |       1 |
| ETTh1         | current_metagb |         0.98856  |             nan |        0.829593 |                  0.0368595  |                   0.000673458 |       1 |
| ETTh2         | current_metagb |         1.44838  |             nan |        1.12106  |                  0.0201005  |                   0.00056025  |       1 |
| ETTh2         | xgb_cpu        |         1.85733  |             nan |        1.35811  |                  0.00391942 |                   0.000481585 |       1 |
| ETTm1         | xgb_cpu        |         0.838664 |             nan |        0.612452 |                  0.00387296 |                   0.000544124 |       1 |
| ETTm1         | current_metagb |         0.908471 |             nan |        0.655087 |                  0.0309592  |                   0.000636292 |       1 |
| ETTm2         | current_metagb |         0.226249 |             nan |        0.204702 |                  0.0209641  |                   0.000701291 |       1 |
| ETTm2         | xgb_cpu        |         0.230865 |             nan |        0.202402 |                  0.00642454 |                   0.000728666 |       1 |
| ILI           | current_metagb |         0.643296 |             nan |        0.520982 |                  0.0197393  |                   0.000613251 |       1 |
| ILI           | xgb_cpu        |         0.669335 |             nan |        0.544687 |                  0.00461904 |                   0.00058875  |       1 |
| PEMS03        | current_metagb |         0.995462 |             nan |        0.960425 |                  0.26059    |                   0.000803959 |       1 |
| PEMS03        | xgb_cpu        |         1.08871  |             nan |        1.05207  |                  0.0107469  |                   0.000606167 |       1 |
| PEMS04        | xgb_cpu        |         3.97649  |             nan |        3.70587  |                  0.00841583 |                   0.000566124 |       1 |
| PEMS04        | current_metagb |         4.44158  |             nan |        4.14672  |                  0.231273   |                   0.000811124 |       1 |
| PEMS07        | xgb_cpu        |         0.750988 |             nan |        0.679113 |                  0.00917262 |                   0.000548084 |       1 |
| PEMS07        | current_metagb |         0.816517 |             nan |        0.745958 |                  0.271147   |                   0.000925626 |       1 |
| PEMS08        | xgb_cpu        |         2.75737  |             nan |        2.10583  |                  0.006772   |                   0.000500166 |       1 |
| PEMS08        | current_metagb |         2.85354  |             nan |        2.17127  |                  0.29082    |                   0.000949167 |       1 |
| Weather       | current_metagb |         1.10053  |             nan |        0.88198  |                  0.12854    |                   0.000777542 |       1 |
| Weather       | xgb_cpu        |         1.38965  |             nan |        1.11539  |                  0.00903667 |                   0.0005905   |       1 |
| electricity   | xgb_cpu        |         0.404263 |             nan |        0.32175  |                  0.009      |                   0.000526708 |       1 |
| electricity   | current_metagb |         0.422354 |             nan |        0.338653 |                  0.456741   |                   0.000878999 |       1 |
| exchange_rate | current_metagb |         2.15707  |             nan |        1.99813  |                  0.0194302  |                   0.000568083 |       1 |
| exchange_rate | xgb_cpu        |         4.22759  |             nan |        3.8371   |                  0.00499725 |                   0.000525208 |       1 |
| solar         | current_metagb |         0.23203  |             nan |        0.23203  |                  0.26553    |                   0.00089225  |       1 |
| solar         | xgb_cpu        |         0.235246 |             nan |        0.235246 |                  0.00674283 |                   0.00055675  |       1 |
| traffic       | xgb_cpu        |         0.546004 |             nan |        0.467646 |                  0.00988946 |                   0.000595834 |       1 |
| traffic       | current_metagb |         0.645502 |             nan |        0.553522 |                  0.535522   |                   0.000884457 |       1 |

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
