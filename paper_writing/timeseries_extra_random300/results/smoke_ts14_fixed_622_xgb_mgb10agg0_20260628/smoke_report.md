# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0623-adapter/timeseries_extra_random300/results/smoke_ts14_fixed_622_xgb_mgb10agg0_20260628`

## Aggregate

| dataset       | model          |   test_rmse_mean |   test_rmse_std |   test_mae_mean |   total_train_time_sec_mean |   total_predict_time_sec_mean |   folds |
|:--------------|:---------------|-----------------:|----------------:|----------------:|----------------------------:|------------------------------:|--------:|
| ETTh1         | xgb_cpu        |        0.633613  |             nan |       0.533048  |                   0.138195  |                   0.00120608  |       1 |
| ETTh1         | current_metagb |        0.677425  |             nan |       0.573166  |                   0.49928   |                   0.000617709 |       1 |
| ETTh2         | xgb_cpu        |        0.370351  |             nan |       0.321939  |                   0.104529  |                   0.000917626 |       1 |
| ETTh2         | current_metagb |        0.417555  |             nan |       0.339257  |                   0.288815  |                   0.000679667 |       1 |
| ETTm1         | xgb_cpu        |        0.288076  |             nan |       0.225382  |                   0.0110158 |                   0.000438709 |       1 |
| ETTm1         | current_metagb |        0.307894  |             nan |       0.252503  |                   1.86978   |                   0.000690626 |       1 |
| ETTm2         | xgb_cpu        |        0.172995  |             nan |       0.153981  |                   0.043581  |                   0.00054225  |       1 |
| ETTm2         | current_metagb |        0.263334  |             nan |       0.258636  |                   0.261392  |                   0.000656334 |       1 |
| ILI           | current_metagb |        0.699969  |             nan |       0.434295  |                   0.453245  |                   0.000632209 |       1 |
| ILI           | xgb_cpu        |        0.89869   |             nan |       0.666121  |                   0.0202264 |                   0.000494292 |       1 |
| PEMS03        | current_metagb |        0.660559  |             nan |       0.512538  |                   1.10029   |                   0.0006955   |       1 |
| PEMS03        | xgb_cpu        |        1.36955   |             nan |       1.07803   |                   0.0406812 |                   0.000560416 |       1 |
| PEMS04        | xgb_cpu        |        1.24933   |             nan |       1.06287   |                   0.0243995 |                   0.000637583 |       1 |
| PEMS04        | current_metagb |        1.596     |             nan |       1.29762   |                   8.33252   |                   0.000721667 |       1 |
| PEMS07        | current_metagb |        0.616869  |             nan |       0.565144  |                   1.08684   |                   0.000688041 |       1 |
| PEMS07        | xgb_cpu        |        0.920019  |             nan |       0.858451  |                   0.106563  |                   0.000782083 |       1 |
| PEMS08        | xgb_cpu        |        2.48095   |             nan |       1.9799    |                   0.027015  |                   0.000532501 |       1 |
| PEMS08        | current_metagb |        2.51461   |             nan |       2.06019   |                   7.93268   |                   0.000672041 |       1 |
| Weather       | current_metagb |        0.842487  |             nan |       0.532156  |                   1.99051   |                   0.00065525  |       1 |
| Weather       | xgb_cpu        |        0.960425  |             nan |       0.674244  |                   0.0256267 |                   0.000506125 |       1 |
| electricity   | current_metagb |        0.514191  |             nan |       0.387919  |                   8.10862   |                   0.000720042 |       1 |
| electricity   | xgb_cpu        |        0.522076  |             nan |       0.365909  |                   0.122626  |                   0.000911959 |       1 |
| exchange_rate | xgb_cpu        |        0.764019  |             nan |       0.685839  |                   0.0298555 |                   0.000646625 |       1 |
| exchange_rate | current_metagb |        1.07054   |             nan |       1.03639   |                   2.08644   |                   0.000688415 |       1 |
| solar         | xgb_cpu        |        0.0335198 |             nan |       0.0127326 |                   0.0699013 |                   0.000738458 |       1 |
| solar         | current_metagb |        0.288172  |             nan |       0.154777  |                   1.08141   |                   0.0007      |       1 |
| traffic       | xgb_cpu        |        0.505491  |             nan |       0.401792  |                   0.104288  |                   0.000714125 |       1 |
| traffic       | current_metagb |        1.72749   |             nan |       1.07228   |                   1.09593   |                   0.000714542 |       1 |

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.633613, folds=1)
- ETTh2: xgb_cpu (mean RMSE=0.370351, folds=1)
- ETTm1: xgb_cpu (mean RMSE=0.288076, folds=1)
- ETTm2: xgb_cpu (mean RMSE=0.172995, folds=1)
- ILI: current_metagb (mean RMSE=0.699969, folds=1)
- PEMS03: current_metagb (mean RMSE=0.660559, folds=1)
- PEMS04: xgb_cpu (mean RMSE=1.24933, folds=1)
- PEMS07: current_metagb (mean RMSE=0.616869, folds=1)
- PEMS08: xgb_cpu (mean RMSE=2.48095, folds=1)
- Weather: current_metagb (mean RMSE=0.842487, folds=1)
- electricity: current_metagb (mean RMSE=0.514191, folds=1)
- exchange_rate: xgb_cpu (mean RMSE=0.764019, folds=1)
- solar: xgb_cpu (mean RMSE=0.0335198, folds=1)
- traffic: xgb_cpu (mean RMSE=0.505491, folds=1)

## Split Audit

- split records: 28
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
