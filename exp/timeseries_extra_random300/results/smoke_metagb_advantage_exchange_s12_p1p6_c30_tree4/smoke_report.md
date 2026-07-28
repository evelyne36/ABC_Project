# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_metagb_advantage_exchange_s12_p1p6_c30_tree4`

## Aggregate

      dataset          model  test_rmse_mean  test_rmse_std  test_mae_mean  total_train_time_sec_mean  total_predict_time_sec_mean  folds
exchange_rate current_metagb        0.292137       0.123141       0.244838                   0.651508                     0.001654      2
exchange_rate   catboost_cpu        0.413999       0.097792       0.373364                   0.096612                     0.001865      2
exchange_rate        xgb_cpu        0.446680       0.071996       0.395436                   0.033126                     0.001316      2
exchange_rate       lgbm_cpu        0.592577       0.014665       0.533436                   0.033306                     0.002154      2

## Winner By Dataset

- exchange_rate: current_metagb (mean RMSE=0.292137, folds=2)

## Split Audit

- split records: 8
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
