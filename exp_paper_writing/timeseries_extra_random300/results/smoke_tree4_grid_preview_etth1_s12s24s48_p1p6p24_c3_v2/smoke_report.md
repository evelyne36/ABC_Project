# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_tree4_grid_preview_etth1_s12s24s48_p1p6p24_c3_v2`

## Aggregate

dataset          model  test_rmse_mean  test_rmse_std  test_mae_mean  total_train_time_sec_mean  total_predict_time_sec_mean  folds
  ETTh1        xgb_cpu        0.313023       0.086539       0.254015                   0.105032                     0.003330      9
  ETTh1       lgbm_cpu        0.325729       0.088140       0.267218                   0.114997                     0.006525      9
  ETTh1   catboost_cpu        0.331473       0.085846       0.263367                   0.317143                     0.006516      9
  ETTh1 current_metagb        0.425833       0.037249       0.359632                   1.673836                     0.003547      9

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.313023, folds=9)

## Split Audit

- split records: 36
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
