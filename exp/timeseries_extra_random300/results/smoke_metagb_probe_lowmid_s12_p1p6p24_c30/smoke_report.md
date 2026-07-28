# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_metagb_probe_lowmid_s12_p1p6p24_c30`

## Aggregate

      dataset          model  test_rmse_mean  test_rmse_std  test_mae_mean  total_train_time_sec_mean  total_predict_time_sec_mean  folds
        ETTh1        xgb_cpu        0.808319       0.114930       0.633670                   0.200000                     0.004997      3
        ETTh1 current_metagb        0.817949       0.105341       0.638433                   1.637352                     0.004232      3
          ILI current_metagb        1.157758       0.376297       1.026609                   2.632035                     0.004303      3
          ILI        xgb_cpu        1.179727       0.350202       1.030903                   0.232647                     0.005242      3
exchange_rate current_metagb        0.366803       0.155907       0.304199                   1.926095                     0.003954      3
exchange_rate        xgb_cpu        0.446775       0.050909       0.386816                   0.124145                     0.006796      3

## Winner By Dataset

- ETTh1: xgb_cpu (mean RMSE=0.808319, folds=3)
- ILI: current_metagb (mean RMSE=1.15776, folds=3)
- exchange_rate: current_metagb (mean RMSE=0.366803, folds=3)

## Split Audit

- split records: 18
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
