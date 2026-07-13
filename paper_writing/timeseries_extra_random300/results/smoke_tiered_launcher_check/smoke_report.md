# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_tiered_launcher_check`

## Aggregate

      dataset          model  test_rmse_mean  test_rmse_std  test_mae_mean  total_train_time_sec_mean  total_predict_time_sec_mean  folds
        ETTh1   catboost_cpu        0.888641            NaN       0.727021                   0.020452                     0.001231      1
        ETTh1       lgbm_cpu        0.890977            NaN       0.758598                   0.008307                     0.001340      1
        ETTh1        xgb_cpu        1.011303            NaN       0.819631                   0.011091                     0.000855      1
        ETTh1 current_metagb        1.377601            NaN       1.138368                   0.416056                     0.001121      1
        ETTh2 current_metagb        1.377810            NaN       0.945400                   0.448958                     0.001545      1
        ETTh2        xgb_cpu        1.380513            NaN       0.993157                   0.012234                     0.000902      1
        ETTh2       lgbm_cpu        1.540062            NaN       1.077416                   0.008142                     0.001093      1
        ETTh2   catboost_cpu        1.765766            NaN       1.248997                   0.013735                     0.000997      1
        ETTm1        xgb_cpu        0.388553            NaN       0.360957                   0.009339                     0.000806      1
        ETTm1       lgbm_cpu        0.392859            NaN       0.364042                   0.008431                     0.000948      1
        ETTm1   catboost_cpu        0.410024            NaN       0.384424                   0.018161                     0.000954      1
        ETTm1 current_metagb        0.502069            NaN       0.479737                   0.200734                     0.001047      1
        ETTm2        xgb_cpu        0.401597            NaN       0.395916                   0.008128                     0.000738      1
        ETTm2       lgbm_cpu        0.401597            NaN       0.395916                   0.008575                     0.001160      1
        ETTm2   catboost_cpu        0.497374            NaN       0.489824                   0.034096                     0.001348      1
        ETTm2 current_metagb        0.594959            NaN       0.568279                   0.959215                     0.001052      1
          ILI current_metagb        0.230684            NaN       0.209650                   0.194295                     0.000982      1
          ILI       lgbm_cpu        0.242884            NaN       0.208105                   0.009444                     0.001031      1
          ILI   catboost_cpu        0.268351            NaN       0.250022                   0.022887                     0.001217      1
          ILI        xgb_cpu        0.288544            NaN       0.251704                   0.009519                     0.000821      1
exchange_rate       lgbm_cpu        0.213793            NaN       0.174830                   0.009139                     0.001482      1
exchange_rate   catboost_cpu        0.216964            NaN       0.151758                   0.033797                     0.000802      1
exchange_rate        xgb_cpu        0.251879            NaN       0.201173                   0.009357                     0.000696      1
exchange_rate current_metagb        0.334023            NaN       0.230460                   1.396400                     0.001073      1

## Winner By Dataset

- ETTh1: catboost_cpu (mean RMSE=0.888641, folds=1)
- ETTh2: current_metagb (mean RMSE=1.37781, folds=1)
- ETTm1: xgb_cpu (mean RMSE=0.388553, folds=1)
- ETTm2: xgb_cpu (mean RMSE=0.401597, folds=1)
- ILI: current_metagb (mean RMSE=0.230684, folds=1)
- exchange_rate: lgbm_cpu (mean RMSE=0.213793, folds=1)
