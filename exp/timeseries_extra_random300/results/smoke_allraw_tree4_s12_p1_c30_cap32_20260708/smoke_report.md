# Time-Series Smoke Report

results_dir: `/Users/ancd/VSCodeProject/check_codereview0701/timeseries_extra_random300/results/smoke_allraw_tree4_s12_p1_c30_cap32_20260708`

## Aggregate

      dataset          model  test_rmse_mean  test_rmse_std  test_mae_mean  total_train_time_sec_mean  total_predict_time_sec_mean  folds
        ETTh1       lgbm_cpu        0.645671            NaN       0.531620                   0.014541                     0.001072      1
        ETTh1        xgb_cpu        0.681352            NaN       0.554692                   0.013197                     0.000556      1
        ETTh1 current_metagb        0.701795            NaN       0.577229                   0.164936                     0.000640      1
        ETTh1   catboost_cpu        0.826457            NaN       0.676850                   0.028281                     0.000914      1
        ETTh2        xgb_cpu        0.351029            NaN       0.282807                   0.014471                     0.001346      1
        ETTh2 current_metagb        0.358114            NaN       0.295853                   0.174041                     0.000716      1
        ETTh2       lgbm_cpu        0.360263            NaN       0.294759                   0.010899                     0.001597      1
        ETTh2   catboost_cpu        0.428503            NaN       0.370535                   0.027686                     0.000668      1
        ETTm1 current_metagb        0.540957            NaN       0.456304                   0.169674                     0.000754      1
        ETTm1       lgbm_cpu        0.590513            NaN       0.484555                   0.011304                     0.000879      1
        ETTm1        xgb_cpu        0.599962            NaN       0.498766                   0.011524                     0.000632      1
        ETTm1   catboost_cpu        0.701176            NaN       0.576881                   0.031165                     0.000808      1
        ETTm2 current_metagb        0.652745            NaN       0.460260                   0.171608                     0.000687      1
        ETTm2       lgbm_cpu        1.176152            NaN       0.769406                   0.014567                     0.001144      1
        ETTm2        xgb_cpu        1.176620            NaN       0.772554                   0.015329                     0.000695      1
        ETTm2   catboost_cpu        1.357247            NaN       0.926044                   0.022364                     0.001161      1
          ILI       lgbm_cpu        0.853661            NaN       0.679524                   0.019764                     0.001463      1
          ILI        xgb_cpu        0.853883            NaN       0.676590                   0.015764                     0.000602      1
          ILI current_metagb        0.868455            NaN       0.735249                   1.207621                     0.000898      1
          ILI   catboost_cpu        1.229534            NaN       1.132043                   0.039420                     0.000827      1
       PEMS03 current_metagb        0.485015            NaN       0.464574                   4.404984                     0.001281      1
       PEMS03       lgbm_cpu        0.552962            NaN       0.538225                   0.084825                     0.002571      1
       PEMS03        xgb_cpu        0.555062            NaN       0.538303                   0.052878                     0.001108      1
       PEMS03   catboost_cpu        0.612642            NaN       0.594335                   0.154757                     0.001242      1
       PEMS04        xgb_cpu        0.581180            NaN       0.499129                   0.036781                     0.000721      1
       PEMS04       lgbm_cpu        0.583043            NaN       0.500167                   0.053079                     0.002078      1
       PEMS04 current_metagb        0.597301            NaN       0.515495                   3.591230                     0.000853      1
       PEMS04   catboost_cpu        0.603650            NaN       0.523875                   0.090389                     0.001146      1
       PEMS07 current_metagb        0.566921            NaN       0.536481                   3.205300                     0.000740      1
       PEMS07       lgbm_cpu        0.676043            NaN       0.648188                   0.047854                     0.001486      1
       PEMS07        xgb_cpu        0.676571            NaN       0.639955                   0.050132                     0.000645      1
       PEMS07   catboost_cpu        1.099972            NaN       1.056629                   0.067241                     0.000850      1
       PEMS08        xgb_cpu        0.229279            NaN       0.181938                   0.039268                     0.000747      1
       PEMS08 current_metagb        0.254232            NaN       0.203190                   3.836127                     0.000965      1
       PEMS08   catboost_cpu        0.268135            NaN       0.209602                   0.095831                     0.001233      1
       PEMS08       lgbm_cpu        0.510115            NaN       0.228164                   0.044762                     0.001614      1
      Weather        xgb_cpu        0.325038            NaN       0.275573                   0.034665                     0.000998      1
      Weather       lgbm_cpu        0.329816            NaN       0.280053                   0.049931                     0.002626      1
      Weather current_metagb        0.475795            NaN       0.382623                  11.349932                     0.000923      1
      Weather   catboost_cpu        0.788114            NaN       0.729594                   0.044598                     0.001139      1
exchange_rate current_metagb        0.205064            NaN       0.172841                   0.211850                     0.000752      1
exchange_rate   catboost_cpu        0.344850            NaN       0.311617                   0.021461                     0.001126      1
exchange_rate        xgb_cpu        0.395772            NaN       0.356453                   0.010105                     0.000535      1
exchange_rate       lgbm_cpu        0.582208            NaN       0.534904                   0.011003                     0.001295      1

## Winner By Dataset

- ETTh1: lgbm_cpu (mean RMSE=0.645671, folds=1)
- ETTh2: xgb_cpu (mean RMSE=0.351029, folds=1)
- ETTm1: current_metagb (mean RMSE=0.540957, folds=1)
- ETTm2: current_metagb (mean RMSE=0.652745, folds=1)
- ILI: lgbm_cpu (mean RMSE=0.853661, folds=1)
- PEMS03: current_metagb (mean RMSE=0.485015, folds=1)
- PEMS04: xgb_cpu (mean RMSE=0.58118, folds=1)
- PEMS07: current_metagb (mean RMSE=0.566921, folds=1)
- PEMS08: xgb_cpu (mean RMSE=0.229279, folds=1)
- Weather: xgb_cpu (mean RMSE=0.325038, folds=1)
- exchange_rate: current_metagb (mean RMSE=0.205064, folds=1)

## Split Audit

- split records: 44
- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.
