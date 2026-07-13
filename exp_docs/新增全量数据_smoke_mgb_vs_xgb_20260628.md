# 新增全量数据 Smoke MGB vs XGB 对比

日期：2026-06-28

## 排序检查

- TS 与 tabular 的 `candidate_inventory.csv` 均已改为按压力升序。
- TS 综合脚本显式按 `execution_order_by_pressure.csv` 的顺序传入数据。
- Tabular runner 的任务排序已由行数排序改为 `rows * features^2` 压力排序。

## TS 全量 14 个

- MetaGB 胜出：7
- XGB 胜出：7

| dataset       |   pressure_proxy |   xgb_cpu |   current_metagb | winner         |   relative_gain_vs_xgb |   rows_used |   features_used |
|:--------------|-----------------:|----------:|-----------------:|:---------------|-----------------------:|------------:|----------------:|
| ILI           |         27264384 |  0.669335 |         0.643296 | current_metagb |              0.0389022 |         240 |               7 |
| exchange_rate |        279724032 |  4.22759  |         2.15707  | current_metagb |              0.489765  |         240 |               8 |
| ETTh1         |        491720320 |  0.749937 |         0.98856  | xgb_cpu        |             -0.31819   |         240 |               7 |
| ETTh2         |        491720320 |  1.85733  |         1.44838  | current_metagb |              0.220184  |         240 |               7 |
| ETTm1         |       1966881280 |  0.838664 |         0.908471 | xgb_cpu        |             -0.0832351 |         240 |               7 |
| ETTm2         |       1966881280 |  0.230865 |         0.226249 | current_metagb |              0.019993  |         240 |               7 |
| Weather       |      13384433664 |  1.38965  |         1.10053  | current_metagb |              0.208052  |         240 |              21 |
| PEMS08        |      74290176000 |  2.75737  |         2.85354  | xgb_cpu        |             -0.0348793 |         240 |              32 |
| PEMS04        |     230701687488 |  3.97649  |         4.44158  | xgb_cpu        |             -0.116957  |         240 |              32 |
| PEMS03        |     483607652352 |  1.08871  |         0.995462 | current_metagb |              0.0856489 |         240 |              32 |
| solar         |     568269158400 |  0.235246 |         0.23203  | current_metagb |              0.0136736 |         240 |              32 |
| electricity   |    1562019262464 |  0.404263 |         0.422354 | xgb_cpu        |             -0.0447501 |         240 |              32 |
| PEMS07        |    3169267490304 |  0.750988 |         0.816517 | xgb_cpu        |             -0.0872571 |         240 |              32 |
| traffic       |    7508616470016 |  0.546004 |         0.645502 | xgb_cpu        |             -0.182229  |         240 |              32 |

详细结果：`timeseries_extra_random300/results/smoke_ts_all14_pressure_order_mgb_vs_xgb_20260628/mgb_vs_xgb_comparison.csv`

## Tabular 全量 27 个

- MetaGB 胜出：11
- XGB 胜出：16

| task_type   | dataset                                |   pressure_proxy | metric   |    xgb_cpu |   current_metagb | winner         |   relative_gain_vs_xgb |   rows_used |   features_used |
|:------------|:---------------------------------------|-----------------:|:---------|-----------:|-----------------:|:---------------|-----------------------:|------------:|----------------:|
| binary      | binary__haberman.csv                   |             2754 | auc      |  0.699758  |         0.601291 | xgb_cpu        |            -0.140715   |         160 |               3 |
| multiclass  | multiclass__balance-scale.csv          |            10000 | logloss  |  0.677163  |         0.626619 | current_metagb |             0.0746414  |         160 |               4 |
| binary      | binary__banknote-authentication.csv    |            21952 | auc      |  0.986032  |         0.993016 | current_metagb |             0.00708307 |         160 |               4 |
| regression  | regression__Moneyball.csv              |            22344 | rmse     | 30.781     |       277.371    | xgb_cpu        |            -8.01112    |         114 |              14 |
| regression  | regression__energy_efficiency.csv      |            49152 | rmse     |  4.41117   |         9.68084  | xgb_cpu        |            -1.19462    |         160 |               8 |
| binary      | binary__breast-w.csv                   |            55323 | auc      |  0.958104  |         0.98386  | current_metagb |             0.0268817  |         160 |               9 |
| binary      | binary__ilpd.csv                       |            58300 | auc      |  0.625095  |         0.643783 | current_metagb |             0.0298963  |         160 |              10 |
| regression  | regression__boston.csv                 |            85514 | rmse     |  5.6058    |         9.96115  | xgb_cpu        |            -0.776936   |         160 |              13 |
| regression  | regression__space_ga.csv               |           111852 | rmse     |  0.14613   |         0.262747 | xgb_cpu        |            -0.79803    |         160 |               6 |
| multiclass  | multiclass__vowel.csv                  |           119790 | logloss  |  1.73985   |         1.78215  | xgb_cpu        |            -0.0243151  |         160 |              11 |
| binary      | binary__phoneme.csv                    |           135100 | auc      |  0.873884  |         0.880208 | current_metagb |             0.00723712 |         160 |               5 |
| binary      | binary__credit-approval.csv            |           146925 | auc      |  0.954431  |         0.915462 | xgb_cpu        |            -0.0408298  |         160 |              15 |
| regression  | regression__friedman1.csv              |           200000 | rmse     |  3.62758   |         6.48569  | xgb_cpu        |            -0.787882   |         160 |              10 |
| regression  | regression__abalone.csv                |           267328 | rmse     |  3.10996   |         5.09329  | xgb_cpu        |            -0.637738   |         160 |               8 |
| multiclass  | multiclass__vehicle.csv                |           274104 | logloss  |  0.993235  |         0.96231  | current_metagb |             0.0311358  |         160 |              18 |
| binary      | binary__mozilla4.csv                   |           388625 | auc      |  0.92348   |         0.915444 | xgb_cpu        |            -0.00870223 |         160 |               5 |
| binary      | binary__pc1.csv                        |           489069 | auc      |  0.71509   |         0.837838 | current_metagb |             0.171654   |         160 |              21 |
| regression  | regression__kin8nm.csv                 |           524288 | rmse     |  0.259161  |         0.41451  | xgb_cpu        |            -0.599431   |         160 |               8 |
| multiclass  | multiclass__segment.csv                |           833910 | logloss  |  0.799548  |         0.791798 | current_metagb |             0.009692   |         160 |              19 |
| binary      | binary__kc1.csv                        |           930069 | auc      |  0.811137  |         0.707807 | xgb_cpu        |            -0.127389   |         160 |              21 |
| binary      | binary__electricity.csv                |          1885226 | auc      |  0.78125   |         0.802812 | current_metagb |             0.0276     |         160 |               7 |
| regression  | regression__naval_propulsion_plant.csv |          2339064 | rmse     |  0.0127005 |         0.338104 | xgb_cpu        |           -25.6213     |         160 |              14 |
| regression  | regression__cpu_activity.csv           |          3612672 | rmse     | 13.7885    |        33.5871   | xgb_cpu        |            -1.43589    |         160 |              21 |
| multiclass  | multiclass__mfeat-karhunen.csv         |          8192000 | logloss  |  1.56579   |         1.62813  | xgb_cpu        |            -0.0398092  |         160 |              64 |
| multiclass  | multiclass__satimage.csv               |          8333280 | logloss  |  0.993781  |         1.04558  | xgb_cpu        |            -0.0521202  |         160 |              36 |
| multiclass  | multiclass__mfeat-fourier.csv          |         11552000 | logloss  |  1.49493   |         1.46304  | current_metagb |             0.02133    |         160 |              76 |
| multiclass  | multiclass__optdigits.csv              |         23019520 | logloss  |  1.50195   |         1.45353  | current_metagb |             0.0322386  |         160 |              64 |

详细结果：`tabular_extra_random300/results/smoke_tabular_all27_mgb_vs_xgb_tiny_20260628/mgb_vs_xgb_comparison.csv`

## 注意

- 这些是轻量 smoke，用于提前看方向，不是 full nested-CV 结论。
- TS smoke: `seq_len=4, pred_len=1, max_raw_rows=240, max_features=32, max_candidates=30`。
- Tabular smoke: `max_rows_per_dataset=160, max_candidates=30, tiny_smoke_search_space.json`。