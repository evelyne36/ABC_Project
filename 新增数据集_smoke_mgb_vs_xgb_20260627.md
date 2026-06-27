# 新增 TS 与 Tabular 全数据集 Smoke 对比

日期：2026-06-27

## 口径

- 目的：快速预览 `current_metagb` vs `xgb_cpu`，不是 final full nested-CV 结论。
- TS：`outer_folds=2`、`inner_folds=2`、`outer_fold_index=0`、`max_candidates=30`、`seq_len=4`、`pred_len=1`、`max_raw_rows=240`、`max_features=32`。
- Tabular：`outer_folds=2`、`inner_folds=2`、`outer_fold_index=0`、`max_candidates=30`、`max_rows_per_dataset=240`，并使用 `tiny_smoke_search_space.json` 的小树配置。
- `relative_gain_vs_xgb > 0` 表示 MetaGB 优于 XGB；分类 AUC 越高越好，logloss/RMSE 越低越好。

## Time-Series 汇总

- 数据集数：14
- MetaGB 胜出：7
- XGB 胜出：7

| dataset       |   xgb_cpu |   current_metagb | winner         |   relative_gain_vs_xgb |   rows_used |   features_used |
|:--------------|----------:|-----------------:|:---------------|-----------------------:|------------:|----------------:|
| exchange_rate |  4.22759  |         2.15707  | current_metagb |              0.489765  |         240 |               8 |
| ETTh2         |  1.85733  |         1.44838  | current_metagb |              0.220184  |         240 |               7 |
| Weather       |  1.38965  |         1.10053  | current_metagb |              0.208052  |         240 |              21 |
| PEMS03        |  1.08871  |         0.995462 | current_metagb |              0.0856489 |         240 |              32 |
| ILI           |  0.669335 |         0.643296 | current_metagb |              0.0389022 |         240 |               7 |
| ETTm2         |  0.230865 |         0.226249 | current_metagb |              0.019993  |         240 |               7 |
| solar         |  0.235246 |         0.23203  | current_metagb |              0.0136736 |         240 |              32 |
| PEMS08        |  2.75737  |         2.85354  | xgb_cpu        |             -0.0348793 |         240 |              32 |
| electricity   |  0.404263 |         0.422354 | xgb_cpu        |             -0.0447501 |         240 |              32 |
| ETTm1         |  0.838664 |         0.908471 | xgb_cpu        |             -0.0832351 |         240 |               7 |
| PEMS07        |  0.750988 |         0.816517 | xgb_cpu        |             -0.0872571 |         240 |              32 |
| PEMS04        |  3.97649  |         4.44158  | xgb_cpu        |             -0.116957  |         240 |              32 |
| traffic       |  0.546004 |         0.645502 | xgb_cpu        |             -0.182229  |         240 |              32 |
| ETTh1         |  0.749937 |         0.98856  | xgb_cpu        |             -0.31819   |         240 |               7 |

详细结果：`timeseries_extra_random300/results/smoke_ts_all_datasets_mgb_vs_xgb_20260627/mgb_vs_xgb_comparison.csv`

## Tabular 汇总

- 数据集数：15
- MetaGB 胜出：7
- XGB 胜出：8

| task_type   | dataset                                | metric   |   xgb_cpu |   current_metagb | winner         |   relative_gain_vs_xgb |   rows_used |   features_used |
|:------------|:---------------------------------------|:---------|----------:|-----------------:|:---------------|-----------------------:|------------:|----------------:|
| binary      | binary__pc1.csv                        | auc      | 0.756256  |         0.798799 | current_metagb |            0.0562541   |         240 |              21 |
| binary      | binary__phoneme.csv                    | auc      | 0.825714  |         0.864538 | current_metagb |            0.0470181   |         240 |               5 |
| binary      | binary__banknote-authentication.csv    | auc      | 0.979442  |         0.994931 | current_metagb |            0.0158137   |         240 |               4 |
| binary      | binary__electricity.csv                | auc      | 0.824722  |         0.805    | xgb_cpu        |           -0.0239138   |         240 |               7 |
| binary      | binary__credit-approval.csv            | auc      | 0.932308  |         0.892308 | xgb_cpu        |           -0.0429043   |         240 |              15 |
| binary      | binary__kc1.csv                        | auc      | 0.69359   |         0.654247 | xgb_cpu        |           -0.0567243   |         240 |              21 |
| multiclass  | multiclass__segment.csv                | logloss  | 0.913641  |         0.824799 | current_metagb |            0.0972396   |         240 |              19 |
| multiclass  | multiclass__balance-scale.csv          | logloss  | 0.665266  |         0.645589 | current_metagb |            0.0295768   |         240 |               4 |
| multiclass  | multiclass__vowel.csv                  | logloss  | 1.74012   |         1.69288  | current_metagb |            0.0271475   |         240 |              11 |
| multiclass  | multiclass__vehicle.csv                | logloss  | 0.969823  |         0.969212 | current_metagb |            0.000630105 |         240 |              18 |
| multiclass  | multiclass__satimage.csv               | logloss  | 0.878594  |         0.937737 | xgb_cpu        |           -0.0673161   |         240 |              36 |
| regression  | regression__kin8nm.csv                 | rmse     | 0.222542  |         0.331476 | xgb_cpu        |           -0.489497    |         240 |               8 |
| regression  | regression__space_ga.csv               | rmse     | 0.131039  |         0.24307  | xgb_cpu        |           -0.854942    |         240 |               6 |
| regression  | regression__energy_efficiency.csv      | rmse     | 4.18467   |         9.03329  | xgb_cpu        |           -1.15866     |         240 |               8 |
| regression  | regression__naval_propulsion_plant.csv | rmse     | 0.0117766 |         0.342521 | xgb_cpu        |          -28.085       |         240 |              14 |

详细结果：`tabular_extra_random300/results/smoke_tabular_all_datasets_mgb_vs_xgb_tiny_20260627/mgb_vs_xgb_comparison.csv`