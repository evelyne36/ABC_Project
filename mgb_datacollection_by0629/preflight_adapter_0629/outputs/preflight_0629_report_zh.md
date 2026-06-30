# 0629 Preflight Score vs Outcomes 审计报告

## 核心结论

- 可用记录数：142。
- score vs MGB win ROC-AUC：0.7492
- score vs MGB win Average Precision：0.8003
- score 与 test_improvement 的 Spearman 相关：0.4462

本版使用 family-calibrated adapter score：synthetic 仍采用机制适配分，TabArena/otherdata 额外压低真实数据中的 decoupling 假阳性，并使用更保守的高分阈值。

- synthetic：高适配组 18 个，MGB胜 18 个，XGB胜 0 个，优势占比 100.0% (18/18)，平均 test_improvement 0.3723。
- TabArena：高适配组 5 个，MGB胜 5 个，XGB胜 0 个，优势占比 100.0% (5/5)，平均 test_improvement 0.0038。
- tabular otherdata：高适配组 3 个，MGB胜 3 个，XGB胜 0 个，优势占比 100.0% (3/3)，平均 test_improvement 0.0207。

## 数据覆盖

当前adapter审计以 `dataset_level_mgb_vs_xgb.csv` 中已经形成MGB/XGB配对结果的dataset-level记录为单位；TabArena包含主实验与tabarena30后续实验。

| data_group        |   total_dataset_records | complete_mgb_xgb_results   | complete_result_rate   | preflight_profile_ok   |
|:------------------|------------------------:|:---------------------------|:-----------------------|:-----------------------|
| synthetic         |                      35 | 35/35                      | 100.0% (35/35)         | 35/35                  |
| tabarena          |                      78 | 78/78                      | 100.0% (78/78)         | 78/78                  |
| tabular_otherdata |                      29 | 29/29                      | 100.0% (29/29)         | 29/29                  |

## 结果概述（含数据集数）

- 高适配组 HIGH_MGB_FIT：26 个数据集，MGB/Tie/XGB = 26/0/0，MGB优势占比 100.0% (26/26)，MGB非劣占比 100.0% (26/26)，平均 test_improvement 0.2608。
- 中适配组 MEDIUM_MGB_FIT：40 个数据集，MGB/Tie/XGB = 22/3/15，MGB优势占比 59.5% (22/37)，MGB非劣占比 62.5% (25/40)，平均 test_improvement 0.0542。
- 低适配组 LOW_MGB_FIT：76 个数据集，MGB/Tie/XGB = 26/4/46，MGB优势占比 36.1% (26/72)，MGB非劣占比 39.5% (30/76)，平均 test_improvement -0.0184。

分数据来源来看，高适配组在当前0629回测样本中的MGB优势占比均达到100%，但覆盖面不同：synthetic覆盖18个强机制数据，TabArena覆盖5个真实数据，tabular otherdata覆盖3个数据。中低分组覆盖更多数据，但MGB优势占比下降，说明该adapter更适合用作事前分层和实验优先级排序，而不是替代完整评测。

### 分数据来源结果表

| data_group        | mechanism_band   |   datasets | mgb/tie/xgb   | mgb_advantage_share   | mgb_nonloss_share   |   mean_improvement |   median_score |   mean_runtime_ratio |
|:------------------|:-----------------|-----------:|:--------------|:----------------------|:--------------------|-------------------:|---------------:|---------------------:|
| synthetic         | HIGH_MGB_FIT     |         18 | 18/0/0        | 100.0% (18/18)        | 100.0% (18/18)      |         0.372276   |      0.848386  |              22.0205 |
| synthetic         | MEDIUM_MGB_FIT   |         13 | 10/1/2        | 83.3% (10/12)         | 84.6% (11/13)       |         0.133456   |      0.4       |              21.4162 |
| synthetic         | LOW_MGB_FIT      |          4 | 2/0/2         | 50.0% (2/4)           | 50.0% (2/4)         |        -0.0014358  |      0.0622576 |              16.1977 |
| tabarena          | HIGH_MGB_FIT     |          5 | 5/0/0         | 100.0% (5/5)          | 100.0% (5/5)        |         0.00377186 |      0.573533  |              28.4505 |
| tabarena          | MEDIUM_MGB_FIT   |         17 | 9/0/8         | 52.9% (9/17)          | 52.9% (9/17)        |         0.00978038 |      0.438995  |             111.309  |
| tabarena          | LOW_MGB_FIT      |         56 | 20/2/34       | 37.0% (20/54)         | 39.3% (22/56)       |        -0.00718912 |      0.103072  |              37.6191 |
| tabular_otherdata | HIGH_MGB_FIT     |          3 | 3/0/0         | 100.0% (3/3)          | 100.0% (3/3)        |         0.020732   |      0.535373  |             105.999  |
| tabular_otherdata | MEDIUM_MGB_FIT   |         10 | 3/2/5         | 37.5% (3/8)           | 50.0% (5/10)        |         0.0268579  |      0.39286   |              28.1147 |
| tabular_otherdata | LOW_MGB_FIT      |         16 | 4/2/10        | 28.6% (4/14)          | 37.5% (6/16)        |        -0.0620808  |      0.216551  |              19.948  |

## 阈值解释与使用边界

- synthetic 使用 `0.00-0.35 / 0.35-0.60 / 0.60-1.00`；TabArena 与 tabular otherdata 使用更保守的 `0.00-0.30 / 0.30-0.50 / 0.50-1.00`。
- `LOW_MGB_FIT`：机制证据较弱，可先保留XGB基线或只做小预算MGB smoke。本批低分组 76 个数据集中，XGB胜出 46 个，MGB胜出 26 个，平均 test_improvement 为 -0.0184。
- `MEDIUM_MGB_FIT`：可作为轻量验证区间，用少量候选或较短时间预算检查MGB是否有增益。本批中分组 40 个数据集中，MGB胜出 22 个，平均 test_improvement 为 0.0542。
- `HIGH_MGB_FIT`：机制上更符合MGB，可作为高优先级候选；真实数据高分阈值提高到0.50后，假阳性明显减少。本批高分组 26 个数据集中，MGB胜出 26 个，平均 test_improvement 为 0.2608。

解释边界：这些阈值来自当前0629结果的回测校准，可用于当前结果解释和下一轮实验优先级排序；在新数据上仍应视作事前筛选，而不是已泛化的胜负分类器。

## Band Summary

| mechanism_band   |   datasets | mgb/tie/xgb   | mgb_advantage_share   | mgb_nonloss_share   |   mean_improvement |   median_improvement |   mean_runtime_ratio |
|:-----------------|-----------:|:--------------|:----------------------|:--------------------|-------------------:|---------------------:|---------------------:|
| HIGH_MGB_FIT     |         26 | 26/0/0        | 100.0% (26/26)        | 100.0% (26/26)      |          0.260847  |            0.200004  |              32.9469 |
| MEDIUM_MGB_FIT   |         40 | 22/3/15       | 59.5% (22/37)         | 62.5% (25/40)       |          0.0542442 |            0.001773  |              61.2951 |
| LOW_MGB_FIT      |         76 | 26/4/46       | 36.1% (26/72)         | 39.5% (30/76)       |         -0.0184424 |           -0.0009999 |              32.7714 |

## Data Group/Band Summary

| data_group        | mechanism_band   |   datasets | mgb/tie/xgb   | mgb_advantage_share   | mgb_nonloss_share   |   mean_improvement |   median_score |   mean_runtime_ratio |
|:------------------|:-----------------|-----------:|:--------------|:----------------------|:--------------------|-------------------:|---------------:|---------------------:|
| synthetic         | HIGH_MGB_FIT     |         18 | 18/0/0        | 100.0% (18/18)        | 100.0% (18/18)      |         0.372276   |      0.848386  |              22.0205 |
| synthetic         | MEDIUM_MGB_FIT   |         13 | 10/1/2        | 83.3% (10/12)         | 84.6% (11/13)       |         0.133456   |      0.4       |              21.4162 |
| synthetic         | LOW_MGB_FIT      |          4 | 2/0/2         | 50.0% (2/4)           | 50.0% (2/4)         |        -0.0014358  |      0.0622576 |              16.1977 |
| tabarena          | HIGH_MGB_FIT     |          5 | 5/0/0         | 100.0% (5/5)          | 100.0% (5/5)        |         0.00377186 |      0.573533  |              28.4505 |
| tabarena          | MEDIUM_MGB_FIT   |         17 | 9/0/8         | 52.9% (9/17)          | 52.9% (9/17)        |         0.00978038 |      0.438995  |             111.309  |
| tabarena          | LOW_MGB_FIT      |         56 | 20/2/34       | 37.0% (20/54)         | 39.3% (22/56)       |        -0.00718912 |      0.103072  |              37.6191 |
| tabular_otherdata | HIGH_MGB_FIT     |          3 | 3/0/0         | 100.0% (3/3)          | 100.0% (3/3)        |         0.020732   |      0.535373  |             105.999  |
| tabular_otherdata | MEDIUM_MGB_FIT   |         10 | 3/2/5         | 37.5% (3/8)           | 50.0% (5/10)        |         0.0268579  |      0.39286   |              28.1147 |
| tabular_otherdata | LOW_MGB_FIT      |         16 | 4/2/10        | 28.6% (4/14)          | 37.5% (6/16)        |        -0.0620808  |      0.216551  |              19.948  |

## Data Group Threshold Diagnostic

| data_group        |   threshold | try_rate      | mgb_advantage_share   | retained_mgb_win_rate   | rejected_xgb_win_rate   |   mean_improvement_if_try |   mean_improvement_if_skip |
|:------------------|------------:|:--------------|:----------------------|:------------------------|:------------------------|--------------------------:|---------------------------:|
| synthetic         |        0.35 | 88.6% (31/35) | 93.3% (28/30)         | 93.3% (28/30)           | 50.0% (2/4)             |                0.272126   |               -0.0014358   |
| synthetic         |        0.45 | 62.9% (22/35) | 95.5% (21/22)         | 70.0% (21/30)           | 75.0% (3/4)             |                0.312812   |                0.119098    |
| synthetic         |        0.6  | 51.4% (18/35) | 100.0% (18/18)        | 60.0% (18/30)           | 100.0% (4/4)            |                0.372276   |                0.101716    |
| synthetic         |        0.7  | 45.7% (16/35) | 100.0% (16/16)        | 53.3% (16/30)           | 100.0% (4/4)            |                0.371297   |                0.131021    |
| tabarena          |        0.25 | 35.9% (28/78) | 53.6% (15/28)         | 44.1% (15/34)           | 69.0% (29/42)           |               -0.0091571  |                0.000778675 |
| tabarena          |        0.3  | 28.2% (22/78) | 63.6% (14/22)         | 41.2% (14/34)           | 81.0% (34/42)           |                0.00841481 |               -0.00718912  |
| tabarena          |        0.35 | 23.1% (18/78) | 61.1% (11/18)         | 32.4% (11/34)           | 83.3% (35/42)           |               -0.00753828 |               -0.00136293  |
| tabarena          |        0.5  | 6.4% (5/78)   | 100.0% (5/5)          | 14.7% (5/34)            | 100.0% (42/42)          |                0.00377186 |               -0.00323732  |
| tabarena          |        0.6  | 2.6% (2/78)   | 100.0% (2/2)          | 5.9% (2/34)             | 100.0% (42/42)          |                0.00460176 |               -0.00298248  |
| tabular_otherdata |        0.25 | 58.6% (17/29) | 46.7% (7/15)          | 70.0% (7/10)            | 46.7% (7/15)            |               -0.0346824  |               -0.00607636  |
| tabular_otherdata |        0.3  | 44.8% (13/29) | 54.5% (6/11)          | 60.0% (6/10)            | 66.7% (10/15)           |                0.0254442  |               -0.0620808   |
| tabular_otherdata |        0.35 | 34.5% (10/29) | 50.0% (5/10)          | 50.0% (5/10)            | 66.7% (10/15)           |                0.0312999  |               -0.0513429   |
| tabular_otherdata |        0.45 | 13.8% (4/29)  | 75.0% (3/4)           | 30.0% (3/10)            | 93.3% (14/15)           |                0.0146029  |               -0.0288371   |
| tabular_otherdata |        0.5  | 10.3% (3/29)  | 100.0% (3/3)          | 30.0% (3/10)            | 100.0% (15/15)          |                0.020732   |               -0.0278736   |

## Threshold Diagnostic

|   threshold | try_rate       | mgb_advantage_share   | retained_mgb_win_rate   | rejected_xgb_win_rate   |   mean_improvement_if_try |   mean_improvement_if_skip |
|------------:|:---------------|:----------------------|:------------------------|:------------------------|--------------------------:|---------------------------:|
|        0.25 | 53.5% (76/142) | 68.5% (50/73)         | 67.6% (50/74)           | 62.3% (38/61)           |                  0.099867 |               -0.000601905 |
|        0.35 | 41.5% (59/142) | 75.9% (44/58)         | 59.5% (44/74)           | 77.0% (47/61)           |                  0.145986 |               -0.0128077   |
|        0.45 | 27.5% (39/142) | 79.5% (31/39)         | 41.9% (31/74)           | 86.9% (53/61)           |                  0.174057 |                0.00739743  |
|        0.6  | 14.8% (21/142) | 100.0% (21/21)        | 28.4% (21/74)           | 100.0% (61/61)          |                  0.319961 |                0.0068677   |

## Family/Band Summary

| family                 | mechanism_band   |   count |   mgb_win |   xgb_win |   mean_improvement |   median_score |
|:-----------------------|:-----------------|--------:|----------:|----------:|-------------------:|---------------:|
| synthetic_0629         | HIGH_MGB_FIT     |      18 |        18 |         0 |         0.372276   |      0.848386  |
| synthetic_0629         | LOW_MGB_FIT      |       4 |         2 |         2 |        -0.0014358  |      0.0622576 |
| synthetic_0629         | MEDIUM_MGB_FIT   |      13 |        10 |         2 |         0.133456   |      0.4       |
| tabarena30_0629        | HIGH_MGB_FIT     |       1 |         1 |         0 |         0.0021978  |      0.573533  |
| tabarena30_0629        | LOW_MGB_FIT      |      15 |         5 |         9 |         0.00237891 |      0.113379  |
| tabarena30_0629        | MEDIUM_MGB_FIT   |       4 |         3 |         1 |         0.0808298  |      0.387356  |
| tabarena_main_0624     | HIGH_MGB_FIT     |       4 |         4 |         0 |         0.00416538 |      0.630095  |
| tabarena_main_0624     | LOW_MGB_FIT      |      41 |        15 |        25 |        -0.0106896  |      0.0978075 |
| tabarena_main_0624     | MEDIUM_MGB_FIT   |      13 |         6 |         7 |        -0.012081   |      0.451162  |
| tabular_otherdata_0629 | HIGH_MGB_FIT     |       3 |         3 |         0 |         0.020732   |      0.535373  |
| tabular_otherdata_0629 | LOW_MGB_FIT      |      16 |         4 |        10 |        -0.0620808  |      0.216551  |
| tabular_otherdata_0629 | MEDIUM_MGB_FIT   |      10 |         3 |         5 |         0.0268579  |      0.39286   |

## Strong Positive Cases

| dataset_key                                                                                           |   mechanism_alignment_score | mechanism_band   | observed_winner   |   test_improvement |   runtime_ratio_mean |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_signal_support |
|:------------------------------------------------------------------------------------------------------|----------------------------:|:-----------------|:------------------|-------------------:|---------------------:|----------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|noise_regime_slope__dim10         |                    1        | HIGH_MGB_FIT     | mgb_win           |          0.0986192 |              11.8034 |                          1        |                        1         |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|regime_slope__dim10               |                    0.97938  | HIGH_MGB_FIT     | mgb_win           |          0.301389  |              12.3134 |                          1        |                        1         |                               1 |                   0.793802 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|multiplicative__dim20             |                    0.871429 | HIGH_MGB_FIT     | mgb_win           |          0.404238  |              20.1889 |                          1        |                        0.142857  |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|sparse_multiplicative__dim20      |                    0.868178 | HIGH_MGB_FIT     | mgb_win           |          0.552904  |              20.189  |                          1        |                        0.142857  |                               1 |                   0.967499 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|mixed_gate_regime__dim10          |                    0.8625   | HIGH_MGB_FIT     | mgb_win           |          0.363345  |              15.334  |                          1        |                        0.0833333 |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|sparse_regime_slope__dim20        |                    0.85     | HIGH_MGB_FIT     | mgb_win           |          0.470424  |              20.7843 |                          1        |                        0         |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|sparse_regime_slope__dim10        |                    0.85     | HIGH_MGB_FIT     | mgb_win           |          0.478562  |              12.718  |                          1        |                        0         |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|sparse_multiplicative__dim10      |                    0.85     | HIGH_MGB_FIT     | mgb_win           |          0.530304  |              12.4945 |                          1        |                        0         |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|conditional_multiplicative__dim20 |                    0.846111 | HIGH_MGB_FIT     | mgb_win           |          0.835021  |              18.8098 |                          1        |                        1         |                               1 |                   0.234547 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|mixed_gate_regime__dim20          |                    0.836787 | HIGH_MGB_FIT     | mgb_win           |          0.580614  |              23.3185 |                          1        |                        0         |                               1 |                   0.941362 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|binary|regime_logit__dim20                   |                    0.778651 | HIGH_MGB_FIT     | mgb_win           |          0.0203423 |              60.5012 |                          1        |                        0.777778  |                               1 |                   0.119845 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|regime_slope__dim20               |                    0.774386 | HIGH_MGB_FIT     | mgb_win           |          0.461747  |              20.3437 |                          0.824712 |                        1         |                               1 |                   0.303916 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|categorical_regime_slope__dim10   |                    0.709784 | HIGH_MGB_FIT     | mgb_win           |          0.360544  |              18.621  |                          0.577298 |                        0.333333  |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|categorical_regime_slope__dim20   |                    0.709784 | HIGH_MGB_FIT     | mgb_win           |          0.472873  |              26.3269 |                          0.577298 |                        0.333333  |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|conditional_multiplicative__dim10 |                    0.679781 | HIGH_MGB_FIT     | mgb_win           |          0.758279  |              11.7665 |                          0.930573 |                        0         |                               1 |                   0.176136 |
| tabular_otherdata_0629|random300|onehot|regression|Moneyball.csv                                      |                    0.510577 | HIGH_MGB_FIT     | mgb_win           |          0.0461205 |             289.436  |                          0        |                        0         |                               0 |                   1        |
| tabarena_main_0624|random600|onehot|regression|healthcare_insurance_expenses.csv                      |                    0.481452 | MEDIUM_MGB_FIT   | mgb_win           |          0.0116465 |              38.1041 |                          0        |                        0         |                               0 |                   0.417474 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|categorical_feature_switch__dim10 |                    0.475    | MEDIUM_MGB_FIT   | mgb_win           |          0.137145  |              13.3875 |                          0        |                        0.5       |                               1 |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|categorical_feature_switch__dim20 |                    0.474401 | MEDIUM_MGB_FIT   | mgb_win           |          0.0980909 |              18.874  |                          0        |                        0.5       |                               1 |                   1        |
| tabular_otherdata_0629|random300|drop|multiclass|vowel.csv                                            |                    0.44826  | MEDIUM_MGB_FIT   | mgb_win           |          0.42127   |              37.0949 |                          1        |                        0.380952  |                               0 |                   0.695168 |

## Strong Negative Cases

| dataset_key                                                                                  |   mechanism_alignment_score | mechanism_band   | observed_winner   |   test_improvement |   runtime_ratio_mean |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_signal_support |
|:---------------------------------------------------------------------------------------------|----------------------------:|:-----------------|:------------------|-------------------:|---------------------:|----------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|
| tabarena_main_0624|random300|drop|binary|polish_companies_bankruptcy.csv                     |                  0          | LOW_MGB_FIT      | xgb_win           |         -0.087593  |           127.068    |                         0         |                         1        |                       0         |                  0.139357  |
| tabarena_main_0624|random300|drop|binary|seismic-bumps.csv                                   |                  0.00405084 | LOW_MGB_FIT      | xgb_win           |         -0.0195404 |            39.2718   |                         0.412195  |                         0.857143 |                       0         |                  0.178237  |
| tabular_otherdata_0629|random300|drop|binary|haberman.csv                                    |                  0.0161935  | LOW_MGB_FIT      | xgb_win           |         -0.0162092 |             5.67014  |                         0.0647739 |                         0        |                       0         |                  0.304859  |
| tabular_otherdata_0629|random300|onehot|binary|haberman.csv                                  |                  0.0161935  | LOW_MGB_FIT      | xgb_win           |         -0.0162092 |             5.70968  |                         0.0647739 |                         0        |                       0         |                  0.304859  |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|pure_step_control__dim10 |                  0.0745152  | LOW_MGB_FIT      | xgb_win           |         -0.0329375 |            13.6225   |                         0         |                         0.291667 |                       0         |                  0.0209881 |
| tabular_otherdata_0629|random300|onehot|regression|energy_efficiency.csv                     |                  0.0877049  | LOW_MGB_FIT      | xgb_win           |         -0.0325849 |            40.6924   |                         0         |                         0.291667 |                       0         |                  1         |
| tabular_otherdata_0629|random300|drop|regression|energy_efficiency.csv                       |                  0.0877049  | LOW_MGB_FIT      | xgb_win           |         -0.0325849 |            32.0174   |                         0         |                         0.291667 |                       0         |                  1         |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|pure_step_control__dim20 |                  0.133902   | LOW_MGB_FIT      | xgb_win           |         -0.0450405 |            24.2182   |                         0         |                         0.916667 |                       0         |                  0.0659968 |
| tabarena_main_0624|random600|drop|regression|diamonds.csv                                    |                  0.200572   | LOW_MGB_FIT      | xgb_win           |         -0.0130364 |            10.1025   |                         0.0510362 |                         0.708333 |                       0         |                  0.384638  |
| tabular_otherdata_0629|random300|onehot|regression|space_ga.csv                              |                  0.21297    | LOW_MGB_FIT      | xgb_win           |         -0.0156538 |             7.86251  |                         0         |                         0        |                       0         |                  0.417105  |
| tabular_otherdata_0629|random300|drop|regression|space_ga.csv                                |                  0.21297    | LOW_MGB_FIT      | xgb_win           |         -0.0156538 |             6.54929  |                         0         |                         0        |                       0         |                  0.417105  |
| tabarena30_0629|xgb_best_mgb30|drop|binary|APSFailure.csv                                    |                  0.221329   | LOW_MGB_FIT      | xgb_win           |         -0.0179065 |            12.6162   |                         0         |                         1        |                       0         |                  0.857713  |
| tabarena30_0629|xgb_best_mgb30|drop|binary|taiwanese_bankruptcy_prediction.csv               |                  0.25197    | LOW_MGB_FIT      | xgb_win           |         -0.0274287 |            13.0663   |                         1         |                         1        |                       0         |                  0.284406  |
| tabular_otherdata_0629|random300|onehot|multiclass|balance-scale.csv                         |                  0.285915   | LOW_MGB_FIT      | xgb_win           |         -0.46675   |            13.625    |                         1         |                         0        |                       0.0942508 |                  0.482917  |
| tabular_otherdata_0629|random300|drop|multiclass|balance-scale.csv                           |                  0.285915   | LOW_MGB_FIT      | xgb_win           |         -0.46675   |            14.1765   |                         1         |                         0        |                       0.0942508 |                  0.482917  |
| tabarena_main_0624|random600|drop|regression|houses.csv                                      |                  0.28784    | LOW_MGB_FIT      | xgb_win           |         -0.203699  |             8.03045  |                         0         |                         0.291667 |                       0         |                  0.512942  |
| tabarena_main_0624|random600|onehot|regression|houses.csv                                    |                  0.28784    | LOW_MGB_FIT      | xgb_win           |         -0.203699  |             8.0938   |                         0         |                         0.291667 |                       0         |                  0.512942  |
| tabarena30_0629|xgb_best_mgb30|onehot|binary|blood-transfusion-service-center.csv            |                  0.321945   | MEDIUM_MGB_FIT   | xgb_win           |         -0.0185796 |             0.416488 |                         0.671064  |                         0        |                       0         |                  0.14825   |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|feature_switch__dim20    |                  0.4        | MEDIUM_MGB_FIT   | xgb_win           |         -0.0196882 |            16.4743   |                         0         |                         0        |                       1         |                  1         |
| tabular_otherdata_0629|random300|drop|regression|boston.csv                                  |                  0.411979   | MEDIUM_MGB_FIT   | xgb_win           |         -0.131201  |            20.3086   |                         1         |                         0.708333 |                       0         |                  1         |

## False High-Fit Cases

_No rows._

## Missed Low-Fit Wins

| dataset_key                                                                              |   mechanism_alignment_score | mechanism_band   | observed_winner   |   test_improvement |   runtime_ratio_mean |   component_linear_leaf_advantage |   component_cond_pred_decoupling |   component_local_heterogeneity |   component_signal_support |
|:-----------------------------------------------------------------------------------------|----------------------------:|:-----------------|:------------------|-------------------:|---------------------:|----------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|
| tabarena30_0629|xgb_best_mgb30|onehot|multiclass|maternal_health_risk.csv                |                    0.19645  | LOW_MGB_FIT      | mgb_win           |          0.0594551 |              4.30017 |                          0.106031 |                         0.5      |                        0        |                   0.37331  |
| tabular_otherdata_0629|random300|drop|regression|Moneyball.csv                           |                    0.215281 | LOW_MGB_FIT      | mgb_win           |          0.0445929 |             31.804   |                          0        |                         0        |                        0        |                   1        |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|global_linear__dim10 |                    0.05     | LOW_MGB_FIT      | mgb_win           |          0.041519  |             10.0964  |                          0        |                         0        |                        0        |                   1        |
| tabarena_main_0624|random300|drop|multiclass|anneal.csv                                  |                    0.184052 | LOW_MGB_FIT      | mgb_win           |          0.038439  |             37.7499  |                          0        |                         0.333333 |                        0        |                   0.399469 |
| synthetic_0629|synthetic_random300_single_seed|synthetic|regression|global_linear__dim20 |                    0.05     | LOW_MGB_FIT      | mgb_win           |          0.0307157 |             16.8538  |                          0        |                         0        |                        0        |                   1        |
| tabarena30_0629|xgb_best_mgb30|onehot|multiclass|website_phishing.csv                    |                    0.246636 | LOW_MGB_FIT      | mgb_win           |          0.027517  |              4.3519  |                          0        |                         1        |                        0        |                   0.525755 |
| tabarena_main_0624|random300|onehot|multiclass|website_phishing.csv                      |                    0.246636 | LOW_MGB_FIT      | mgb_win           |          0.0256471 |            245.19    |                          0        |                         1        |                        0        |                   0.525755 |
| tabular_otherdata_0629|random300|drop|binary|ilpd.csv                                    |                    0.259801 | LOW_MGB_FIT      | mgb_win           |          0.0199818 |             31.9211  |                          0.715285 |                         0.5      |                        0.369956 |                   0.411627 |
| tabarena_main_0624|random300|onehot|multiclass|maternal_health_risk.csv                  |                    0.19645  | LOW_MGB_FIT      | mgb_win           |          0.0176666 |             21.1814  |                          0.106031 |                         0.5      |                        0        |                   0.37331  |
| tabarena_main_0624|random300|drop|multiclass|maternal_health_risk.csv                    |                    0.19645  | LOW_MGB_FIT      | mgb_win           |          0.0176666 |             19.8942  |                          0.106031 |                         0.5      |                        0        |                   0.37331  |
| tabarena_main_0624|random600|drop|regression|QSAR_fish_toxicity.csv                      |                    0.19167  | LOW_MGB_FIT      | mgb_win           |          0.0130623 |              8.93715 |                          0        |                         0.916667 |                        0        |                   0.495888 |
| tabarena_main_0624|random600|onehot|regression|QSAR_fish_toxicity.csv                    |                    0.19167  | LOW_MGB_FIT      | mgb_win           |          0.0130622 |              8.27573 |                          0        |                         0.916667 |                        0        |                   0.495888 |
| tabarena_main_0624|random300|drop|binary|credit-g.csv                                    |                    0.247412 | LOW_MGB_FIT      | mgb_win           |          0.0087618 |             21.0721  |                          1        |                         0.857143 |                        0.27979  |                   0.306691 |
| tabular_otherdata_0629|random300|drop|binary|phoneme.csv                                 |                    0.21782  | LOW_MGB_FIT      | mgb_win           |          0.0064284 |              9.52791 |                          0        |                         0        |                        0        |                   0.230204 |
| tabular_otherdata_0629|random300|onehot|binary|phoneme.csv                               |                    0.21782  | LOW_MGB_FIT      | mgb_win           |          0.0064284 |             10.2334  |                          0        |                         0        |                        0        |                   0.230204 |
