# datacollection0702 附录表

## A. 统计单位与过滤规则

应用已确认过滤规则后，解析后的 0702 快照包含 1959 条可用 model/fold 记录。
最终 paired 分析包含 815 个 paired folds 和 163 个 dataset-level units。

按协议排除：

- Synthetic `dim50` 产物。
- `tabarena30_drop0702` 中的 `Bioresponse.csv` 残余任务。
- 任一侧缺少 `ok` test metric 的 unpaired folds。

## B. 主分组表

| 数据族 | 协议 | 预处理 | 任务 | Dataset units | Paired folds | Dataset M/T/X | Fold M/T/X | 中位 runtime ratio |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| other_tabular_0702 | random300 | drop | binary | 10 | 50 | 3/2/5 | 20/4/26 | 14.19x |
| other_tabular_0702 | random300 | drop | multiclass | 4 | 20 | 1/0/3 | 8/0/12 | 71.85x |
| other_tabular_0702 | random300 | drop | regression | 9 | 45 | 4/0/5 | 22/1/22 | 19.83x |
| other_tabular_0702 | random300 | onehot | binary | 10 | 50 | 3/2/5 | 21/4/25 | 27.05x |
| other_tabular_0702 | random300 | onehot | multiclass | 4 | 20 | 1/0/3 | 8/0/12 | 94.27x |
| other_tabular_0702 | random300 | onehot | regression | 9 | 45 | 4/0/5 | 23/1/21 | 20.00x |
| synthetic_0702 | random300_single_seed | synthetic | binary | 6 | 30 | 6/0/0 | 26/1/3 | 46.40x |
| synthetic_0702 | random300_single_seed | synthetic | regression | 32 | 160 | 25/1/6 | 129/0/31 | 16.39x |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | 10 | 50 | 2/2/6 | 17/6/27 | 6.37x |
| tabarena30_0702 | xgb_best_mgb30 | onehot | binary | 8 | 40 | 3/1/4 | 20/1/19 | 2.02x |
| tabarena30_0702 | xgb_best_mgb30 | onehot | multiclass | 3 | 15 | 3/0/0 | 12/0/3 | 4.59x |
| tabarena_main_0702 | random300 | drop | binary | 17 | 85 | 8/0/9 | 40/1/44 | 22.01x |
| tabarena_main_0702 | random300 | drop | multiclass | 4 | 20 | 2/1/1 | 13/1/6 | 26.23x |
| tabarena_main_0702 | random300 | onehot | binary | 13 | 65 | 5/0/8 | 35/1/29 | 58.98x |
| tabarena_main_0702 | random300 | onehot | multiclass | 3 | 15 | 2/0/1 | 12/0/3 | 21.48x |
| tabarena_main_0702 | random600 | drop | regression | 11 | 55 | 3/0/8 | 18/4/33 | 10.30x |
| tabarena_main_0702 | random600 | onehot | regression | 10 | 50 | 5/0/5 | 27/2/21 | 13.81x |

## C. Synthetic 机制表

| 机制 | Dataset units | Paired folds | MetaGB | Tie | XGB | 中位 improvement | 中位 runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: interaction / XOR / multiplicative | 8 | 40 | 8 | 0 | 0 | 0.2068 | 23.20x |
| B: linear-in-leaves | 10 | 50 | 9 | 0 | 1 | 0.0524 | 16.62x |
| C: conditional-predictive separation | 6 | 30 | 6 | 0 | 0 | 0.3619 | 18.60x |
| D: sparse interaction | 6 | 30 | 6 | 0 | 0 | 0.4745 | 14.54x |
| E: negative controls | 6 | 30 | 0 | 1 | 5 | -0.0390 | 15.62x |
| F: noise pressure | 2 | 10 | 2 | 0 | 0 | 0.1254 | 16.24x |

## D. 代表性正向案例

| 数据族 | 协议 | 预处理 | 任务 | Dataset | Paired folds | Metric | XGB | MetaGB | Improvement | Runtime |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| synthetic_0702 | random300_single_seed | synthetic | regression | conditional_multiplicative__dim20 | 5 | rmse | 2.1523 | 0.3551 | 0.8350 | 18.76x |
| synthetic_0702 | random300_single_seed | synthetic | regression | conditional_multiplicative__dim10 | 5 | rmse | 1.5504 | 0.3748 | 0.7583 | 11.78x |
| synthetic_0702 | random300_single_seed | synthetic | regression | mixed_gate_regime__dim20 | 5 | rmse | 0.4661 | 0.1955 | 0.5806 | 23.28x |
| synthetic_0702 | random300_single_seed | synthetic | regression | sparse_multiplicative__dim20 | 5 | rmse | 0.5578 | 0.2494 | 0.5529 | 20.17x |
| synthetic_0702 | random300_single_seed | synthetic | regression | sparse_multiplicative__dim10 | 5 | rmse | 0.5014 | 0.2355 | 0.5303 | 12.47x |
| synthetic_0702 | random300_single_seed | synthetic | regression | categorical_regime_slope__dim20 | 5 | rmse | 0.3008 | 0.1585 | 0.4729 | 26.18x |
| other_tabular_0702 | random300 | onehot | multiclass | vowel.csv | 5 | logloss | 0.2251 | 0.1271 | 0.4352 | 60.72x |
| other_tabular_0702 | random300 | drop | multiclass | vowel.csv | 5 | logloss | 0.2282 | 0.1320 | 0.4213 | 37.15x |
| synthetic_0702 | random300_single_seed | synthetic | regression | multiplicative__dim20 | 5 | rmse | 0.3028 | 0.1804 | 0.4042 | 20.41x |
| synthetic_0702 | random300_single_seed | synthetic | regression | regime_slope_multi__dim20 | 5 | rmse | 0.3780 | 0.2258 | 0.4028 | 19.85x |

对于 AUC，improvement 表示 MetaGB 减 XGB 的绝对差。对于 RMSE/logloss，
improvement 表示从 XGB 到 MetaGB 的相对下降幅度。

## E. 代表性负向案例

| 数据族 | 协议 | 预处理 | 任务 | Dataset | Paired folds | Metric | XGB | MetaGB | Improvement | Runtime |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| other_tabular_0702 | random300 | drop | regression | cpu_activity.csv | 5 | rmse | 2.1868 | 3.9344 | -0.7992 | 19.99x |
| other_tabular_0702 | random300 | onehot | regression | cpu_activity.csv | 5 | rmse | 2.1868 | 3.9344 | -0.7992 | 18.72x |
| other_tabular_0702 | random300 | drop | multiclass | balance-scale.csv | 5 | logloss | 0.2338 | 0.3429 | -0.4668 | 14.03x |
| other_tabular_0702 | random300 | onehot | multiclass | balance-scale.csv | 5 | logloss | 0.2338 | 0.3429 | -0.4668 | 13.14x |
| tabarena_main_0702 | random600 | drop | regression | houses.csv | 5 | rmse | 0.2127 | 0.2560 | -0.2037 | 7.97x |
| tabarena_main_0702 | random600 | onehot | regression | houses.csv | 5 | rmse | 0.2127 | 0.2560 | -0.2037 | 8.04x |
| other_tabular_0702 | random300 | drop | multiclass | segment.csv | 5 | logloss | 0.0471 | 0.0550 | -0.1672 | 123.80x |
| synthetic_0702 | random300_single_seed | synthetic | regression | additive_nonlinear_control__dim20 | 5 | rmse | 0.1857 | 0.2149 | -0.1570 | 17.61x |
| other_tabular_0702 | random300 | drop | regression | boston.csv | 5 | rmse | 3.0492 | 3.4493 | -0.1312 | 20.47x |
| tabarena_main_0702 | random300 | drop | binary | polish_companies_bankruptcy.csv | 5 | auc | 0.9193 | 0.8317 | -0.0876 | 124.90x |

## F. TabArena30 Paired Units

| 预处理 | 任务 | Dataset | XGB | MetaGB | 结果 | Runtime |
| --- | --- | --- | ---: | ---: | --- | ---: |
| drop | binary | APSFailure.csv | 0.9898 | 0.9673 | XGB | 44.08x |
| drop | binary | NATICUSdroid.csv | 0.9856 | 0.9856 | tie | 37.42x |
| drop | binary | bank-marketing.csv | 0.7073 | 0.7070 | XGB | 0.50x |
| drop | binary | coil2000_insurance_policies.csv | 0.7624 | 0.7612 | XGB | 12.70x |
| drop | binary | credit_card_clients_default.csv | 0.7853 | 0.7821 | XGB | 2.32x |
| drop | binary | customer_satisfaction_in_airline.csv | 0.9919 | 0.9920 | tie | 21.44x |
| drop | binary | heloc.csv | 0.7992 | 0.7988 | XGB | 1.48x |
| drop | binary | jm1.csv | 0.7527 | 0.7552 | MetaGB | 5.88x |
| drop | binary | online_shoppers_intention.csv | 0.9151 | 0.9156 | MetaGB | 1.06x |
| drop | binary | taiwanese_bankruptcy_prediction.csv | 0.9479 | 0.9197 | XGB | 19.02x |
| onehot | binary | credit-g.csv | 0.7878 | 0.7954 | MetaGB | 33.80x |
| onehot | binary | hazelnut-spread-contaminant-detection.csv | 0.9795 | 0.9817 | MetaGB | 2.10x |
| onehot | binary | qsar-biodeg.csv | 0.9325 | 0.9331 | MetaGB | 4.84x |
| onehot | multiclass | anneal.csv | 0.0266 | 0.0262 | MetaGB | 99.75x |
| onehot | multiclass | maternal_health_risk.csv | 0.4005 | 0.3766 | MetaGB | 3.61x |
| onehot | multiclass | website_phishing.csv | 0.2455 | 0.2387 | MetaGB | 4.27x |

Onehot TabArena30 还包含大量只有 XGB、没有 paired MetaGB counterpart 的记录；
这些属于 coverage 记录，不进入 test comparison。

## G. 高压力 Dataset Units

| 数据族 | 协议 | 预处理 | 任务 | Dataset | Rows | Features | rows * features^2 | 结果 | Runtime |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | coil2000_insurance_policies.csv | 9822 | 80 | 62.86M | XGB | 12.70x |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | taiwanese_bankruptcy_prediction.csv | 6819 | 94 | 60.25M | XGB | 19.02x |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | NATICUSdroid.csv | 7491 | 86 | 55.40M | tie | 37.42x |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | customer_satisfaction_in_airline.csv | 129487 | 18 | 41.95M | tie | 21.44x |
| tabarena_main_0702 | random600 | onehot | regression | diamonds.csv | 53940 | 26 | 36.46M | XGB | 83.81x |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | APSFailure.csv | 756 | 170 | 21.85M | XGB | 44.08x |
| tabarena30_0702 | xgb_best_mgb30 | drop | binary | credit_card_clients_default.csv | 30000 | 23 | 15.87M | XGB | 2.32x |
| tabarena_main_0702 | random300 | drop | binary | polish_companies_bankruptcy.csv | 3031 | 64 | 12.41M | XGB | 124.90x |

## H. Coverage 说明

重要的不完整 block：

- `tabarena30_onehot0702`：status 显示 355 total tasks、240 completed、15
  failed、100 pending。因此 paired subset 是 partial evidence。
- Other-tabular multiclass 中有若干 XGB-only completed datasets，包括
  `mfeat-fourier.csv`、`mfeat-karhunen.csv` 和 `satimage.csv`。这些不计入
  win/loss 统计。
- TabArena main binary/multiclass 0624 status files 含有许多历史 pending
  entries。最终比较集合由 paired-fold parser 定义，而不是由 status counters
  定义。

## I. 推荐表格使用方式

论文草稿中建议：

- 使用 Section C 作为 method-mechanism table。
- 使用 Section B 作为 compact empirical summary table。
- 使用 Sections D 和 E 作为 qualitative case studies。
- 使用 Section G 支撑 computational tradeoff 讨论。
