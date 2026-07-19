# MGB vs XGB Comprehensive Analysis (0629)

## Scope

本文件由 `mgb_datacollection_by0629/scripts/analyze_mgb_vs_xgb.py` 生成，证据来自 `datacollection0624` 的 TabArena 主体结果以及 `mgb_datacollection_by0629` 中的 TabArena30、tabular other data、synthetic 结果。比较只纳入 XGB 与 current_metagb 在同一 dataset、outer fold 下均 `status=ok` 的 paired folds。

口径：AUC 越高越好；RMSE/logloss 越低越好。AUC 绝对差 `<=1e-4` 记为 tie；RMSE/logloss 相对差 `<=0.1%` 记为 tie。

## Executive Summary

- Paired fold 数：699；dataset-level 可比较单元：142。
- 总体 dataset-level win/tie/loss (MGB/Tie/XGB)：74/7/61。
- `tabarena30_0629` 是 fixed-backbone 协议：先取 XGB best backbone，再搜索 `10 pred_fraction x 3 score_agg`，因此应与完整 Random300/600 主表分开解释。
- `synthetic_0629` 是机制证据：它更适合回答 MetaGB 理念是否成立，而不是代表真实 TabArena 总体分布。

## Coverage Snapshot

| family | dataset_cells | full_5fold_paired | partial_paired | one_sided_only |
| --- | --- | --- | --- | --- |
| synthetic_0629 | 38.000 | 35.000 | 0 | 3 |
| tabarena30_0629 | 45.000 | 16.000 | 4 | 25.000 |
| tabarena_main_0624 | 72.000 | 58.000 | 0 | 14.000 |
| tabular_otherdata_0629 | 35.000 | 29.000 | 0 | 6 |

Coverage reading: one-sided cells are mostly XGB-only or MGB-only unfinished tails, so they are excluded from win/tie/loss. This matters for `tabarena30_0629`, where many one-hot binary tails currently have XGB backbones but no finished MGB30 result yet.

## Data-Supported Findings

**Synthetic mechanism evidence.** Current synthetic paired results give MGB/Tie/XGB = 30/1/4 at dataset-cell level. The A-D/F mechanism families are strongly positive for MGB, while negative controls are not: this is the cleanest evidence that MGB is doing what it is designed to do rather than simply benefiting from a larger search surface. In concrete terms, conditional multiplicative, regime slope, categorical regime slope, mixed gate, and sparse interaction cases show large RMSE reductions, while additive/curved/step controls do not become uniformly favorable.

**TabArena main results are mixed rather than universally positive.** Regression Random600 is 3/0/8 under drop and 5/0/5 under onehot; binary Random300 is 8/0/9 under drop and 5/0/8 under onehot; multiclass Random300 is 2/1/1 under drop and 2/0/1 under onehot. This supports a careful claim: MGB has clear positive pockets, especially some multiclass and selected regression/binary datasets, but the broad real-data average is not a blanket win over XGB.

**TabArena30 validates the fixed-backbone tail strategy, but binary gains are small.** Current TabArena30 gives binary drop 3/0/6, binary onehot 3/1/4, and multiclass onehot 3/0/0. The median runtime ratio is much lower than full Random300/600 in several groups, because only 30 MGB candidates are searched after choosing the XGB backbone. The multiclass onehot results are currently favorable, while binary results are close and dataset-dependent; many AUC deltas are around 0.001-0.008, so these are table-filling/coverage results more than decisive superiority evidence.

**Other tabular data is useful for finding positive pockets but remains mixed overall.** Across current other-tabular cells, MGB/Tie/XGB = 10/4/15. Strong positives include `vowel.csv`, `kin8nm.csv`, `Moneyball.csv`, `ilpd.csv`, and `phoneme.csv`; strong negatives include `balance-scale.csv`, `boston.csv`, and `energy_efficiency.csv`. This aligns with the working hypothesis that MGB helps when local linear response or conditional/predictive separation is present, but can lose on data where constant-leaf partitioning is already a better inductive bias.

**Runtime tradeoff.** MGB is consistently more expensive in full Random300/600 settings; median runtime ratios are commonly 10x-60x depending on feature count and task family. TabArena30 reduces this cost by shrinking the MGB-specific search to 30 candidates, with several binary groups around 2x-3x median ratio, but wide/high-pressure data can still be much slower. The right paper framing is therefore not speed, but whether the extra local-linear expressivity buys enough test performance in the mechanism-aligned regimes.

## Win/Tie/Loss by Family

| family | protocol | encoding | task_type | datasets | paired_folds | dataset_mgb_win | dataset_tie | dataset_xgb_win | fold_mgb_win | fold_tie | fold_xgb_win | median_runtime_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | binary | 6 | 30.000 | 6 | 0 | 0 | 26.000 | 1 | 3 | 46.403 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | 29.000 | 145.000 | 24.000 | 1 | 4 | 124.000 | 0 | 21.000 | 15.231 |
| tabarena30_0629 | xgb_best_mgb30 | drop | binary | 9 | 38.000 | 3 | 0 | 6 | 16.000 | 1 | 21.000 | 2.625 |
| tabarena30_0629 | xgb_best_mgb30 | onehot | binary | 8 | 40.000 | 3 | 1 | 4 | 20.000 | 1 | 19.000 | 2.01982 |
| tabarena30_0629 | xgb_best_mgb30 | onehot | multiclass | 3 | 11.000 | 3 | 0 | 0 | 10.000 | 0 | 1 | 4.26528 |
| tabarena_main_0624 | random300 | drop | binary | 17.000 | 85.000 | 8 | 0 | 9 | 40.000 | 1 | 44.000 | 22.015 |
| tabarena_main_0624 | random300 | drop | multiclass | 4 | 20.000 | 2 | 1 | 1 | 13.000 | 1 | 6 | 26.228 |
| tabarena_main_0624 | random300 | onehot | binary | 13.000 | 65.000 | 5 | 0 | 8 | 35.000 | 1 | 29.000 | 58.984 |
| tabarena_main_0624 | random300 | onehot | multiclass | 3 | 15.000 | 2 | 0 | 1 | 12.000 | 0 | 3 | 21.480 |
| tabarena_main_0624 | random600 | drop | regression | 11.000 | 55.000 | 3 | 0 | 8 | 18.000 | 4 | 33.000 | 10.301 |
| tabarena_main_0624 | random600 | onehot | regression | 10.000 | 50.000 | 5 | 0 | 5 | 27.000 | 2 | 21.000 | 13.810 |
| tabular_otherdata_0629 | random300 | drop | binary | 8 | 40.000 | 2 | 2 | 4 | 15.000 | 4 | 21.000 | 14.018 |
| tabular_otherdata_0629 | random300 | drop | multiclass | 2 | 10.000 | 1 | 0 | 1 | 5 | 0 | 5 | 25.495 |
| tabular_otherdata_0629 | random300 | drop | regression | 7 | 35.000 | 3 | 0 | 4 | 18.000 | 1 | 16.000 | 10.090 |
| tabular_otherdata_0629 | random300 | onehot | binary | 5 | 25.000 | 2 | 2 | 1 | 13.000 | 4 | 8 | 10.368 |
| tabular_otherdata_0629 | random300 | onehot | multiclass | 1 | 5 | 0 | 0 | 1 | 0 | 0 | 5 | 13.141 |
| tabular_otherdata_0629 | random300 | onehot | regression | 6 | 30.000 | 2 | 0 | 4 | 14.000 | 1 | 15.000 | 22.371 |

## Strongest Positive Cases

| family | protocol | encoding | task_type | dataset | paired_folds | primary_metric | xgb_test_mean | mgb_test_mean | test_improvement | runtime_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | conditional_multiplicative__dim20 | 5 | rmse | 2.15226 | 0.355077 | 0.835021 | 18.810 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | conditional_multiplicative__dim10 | 5 | rmse | 1.55038 | 0.374759 | 0.758279 | 11.766 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | mixed_gate_regime__dim20 | 5 | rmse | 0.466074 | 0.195465 | 0.580614 | 23.319 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | sparse_multiplicative__dim20 | 5 | rmse | 0.557822 | 0.2494 | 0.552904 | 20.189 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | sparse_multiplicative__dim10 | 5 | rmse | 0.501428 | 0.235519 | 0.530304 | 12.494 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | sparse_regime_slope__dim10 | 5 | rmse | 0.479326 | 0.249939 | 0.478562 | 12.718 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | categorical_regime_slope__dim20 | 5 | rmse | 0.300775 | 0.158547 | 0.472873 | 26.327 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | sparse_regime_slope__dim20 | 5 | rmse | 0.56243 | 0.297849 | 0.470424 | 20.784 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | regime_slope__dim20 | 5 | rmse | 0.46974 | 0.252839 | 0.461747 | 20.344 |
| tabular_otherdata_0629 | random300 | drop | multiclass | vowel.csv | 5 | logloss | 0.228168 | 0.132048 | 0.42127 | 37.095 |

## Strongest Negative Cases

| family | protocol | encoding | task_type | dataset | paired_folds | primary_metric | xgb_test_mean | mgb_test_mean | test_improvement | runtime_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabular_otherdata_0629 | random300 | onehot | multiclass | balance-scale.csv | 5 | logloss | 0.233793 | 0.342915 | -0.46675 | 13.625 |
| tabular_otherdata_0629 | random300 | drop | multiclass | balance-scale.csv | 5 | logloss | 0.233793 | 0.342915 | -0.46675 | 14.177 |
| tabarena_main_0624 | random600 | drop | regression | houses.csv | 5 | rmse | 0.212653 | 0.25597 | -0.203699 | 8.03045 |
| tabarena_main_0624 | random600 | onehot | regression | houses.csv | 5 | rmse | 0.212653 | 0.25597 | -0.203699 | 8.0938 |
| tabular_otherdata_0629 | random300 | onehot | regression | boston.csv | 5 | rmse | 3.04923 | 3.4493 | -0.131201 | 25.008 |
| tabular_otherdata_0629 | random300 | drop | regression | boston.csv | 5 | rmse | 3.04923 | 3.4493 | -0.131201 | 20.309 |
| tabarena_main_0624 | random300 | drop | binary | polish_companies_bankruptcy.csv | 5 | auc | 0.919304 | 0.831711 | -0.087593 | 127.068 |
| tabarena_main_0624 | random600 | onehot | regression | airfoil_self_noise.csv | 5 | rmse | 1.40665 | 1.52087 | -0.0811986 | 14.679 |
| tabarena_main_0624 | random600 | drop | regression | airfoil_self_noise.csv | 5 | rmse | 1.40665 | 1.52087 | -0.081196 | 13.878 |
| synthetic_0629 | synthetic_random300_single_seed | synthetic | regression | additive_nonlinear_control__dim10 | 5 | rmse | 0.181412 | 0.191317 | -0.0546014 | 10.726 |

## Synthetic Mechanism Reading

| mechanism_type | mgb_win | tie | xgb_win |
| --- | --- | --- | --- |
| A_interaction | 8 | 0 | 0 |
| B_linear_in_leaves | 9 | 0 | 1 |
| C_conditional_predictive | 6 | 0 | 0 |
| D_sparse_interaction | 6 | 0 | 0 |
| E_negative_control | 0 | 1 | 3 |
| F_noise_pressure | 1 | 0 | 0 |

Interpretation: synthetic positive cases should be read through the MetaGB design lens: conditional features define regions, while predictive features provide local one-feature linear responses inside leaves. Negative controls are important because they check whether MGB is merely winning everywhere by search noise.

## Dataset Scale and Runtime

Dataset rows/features in the tables are the processed training matrix sizes observed in logs or synthetic protocol settings. The pressure proxy is `rows * features^2`, which is the relevant rough cost driver for MetaGB candidate evaluation.

| family | encoding | task_type | dataset | paired_folds | rows | features | pressure_proxy | dataset_winner | runtime_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabarena30_0629 | drop | binary | coil2000_insurance_policies.csv | 5 | 9,822.0 | 80.000 | 62,860,800.0 | xgb_win | 17.836 |
| tabarena30_0629 | drop | binary | taiwanese_bankruptcy_prediction.csv | 3 | 6,819.0 | 94.000 | 60,252,684.0 | xgb_win | 13.066 |
| tabarena30_0629 | drop | binary | NATICUSdroid.csv | 3 | 7,491.0 | 86.000 | 55,403,436.0 | mgb_win | 30.609 |
| tabarena_main_0624 | onehot | regression | diamonds.csv | 5 | 53,940.0 | 26.000 | 36,463,440.0 | xgb_win | 80.921 |
| tabarena30_0629 | drop | binary | APSFailure.csv | 2 | 756.000 | 170.000 | 21,848,400.0 | xgb_win | 12.616 |
| tabarena30_0629 | drop | binary | credit_card_clients_default.csv | 5 | 30,000.0 | 23.000 | 15,870,000.0 | xgb_win | 2.2761 |
| tabarena_main_0624 | drop | binary | polish_companies_bankruptcy.csv | 5 | 3,031.0 | 64.000 | 12,414,976.0 | xgb_win | 127.068 |
| tabarena_main_0624 | onehot | binary | GiveMeSomeCredit.csv | 5 | 120,269.0 | 10.000 | 12,026,900.0 | xgb_win | 18.137 |
| tabarena_main_0624 | drop | binary | GiveMeSomeCredit.csv | 5 | 120,269.0 | 10.000 | 12,026,900.0 | xgb_win | 18.137 |
| tabarena_main_0624 | onehot | multiclass | SDSS17.csv | 5 | 78,053.0 | 11.000 | 9,444,413.0 | xgb_win | 21.001 |
| tabarena_main_0624 | drop | multiclass | SDSS17.csv | 5 | 78,053.0 | 11.000 | 9,444,413.0 | xgb_win | 21.692 |
| tabarena30_0629 | onehot | multiclass | anneal.csv | 1 | 898.000 | 82.000 | 6,038,152.0 | mgb_win | 46.844 |
| tabarena30_0629 | drop | binary | heloc.csv | 5 | 10,459.0 | 23.000 | 5,532,811.0 | xgb_win | 1.41264 |
| tabarena30_0629 | drop | binary | jm1.csv | 5 | 10,880.0 | 21.000 | 4,798,080.0 | mgb_win | 5.40606 |
| tabarena_main_0624 | drop | binary | Diabetes130US.csv | 5 | 71,518.0 | 8 | 4,577,152.0 | xgb_win | 20.226 |
| tabarena_main_0624 | onehot | binary | E-CommereShippingData.csv | 5 | 10,999.0 | 19.000 | 3,970,639.0 | xgb_win | 62.004 |
| tabarena30_0629 | onehot | binary | credit-g.csv | 5 | 1,000.0 | 61.000 | 3,721,000.0 | mgb_win | 46.105 |
| tabarena_main_0624 | onehot | binary | credit-g.csv | 5 | 1,000.0 | 61.000 | 3,721,000.0 | mgb_win | 1,114.7 |
| tabarena_main_0624 | onehot | regression | physiochemical_protein.csv | 5 | 45,730.0 | 9 | 3,704,130.0 | mgb_win | 9.78153 |
| tabarena_main_0624 | onehot | regression | miami_housing.csv | 5 | 13,776.0 | 15.000 | 3,099,600.0 | mgb_win | 12.633 |
| tabarena30_0629 | onehot | binary | Is-this-a-good-customer.csv | 5 | 1,723.0 | 42.000 | 3,039,372.0 | xgb_win | 8.52482 |
| tabarena_main_0624 | onehot | binary | Is-this-a-good-customer.csv | 5 | 1,723.0 | 42.000 | 3,039,372.0 | xgb_win | 503.589 |
| tabarena_main_0624 | drop | binary | Amazon_employee_access.csv | 5 | 32,769.0 | 9 | 2,654,289.0 | xgb_win | 17.324 |
| tabarena_main_0624 | onehot | binary | Amazon_employee_access.csv | 5 | 32,769.0 | 9 | 2,654,289.0 | xgb_win | 13.384 |
| tabarena30_0629 | drop | binary | online_shoppers_intention.csv | 5 | 12,330.0 | 14.000 | 2,416,680.0 | mgb_win | 1.0348 |
| tabarena30_0629 | onehot | binary | hazelnut-spread-contaminant-detection.csv | 5 | 2,400.0 | 30.000 | 2,160,000.0 | mgb_win | 2.36687 |
| tabarena_main_0624 | drop | binary | hazelnut-spread-contaminant-detection.csv | 5 | 2,400.0 | 30.000 | 2,160,000.0 | mgb_win | 55.807 |
| tabarena_main_0624 | onehot | binary | hazelnut-spread-contaminant-detection.csv | 5 | 2,400.0 | 30.000 | 2,160,000.0 | mgb_win | 57.382 |
| tabarena_main_0624 | onehot | binary | churn.csv | 5 | 5,000.0 | 19.000 | 1,805,000.0 | mgb_win | 56.504 |
| tabarena_main_0624 | drop | binary | churn.csv | 5 | 5,000.0 | 19.000 | 1,805,000.0 | mgb_win | 56.770 |

## CV/Test Gap

CV/Test gap columns count fold-level disagreements between validation-selected winner and test winner. Large values here mean the model-selection signal is unstable for that dataset, even if the final test mean favors one model.

| family | encoding | task_type | dataset | paired_folds | cv_test_agree_folds | cv_mgb_test_xgb_folds | cv_xgb_test_mgb_folds | dataset_winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabular_otherdata_0629 | drop | regression | boston.csv | 5 | 0 | 4 | 1 | xgb_win |
| tabular_otherdata_0629 | onehot | regression | boston.csv | 5 | 0 | 4 | 1 | xgb_win |
| synthetic_0629 | synthetic | regression | pure_step_control__dim20 | 5 | 2 | 3 | 0 | xgb_win |
| tabarena30_0629 | drop | binary | heloc.csv | 5 | 1 | 3 | 0 | xgb_win |
| tabarena30_0629 | onehot | binary | diabetes.csv | 5 | 2 | 3 | 0 | tie |
| tabular_otherdata_0629 | drop | binary | breast-w.csv | 5 | 2 | 3 | 0 | tie |
| tabular_otherdata_0629 | onehot | binary | breast-w.csv | 5 | 2 | 3 | 0 | tie |
| tabarena_main_0624 | drop | binary | Fitness_Club.csv | 5 | 1 | 2 | 2 | mgb_win |
| tabarena_main_0624 | drop | binary | churn.csv | 5 | 0 | 2 | 2 | mgb_win |
| tabarena_main_0624 | onehot | binary | churn.csv | 5 | 0 | 2 | 2 | mgb_win |
| synthetic_0629 | synthetic | binary | xor_classification__dim10 | 5 | 1 | 2 | 0 | mgb_win |
| synthetic_0629 | synthetic | regression | feature_switch__dim10 | 5 | 3 | 2 | 0 | mgb_win |
| synthetic_0629 | synthetic | regression | feature_switch__dim20 | 5 | 3 | 2 | 0 | xgb_win |
| tabarena30_0629 | drop | binary | jm1.csv | 5 | 3 | 2 | 0 | mgb_win |
| tabarena30_0629 | onehot | binary | Fitness_Club.csv | 5 | 3 | 2 | 0 | xgb_win |
| tabarena_main_0624 | drop | binary | diabetes.csv | 5 | 3 | 2 | 0 | xgb_win |
| tabarena_main_0624 | drop | binary | qsar-biodeg.csv | 5 | 3 | 2 | 0 | xgb_win |
| tabarena_main_0624 | drop | regression | miami_housing.csv | 5 | 3 | 2 | 0 | mgb_win |
| tabarena_main_0624 | onehot | binary | diabetes.csv | 5 | 3 | 2 | 0 | xgb_win |
| tabarena_main_0624 | onehot | binary | qsar-biodeg.csv | 5 | 3 | 2 | 0 | xgb_win |
| tabarena_main_0624 | onehot | regression | miami_housing.csv | 5 | 3 | 2 | 0 | mgb_win |
| tabular_otherdata_0629 | drop | binary | ilpd.csv | 5 | 3 | 2 | 0 | mgb_win |
| tabular_otherdata_0629 | drop | binary | pc1.csv | 5 | 3 | 2 | 0 | xgb_win |
| tabarena_main_0624 | onehot | regression | wine_quality.csv | 5 | 0 | 1 | 4 | mgb_win |
| tabarena_main_0624 | drop | binary | blood-transfusion-service-center.csv | 5 | 1 | 1 | 3 | mgb_win |
| tabarena_main_0624 | onehot | binary | blood-transfusion-service-center.csv | 5 | 1 | 1 | 3 | mgb_win |
| tabarena30_0629 | onehot | binary | Is-this-a-good-customer.csv | 5 | 2 | 1 | 2 | xgb_win |
| tabarena30_0629 | onehot | binary | qsar-biodeg.csv | 5 | 2 | 1 | 2 | mgb_win |
| tabarena_main_0624 | drop | binary | HR_Analytics_Job_Change_of_Data_Scientists.csv | 5 | 2 | 1 | 2 | mgb_win |
| tabarena_main_0624 | drop | multiclass | anneal.csv | 5 | 2 | 1 | 2 | mgb_win |

## Remaining Work Estimate

The estimate below is serial task-hours from observed median completed-task elapsed time in each run directory. Parallel wall time depends on available workers and should be divided by effective worker count only after accounting for slow-tail skew.

| run_dir | total_tasks | completed | failed | remaining | median_ok_elapsed_hours | remaining_serial_task_hours |
| --- | --- | --- | --- | --- | --- | --- |
| mgb_datacollection_by0629/tabarena30_drop/tabarena30_binary_drop_20260628_023950 | 110.000 | 93.000 | 0 | 17.000 | 0.683139 | 11.613 |
| mgb_datacollection_by0629/tabarena30_onehot | 355.000 | 236.000 | 15.000 | 104.000 | 0.556133 | 57.838 |
| mgb_datacollection_by0629/tabular_otherdata_drop | 270.000 | 185.000 | 0 | 85.000 | 0.559122 | 47.525 |
| mgb_datacollection_by0629/tabular_otherdata_onehot | 270.000 | 135.000 | 0 | 135.000 | 0.286669 | 38.700 |

## Evidence Tables

- `derived/fold_results.csv`: parsed model/fold records.
- `derived/paired_fold_results.csv`: only folds where XGB and MGB both finished.
- `derived/dataset_level_mgb_vs_xgb.csv`: dataset-level aggregation.
- `tables/table_dataset_level_mgb_vs_xgb.md`: full dataset table.
- `tables/table_synthetic_mechanism.md`: synthetic mechanism view.
- `tables/table_runtime_quality_tradeoff.md`: runtime-quality view.
- `tables/table_cv_test_gap.md`: CV/test consistency view.
