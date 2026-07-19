# MetaGB 实验数据参考手册（0702 + 0716）



> 生成日期：2026-07-19。用途：集中保存实验名称、研究目的、主要结论、说服力边界、表格级事实、统计口径和原始路径，供论文写作与后续重排结果表格。本文档不替代 fold-level 日志。



这份手册把现有实验组织成一条证据链：真实数据实验检验 MetaGB 在常规表格任务上的实际竞争力，强基线与排名实验确定其相对位置，合成数据实验验证预期机制，正式消融进一步区分 split 与 predictive leaf 的贡献，时序实验则探索方法能否迁移到窗口化序列预测。整体证据不是“MetaGB 在所有任务上普遍占优”，而是“MetaGB 在匹配其结构假设的条件异质性、交互和局部斜率机制上具有明确优势，但在通用真实数据、负对照、稀疏乘法和计算成本方面存在清楚边界”。



<a id="experiment-index"></a>

## 实验总览与结论索引



下表先回答“做了哪些实验、为什么做、结果说明什么”。其中“说服力”描述该实验能够支撑的论文论断强度，而不是对实验质量作笼统评分。



| 编号 | 实验名称 | 动机与目的 | 当前效果 | 能够支撑的说服力 | 详细数据 |
| --- | --- | --- | --- | --- | --- |
| EXP-R1 | 真实表格主比较：MetaGB vs XGBoost | 检验方法在常规 binary、multiclass、regression 数据上的实际效果，并观察 drop/one-hot 预处理是否改变结论。 | 125 个真实 cells 为 49/8/68；结果高度依赖数据集，MetaGB 在 ILPD、vowel、kin8nm 等数据上有优势，但整体并未超过 XGBoost。 | 中等。能够证明条件性竞争力和失败边界，不能支撑“真实数据全面优于 XGBoost”。 | [第 4 节](#real-mgb-xgb) |
| EXP-R2 | TabArena30 Tail Protocol | 在已选 XGBoost backbone 上增加 MetaGB-specific 30-grid，考察尾部数据集和有限附加搜索下的收益。 | 21 个 cells 为 8/3/10，正负结果接近；它补充了长尾覆盖，但预算与 Random300/600 主协议不同。 | 中等偏弱。适合作为补充覆盖和稳健性材料，不应与主协议合并宣称统一胜率。 | [第 4.3 节](#real-mgb-xgb) |
| EXP-R3 | 强基线与多模型排名 | 回答 MetaGB 相对 Linear、随机森林类模型、梯度提升树和 MLP 类模型处于什么位置，并避免只与 XGBoost 比较。 | 八模型 46-cell 口径中 MetaGB 平均 rank 4.12，列第 3/8；九模型 45-cell 口径中为 4.59，列第 3/9。逐 cell 最强可用 baseline 对 MetaGB 为 42/2/2。 | 强但结论受限。能够支撑“整体排名有竞争力”，同时明确否定“普遍击败最强基线”。 | [第 5 节](#strong-tabular-baselines) |
| EXP-S1 | Synthetic Mechanism Benchmark（10/20 维） | 用可控数据机制检验 MetaGB 是否确实擅长条件异质性、交互和叶内预测结构，而非只在个别真实数据上偶然获益。 | 38 个三模型 cells 中 MetaGB 最优 28 个，XGBoost 3 个，LightGBM 7 个；优势集中在 interaction、conditional-predictive 和部分 linear-in-leaves 机制，negative controls 由 LightGBM 占优。 | 较强的机制证据。正负对照方向一致，但主 benchmark 只有一个 data seed，不能单独支撑跨随机种子稳健性。 | [第 6 节](#synthetic-main) |
| EXP-S2 | Synthetic Dim50 扩展 | 考察机制优势在更高维输入下是否仍然存在。 | 当前 11 个完整五折配对 cells 中 MetaGB 对 XGBoost 为 11/0/0，但只覆盖机制匹配子集，且缺少完整 LightGBM 对齐。 | 支持性证据。结果方向强，但 coverage 有明显选择性，只能作为高维补充。 | [第 6.1 节](#synthetic-main) |
| EXP-A1 | 正式 Fixed-Backbone 组件消融 | 在共享 backbone、split 和 model seed 的条件下，区分 split scoring 与 predictive leaf 两个组件分别带来的因果贡献。 | 完整覆盖 7 scenarios × 2 dimensions × 3 seeds × 5 folds × 16 variants。regime_slope 的 Full-Bias 为 +43.1%/+53.2%，主要由 split 驱动；feature_switch 的 leaf 效应为 +11.3%/+13.6%；controls、regime_logit leaf 和 sparse_multiplicative 暴露了明确失败条件。 | 当前最强的组件级证据。共享 backbone 和 seed-level 配对提高了可解释性，但它回答组件贡献，不等同于各变体独立完全调优后的最优性能。 | [第 7 节](#formal-ablation) |
| EXP-A2 | Predictive-Feature Fraction 敏感性 | 检验 predictive leaf 使用的预测特征比例是否存在统一剂量规律，并判断收益是否依赖覆盖率。 | 不存在跨场景统一单调关系：regime_slope 随比例上升后平台，feature_switch 由负转正，regime_logit 与 sparse_multiplicative 在高比例下恶化。 | 较强的边界证据。能够说明该超参数与数据机制耦合，不能支持“预测特征越多越好”。 | [第 7.4 节](#formal-ablation) |
| EXP-A3 | Score Aggregation 消融 | 比较 greedy、mean 和 median 聚合，判断结果是否依赖特定 score aggregation。 | Mean 的效果依赖场景；median 在 Full 中 11/14 个 cells、SplitOnly 中 13/14 个 cells 的配对区间完全为负，不能作为稳健默认替代。 | 较强的设计选择证据。支持保留 greedy，同时揭示 aggregation 不是可随意替换的实现细节。 | [第 7.5 节](#formal-ablation) |
| EXP-T1 | Time-Series 迁移实验 | 把表格树模型用于窗口化时序预测，探索 MetaGB 的条件建模能力能否迁移到非标准表格场景。 | MetaGB 在 9 个四模型共同 cells 上相对最强常规树为 7/2，RMSE 平均/中位改善 +28.02%/+26.50%，但运行时间中位数为 100.31x，且完整计划仅覆盖 9/42 个 MetaGB cells。 | 探索性证据。效果幅度可观，但 coverage 非随机、fixed split 且成本很高，不能作为成熟的通用时序结论。 | [第 8 节](#time-series) |



### 总体证据判断



现有实验最有说服力的部分是 synthetic mechanism benchmark 与正式 paired ablation：它们共同表明 MetaGB 的收益来自特定的条件结构，且 split scoring 与 predictive leaf 的作用可以被分别观察。真实数据和强基线实验提供了必要的外部约束：MetaGB 的总体平均排名可以进入前列，但收益并不普遍，逐 cell oracle baseline 仍明显更强。因而论文贡献宜写成“面向条件异质性和叶内预测结构的机制创新，并在匹配场景中取得显著收益”，而不宜写成“通用表格学习性能全面领先”。时序结果适合放在扩展性或案例研究位置，不能替代完整的真实表格主结果。



### 阅读路径



- 想快速判断论文论断是否站得住：先读本节、[强基线排名](#strong-tabular-baselines)、[正式消融](#formal-ablation)和[限制](#integrity-limitations)。

- 想重新制作真实数据主表：读[第 4 节](#real-mgb-xgb)，再到[数据入口](#data-entrypoints)读取 canonical CSV。

- 想重新制作 synthetic 或 ablation 表：读[第 6 节](#synthetic-main)和[第 7 节](#formal-ablation)。

- 想核对历史数字为何不同：读[历史口径对照](#scope-history)。



## 文档目录



- [实验总览与结论索引](#experiment-index)

- [1. 使用目的与证据层级](#evidence-hierarchy)

- [2. 统一统计定义](#statistical-definitions)

- [3. Canonical 快照总览](#canonical-snapshot)

- [4. 真实数据：MetaGB 与 XGBoost](#real-mgb-xgb)

- [5. 强 Tabular Baselines](#strong-tabular-baselines)

- [6. Synthetic 主 Benchmark](#synthetic-main)

- [7. 正式 Fixed-Backbone Synthetic 消融](#formal-ablation)

- [8. Time-Series](#time-series)

- [9. 历史口径与当前口径对照](#scope-history)

- [10. 数据完整性、已知差异和限制](#integrity-limitations)

- [11. 推荐的数据入口](#data-entrypoints)



<a id="evidence-hierarchy"></a>

## 1. 使用目的与证据层级

本手册以 dataset/scenario/cell 级结果为主要记录单位。底层证据仍是原始 log、JSON、CSV 和 split/parameter audit；canonical CSV 用于去重、过滤和统一统计；Markdown 报告用于解释，不覆盖结构化产物。重复镜像目录不重复计数。

证据优先级为：原始成功任务与审计文件 > canonical 结构化结果 > 最新 0716 结论文档 > 0702 历史分析文档。若历史文档与最新 canonical 结果冲突，以最新 canonical 结果为准，并在本手册中保留历史值及变化原因。

| 证据块 | 原始目录 | 推荐结构化入口 | 统计单位 |
| --- | --- | --- | --- |
| 0702 MetaGB-XGBoost tabular | description_project/datacollection0702/{binary,multiclass,regression,tabular}_* | dataset_level_results.csv; real_mgb_xgb_dataset_comparisons.csv | paired outer fold -> dataset/preprocessing cell |
| 0702 TabArena30 tail | description_project/datacollection0702/tabarena30_{drop,onehot}0702 | dataset_level_results.csv | fixed XGB backbone + MetaGB-specific 30-grid |
| 0702 synthetic main | description_project/datacollection0702/synthetic_main1020full_random300 | dataset_level_results.csv | scenario x dimension x five outer folds; seed 42 |
| 0716 synthetic LightGBM | dataAddition0716/synbaseline_lgbm0716 | synthetic_three_model_cells.csv; synthetic_three_model_summary.csv | 38 cells x five folds; LightGBM 190/190 |
| 0716 dim50 partial | dataAddition0716/syn102050part0716 | raw task JSON; 本手册离线重算 | only complete paired five-fold cells enter the table |
| 0716 formal ablation | dataAddition0716/metagb_ablation_syn/reanalysis/syn_formal_analysis | main_ablation.csv; main_effects.csv; pred_effects.csv; scoreagg_effects.csv | three seed-level OOF metrics; folds are not iid repeats |
| 0702+0716 strong tabular baselines | description_project/datacollection0702/datacollection_baselinemodel_random300 + dataAddition0716/tab_otherBaselinemodel0716 | baseline_dataset_results.csv; real_baseline_rank_matrix.csv | task union with newer logs overriding older duplicates |
| 0702+0716 time series | description_project/datacollection0702/datacollection_ts_tree_seed42__non_pems + dataAddition0716/ts0716 | timeseries_results.csv; timeseries_metagb_comparison.csv | fixed split dataset x seq_len x horizon cell |

<a id="statistical-definitions"></a>

## 2. 统一统计定义

Binary 主指标为 AUC（越大越好）；multiclass 主指标为 logloss（越小越好）；regression、synthetic regression 和 time series 主指标为 RMSE（越小越好）。AUC 的描述性 tie 为绝对差不超过 1e-4；RMSE/logloss 的 tie 为相对差不超过 0.1%。

Tabular dataset cell 先对完整五个 outer folds 的 test metric 取均值。Synthetic main cell 同样按 scenario × dimension 聚合五个 outer folds。正式消融先用五折 OOF prediction 为每个 data seed 重算一个指标，再对三个 seed-level 配对点计算效应；五个 outer folds 不是五个独立重复。Time-series cell 使用固定 split，不具有跨 split 方差。

Rank 只在同一 cell 内比较相同指标方向。精确相等时使用 average rank；average rank 再对 cells 等权平均。Rank tie 与描述性 W/T/L tie 不是同一规则。Drop 和 one-hot 被视为不同 cells，因此同一底层数据集可以贡献两个 cells。

对于 AUC，signed improvement 定义为 MetaGB - baseline；对于 RMSE/logloss，定义为 (baseline - MetaGB) / |baseline|。正值均表示 MetaGB 更好。Runtime ratio 使用任务日志中的端到端 elapsed time，包含模型选择和最终拟合，不是纯训练复杂度或推理吞吐。

<a id="canonical-snapshot"></a>

## 3. Canonical 快照总览

| 项目 | 值 |
| --- | --- |
| Canonical parser | canonical-v1.2.0 |
| 解析后的 fold rows | 1844 |
| 完整 dataset/scenario units | 163 |
| Paired outer folds | 815 |
| 全部 M/T/X | 80/9/74 |
| 真实 tabular units | 125 |
| 真实 tabular M/T/X | 49/8/68 |
| Synthetic 10/20 units | 38 |
| Synthetic M/T/X | 31/1/6 |
| Synthetic 三模型最优 | MetaGB 28 / XGBoost 3 / LightGBM 7 |
| Time-series summary rows | 72 |
| Time-series MetaGB common cells | 9 |

跨协议总计 80/9/74 仅用于归档，不应作为论文主结论。它混合了 single-seed synthetic、完整随机搜索、fixed-backbone tail protocol 和不同任务指标。

| 数据族 | 协议 | 预处理 | 任务 | Cells | Paired folds | Dataset M/T/X | Fold M/T/X | Median runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Other tabular | random300 | drop | binary | 10 | 50 | 3/2/5 | 20/4/26 | 14.19x |
| Other tabular | random300 | drop | multiclass | 4 | 20 | 1/0/3 | 8/0/12 | 71.85x |
| Other tabular | random300 | drop | regression | 9 | 45 | 4/0/5 | 22/1/22 | 19.83x |
| Other tabular | random300 | onehot | binary | 10 | 50 | 3/2/5 | 21/4/25 | 27.05x |
| Other tabular | random300 | onehot | multiclass | 4 | 20 | 1/0/3 | 8/0/12 | 94.27x |
| Other tabular | random300 | onehot | regression | 9 | 45 | 4/0/5 | 23/1/21 | 20.00x |
| Synthetic main | random300_single_seed | synthetic | binary | 6 | 30 | 6/0/0 | 26/1/3 | 46.40x |
| Synthetic main | random300_single_seed | synthetic | regression | 32 | 160 | 25/1/6 | 129/0/31 | 16.39x |
| TabArena30 | xgb_best_mgb30 | drop | binary | 10 | 50 | 2/2/6 | 17/6/27 | 6.37x |
| TabArena30 | xgb_best_mgb30 | onehot | binary | 8 | 40 | 3/1/4 | 20/1/19 | 2.02x |
| TabArena30 | xgb_best_mgb30 | onehot | multiclass | 3 | 15 | 3/0/0 | 12/0/3 | 4.59x |
| TabArena main | random300 | drop | binary | 17 | 85 | 8/0/9 | 40/1/44 | 22.01x |
| TabArena main | random300 | drop | multiclass | 4 | 20 | 2/1/1 | 13/1/6 | 26.23x |
| TabArena main | random300 | onehot | binary | 13 | 65 | 5/0/8 | 35/1/29 | 58.98x |
| TabArena main | random300 | onehot | multiclass | 3 | 15 | 2/0/1 | 12/0/3 | 21.48x |
| TabArena main | random600 | drop | regression | 11 | 55 | 3/0/8 | 18/4/33 | 10.30x |
| TabArena main | random600 | onehot | regression | 10 | 50 | 5/0/5 | 27/2/21 | 13.81x |

<a id="real-mgb-xgb"></a>

## 4. 真实数据：MetaGB 与 XGBoost

本节保留 125 个真实 dataset/preprocessing cells：TabArena main 58、TabArena30 21、Other tabular 46。TabArena30 使用 XGB-best backbone 加 MetaGB-specific 30-grid，不能与完整 Random300/Random600 搜索合并为同预算结果。

### 4.1 Other tabular 全部 cell

| 协议 | Prep. | 任务 | Dataset | Metric | XGB | MetaGB | Delta | W | Runtime | Folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random300 | drop | binary | banknote-authentication.csv | auc | 0.999989 | 0.999968 | -0.000022 | T | 10.63x | 5 |
| random300 | drop | binary | breast-w.csv | auc | 0.994144 | 0.994137 | -0.000006 | T | 37.14x | 5 |
| random300 | drop | binary | credit-approval.csv | auc | 0.865395 | 0.858539 | -0.006857 | X | 16.37x | 5 |
| random300 | drop | binary | electricity.csv | auc | 0.965548 | 0.965284 | -0.000264 | X | 11.99x | 5 |
| random300 | drop | binary | haberman.csv | auc | 0.674657 | 0.658448 | -0.016209 | X | 5.69x | 5 |
| random300 | drop | binary | ilpd.csv | auc | 0.739726 | 0.759708 | +0.019982 | M | 31.93x | 5 |
| random300 | drop | binary | kc1.csv | auc | 0.814055 | 0.822601 | +0.008546 | M | 73.39x | 5 |
| random300 | drop | binary | mozilla4.csv | auc | 0.982735 | 0.981277 | -0.001458 | X | 11.71x | 5 |
| random300 | drop | binary | pc1.csv | auc | 0.847202 | 0.843418 | -0.003784 | X | 95.66x | 5 |
| random300 | drop | binary | phoneme.csv | auc | 0.959221 | 0.965649 | +0.006428 | M | 9.56x | 5 |
| random300 | drop | multiclass | balance-scale.csv | logloss | 0.2338 | 0.3429 | -46.7% | X | 14.03x | 5 |
| random300 | drop | multiclass | segment.csv | logloss | 0.0471 | 0.0550 | -16.7% | X | 123.76x | 5 |
| random300 | drop | multiclass | vehicle.csv | logloss | 0.4461 | 0.4481 | -0.4% | X | 107.83x | 5 |
| random300 | drop | multiclass | vowel.csv | logloss | 0.2282 | 0.1320 | +42.1% | M | 37.15x | 5 |
| random300 | drop | regression | Moneyball.csv | rmse | 24.0159 | 22.9450 | +4.5% | M | 32.24x | 5 |
| random300 | drop | regression | abalone.csv | rmse | 2.1594 | 2.1441 | +0.7% | M | 8.82x | 5 |
| random300 | drop | regression | boston.csv | rmse | 3.0492 | 3.4493 | -13.1% | X | 20.47x | 5 |
| random300 | drop | regression | cpu_activity.csv | rmse | 2.1868 | 3.9344 | -79.9% | X | 19.99x | 5 |
| random300 | drop | regression | energy_efficiency.csv | rmse | 0.3461 | 0.3574 | -3.3% | X | 32.11x | 5 |
| random300 | drop | regression | friedman1.csv | rmse | 1.2385 | 1.2467 | -0.7% | X | 8.37x | 5 |
| random300 | drop | regression | kin8nm.csv | rmse | 0.1125 | 0.1003 | +10.9% | M | 10.09x | 5 |
| random300 | drop | regression | naval_propulsion_plant.csv | rmse | 0.000860 | 0.000804 | +6.5% | M | 25.02x | 5 |
| random300 | drop | regression | space_ga.csv | rmse | 0.1084 | 0.1101 | -1.6% | X | 6.32x | 5 |
| random300 | onehot | binary | banknote-authentication.csv | auc | 0.999989 | 0.999968 | -0.000022 | T | 10.13x | 5 |
| random300 | onehot | binary | breast-w.csv | auc | 0.994144 | 0.994137 | -0.000006 | T | 43.93x | 5 |
| random300 | onehot | binary | credit-approval.csv | auc | 0.939312 | 0.930379 | -0.008933 | X | 564.94x | 5 |
| random300 | onehot | binary | electricity.csv | auc | 0.965548 | 0.965284 | -0.000264 | X | 11.88x | 5 |
| random300 | onehot | binary | haberman.csv | auc | 0.674657 | 0.658448 | -0.016209 | X | 5.76x | 5 |
| random300 | onehot | binary | ilpd.csv | auc | 0.751779 | 0.769598 | +0.017819 | M | 52.19x | 5 |
| random300 | onehot | binary | kc1.csv | auc | 0.814055 | 0.822601 | +0.008546 | M | 77.16x | 5 |
| random300 | onehot | binary | mozilla4.csv | auc | 0.982735 | 0.981277 | -0.001458 | X | 12.07x | 5 |
| random300 | onehot | binary | pc1.csv | auc | 0.847202 | 0.843418 | -0.003784 | X | 103.43x | 5 |
| random300 | onehot | binary | phoneme.csv | auc | 0.959221 | 0.965649 | +0.006428 | M | 10.22x | 5 |
| random300 | onehot | multiclass | balance-scale.csv | logloss | 0.2338 | 0.3429 | -46.7% | X | 13.14x | 5 |
| random300 | onehot | multiclass | segment.csv | logloss | 0.0471 | 0.0550 | -16.7% | X | 129.05x | 5 |
| random300 | onehot | multiclass | vehicle.csv | logloss | 0.4461 | 0.4481 | -0.4% | X | 147.40x | 5 |
| random300 | onehot | multiclass | vowel.csv | logloss | 0.2251 | 0.1271 | +43.5% | M | 60.72x | 5 |
| random300 | onehot | regression | Moneyball.csv | rmse | 23.3384 | 22.2620 | +4.6% | M | 292.07x | 5 |
| random300 | onehot | regression | abalone.csv | rmse | 2.1462 | 2.1269 | +0.9% | M | 19.60x | 5 |
| random300 | onehot | regression | boston.csv | rmse | 3.0492 | 3.4493 | -13.1% | X | 24.99x | 5 |
| random300 | onehot | regression | cpu_activity.csv | rmse | 2.1868 | 3.9344 | -79.9% | X | 18.72x | 5 |
| random300 | onehot | regression | energy_efficiency.csv | rmse | 0.3461 | 0.3574 | -3.3% | X | 40.57x | 5 |
| random300 | onehot | regression | friedman1.csv | rmse | 1.2385 | 1.2467 | -0.7% | X | 10.21x | 5 |
| random300 | onehot | regression | kin8nm.csv | rmse | 0.1125 | 0.1003 | +10.9% | M | 9.56x | 5 |
| random300 | onehot | regression | naval_propulsion_plant.csv | rmse | 0.000860 | 0.000804 | +6.5% | M | 27.28x | 5 |
| random300 | onehot | regression | space_ga.csv | rmse | 0.1084 | 0.1101 | -1.6% | X | 7.85x | 5 |

### 4.2 TabArena main 全部 cell

| 协议 | Prep. | 任务 | Dataset | Metric | XGB | MetaGB | Delta | W | Runtime | Folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random300 | drop | binary | Amazon_employee_access.csv | auc | 0.859789 | 0.858608 | -0.001180 | X | 16.69x | 5 |
| random300 | drop | binary | Bank_Customer_Churn.csv | auc | 0.850385 | 0.848853 | -0.001532 | X | 20.34x | 5 |
| random300 | drop | binary | Diabetes130US.csv | auc | 0.599449 | 0.598479 | -0.000970 | X | 18.86x | 5 |
| random300 | drop | binary | E-CommereShippingData.csv | auc | 0.744538 | 0.745572 | +0.001034 | M | 14.18x | 5 |
| random300 | drop | binary | Fitness_Club.csv | auc | 0.810147 | 0.811218 | +0.001071 | M | 6.06x | 5 |
| random300 | drop | binary | GiveMeSomeCredit.csv | auc | 0.856136 | 0.855610 | -0.000526 | X | 18.16x | 5 |
| random300 | drop | binary | HR_Analytics_Job_Change_of_Data_Scientists.csv | auc | 0.724459 | 0.725224 | +0.000765 | M | 6.97x | 5 |
| random300 | drop | binary | Is-this-a-good-customer.csv | auc | 0.626316 | 0.623842 | -0.002475 | X | 40.98x | 5 |
| random300 | drop | binary | Marketing_Campaign.csv | auc | 0.873715 | 0.877472 | +0.003757 | M | 129.63x | 5 |
| random300 | drop | binary | blood-transfusion-service-center.csv | auc | 0.717556 | 0.720450 | +0.002895 | M | 7.57x | 5 |
| random300 | drop | binary | churn.csv | auc | 0.922337 | 0.922750 | +0.000413 | M | 57.33x | 5 |
| random300 | drop | binary | credit-g.csv | auc | 0.675262 | 0.684024 | +0.008762 | M | 21.40x | 5 |
| random300 | drop | binary | diabetes.csv | auc | 0.828799 | 0.827769 | -0.001030 | X | 22.95x | 5 |
| random300 | drop | binary | hazelnut-spread-contaminant-detection.csv | auc | 0.979490 | 0.983219 | +0.003729 | M | 55.55x | 5 |
| random300 | drop | binary | polish_companies_bankruptcy.csv | auc | 0.919304 | 0.831711 | -0.087593 | X | 124.94x | 5 |
| random300 | drop | binary | qsar-biodeg.csv | auc | 0.932492 | 0.931635 | -0.000857 | X | 225.45x | 5 |
| random300 | drop | binary | seismic-bumps.csv | auc | 0.781116 | 0.761576 | -0.019540 | X | 38.90x | 5 |
| random300 | drop | multiclass | SDSS17.csv | logloss | 0.0726 | 0.0728 | -0.2% | X | 21.80x | 5 |
| random300 | drop | multiclass | anneal.csv | logloss | 0.2266 | 0.2179 | +3.8% | M | 39.37x | 5 |
| random300 | drop | multiclass | maternal_health_risk.csv | logloss | 0.4004 | 0.3934 | +1.8% | M | 19.94x | 5 |
| random300 | drop | multiclass | students_dropout_and_academic_success.csv | logloss | 0.6029 | 0.6030 | -0.0% | T | 62.20x | 5 |
| random300 | onehot | binary | Amazon_employee_access.csv | auc | 0.859789 | 0.858608 | -0.001180 | X | 13.25x | 5 |
| random300 | onehot | binary | Bank_Customer_Churn.csv | auc | 0.868422 | 0.870679 | +0.002258 | M | 30.32x | 5 |
| random300 | onehot | binary | E-CommereShippingData.csv | auc | 0.745365 | 0.744980 | -0.000385 | X | 61.84x | 5 |
| random300 | onehot | binary | Fitness_Club.csv | auc | 0.810823 | 0.808593 | -0.002230 | X | 88.88x | 5 |
| random300 | onehot | binary | GiveMeSomeCredit.csv | auc | 0.856136 | 0.855610 | -0.000526 | X | 17.64x | 5 |
| random300 | onehot | binary | Is-this-a-good-customer.csv | auc | 0.734754 | 0.728662 | -0.006092 | X | 503.65x | 5 |
| random300 | onehot | binary | blood-transfusion-service-center.csv | auc | 0.717556 | 0.720450 | +0.002895 | M | 7.83x | 5 |
| random300 | onehot | binary | churn.csv | auc | 0.922337 | 0.922750 | +0.000413 | M | 57.11x | 5 |
| random300 | onehot | binary | credit-g.csv | auc | 0.787786 | 0.793047 | +0.005262 | M | 1111.97x | 5 |
| random300 | onehot | binary | diabetes.csv | auc | 0.828799 | 0.827769 | -0.001030 | X | 23.17x | 5 |
| random300 | onehot | binary | hazelnut-spread-contaminant-detection.csv | auc | 0.979490 | 0.983219 | +0.003729 | M | 56.60x | 5 |
| random300 | onehot | binary | qsar-biodeg.csv | auc | 0.932492 | 0.931635 | -0.000857 | X | 245.06x | 5 |
| random300 | onehot | binary | seismic-bumps.csv | auc | 0.783599 | 0.775025 | -0.008574 | X | 115.00x | 5 |
| random300 | onehot | multiclass | SDSS17.csv | logloss | 0.0726 | 0.0728 | -0.2% | X | 20.81x | 5 |
| random300 | onehot | multiclass | maternal_health_risk.csv | logloss | 0.4004 | 0.3934 | +1.8% | M | 21.40x | 5 |
| random300 | onehot | multiclass | website_phishing.csv | logloss | 0.2455 | 0.2392 | +2.6% | M | 246.09x | 5 |
| random600 | drop | regression | Another-Dataset-on-used-Fiat-500.csv | rmse | 750.6218 | 761.0658 | -1.4% | X | 9.09x | 5 |
| random600 | drop | regression | Food_Delivery_Time.csv | rmse | 7.5250 | 7.5350 | -0.1% | X | 10.53x | 5 |
| random600 | drop | regression | QSAR_fish_toxicity.csv | rmse | 0.9029 | 0.8911 | +1.3% | M | 9.47x | 5 |
| random600 | drop | regression | airfoil_self_noise.csv | rmse | 1.4066 | 1.5209 | -8.1% | X | 13.86x | 5 |
| random600 | drop | regression | concrete_compressive_strength.csv | rmse | 3.8967 | 3.9069 | -0.3% | X | 11.94x | 5 |
| random600 | drop | regression | diamonds.csv | rmse | 1338.2900 | 1355.7340 | -1.3% | X | 9.97x | 5 |
| random600 | drop | regression | healthcare_insurance_expenses.csv | rmse | 11508.0340 | 11608.5000 | -0.9% | X | 6.34x | 5 |
| random600 | drop | regression | houses.csv | rmse | 0.2127 | 0.2560 | -20.4% | X | 7.97x | 5 |
| random600 | drop | regression | miami_housing.csv | rmse | 83836.7400 | 83450.9400 | +0.5% | M | 13.96x | 5 |
| random600 | drop | regression | physiochemical_protein.csv | rmse | 3.4355 | 3.4109 | +0.7% | M | 9.70x | 5 |
| random600 | drop | regression | wine_quality.csv | rmse | 0.5880 | 0.5931 | -0.9% | X | 16.82x | 5 |
| random600 | onehot | regression | Another-Dataset-on-used-Fiat-500.csv | rmse | 740.7772 | 748.8338 | -1.1% | X | 17.63x | 5 |
| random600 | onehot | regression | QSAR_fish_toxicity.csv | rmse | 0.9029 | 0.8911 | +1.3% | M | 8.77x | 5 |
| random600 | onehot | regression | airfoil_self_noise.csv | rmse | 1.4066 | 1.5209 | -8.1% | X | 14.66x | 5 |
| random600 | onehot | regression | concrete_compressive_strength.csv | rmse | 3.8967 | 3.9069 | -0.3% | X | 11.60x | 5 |
| random600 | onehot | regression | diamonds.csv | rmse | 530.3432 | 533.0934 | -0.5% | X | 83.81x | 5 |
| random600 | onehot | regression | healthcare_insurance_expenses.csv | rmse | 4546.9380 | 4493.9820 | +1.2% | M | 38.56x | 5 |
| random600 | onehot | regression | houses.csv | rmse | 0.2127 | 0.2560 | -20.4% | X | 8.04x | 5 |
| random600 | onehot | regression | miami_housing.csv | rmse | 83836.7400 | 83450.9400 | +0.5% | M | 12.56x | 5 |
| random600 | onehot | regression | physiochemical_protein.csv | rmse | 3.4355 | 3.4109 | +0.7% | M | 9.72x | 5 |
| random600 | onehot | regression | wine_quality.csv | rmse | 0.5902 | 0.5889 | +0.2% | M | 18.04x | 5 |

### 4.3 TabArena30 全部 cell

| 协议 | Prep. | 任务 | Dataset | Metric | XGB | MetaGB | Delta | W | Runtime | Folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_best_mgb30 | drop | binary | APSFailure.csv | auc | 0.989820 | 0.967346 | -0.022474 | X | 44.08x | 5 |
| xgb_best_mgb30 | drop | binary | NATICUSdroid.csv | auc | 0.985636 | 0.985634 | -0.000002 | T | 37.42x | 5 |
| xgb_best_mgb30 | drop | binary | bank-marketing.csv | auc | 0.707328 | 0.706994 | -0.000333 | X | 0.50x | 5 |
| xgb_best_mgb30 | drop | binary | coil2000_insurance_policies.csv | auc | 0.762398 | 0.761193 | -0.001205 | X | 12.70x | 5 |
| xgb_best_mgb30 | drop | binary | credit_card_clients_default.csv | auc | 0.785290 | 0.782138 | -0.003152 | X | 2.32x | 5 |
| xgb_best_mgb30 | drop | binary | customer_satisfaction_in_airline.csv | auc | 0.991895 | 0.991958 | +0.000064 | T | 21.44x | 5 |
| xgb_best_mgb30 | drop | binary | heloc.csv | auc | 0.799162 | 0.798763 | -0.000399 | X | 1.48x | 5 |
| xgb_best_mgb30 | drop | binary | jm1.csv | auc | 0.752656 | 0.755191 | +0.002535 | M | 5.88x | 5 |
| xgb_best_mgb30 | drop | binary | online_shoppers_intention.csv | auc | 0.915138 | 0.915564 | +0.000426 | M | 1.06x | 5 |
| xgb_best_mgb30 | drop | binary | taiwanese_bankruptcy_prediction.csv | auc | 0.947862 | 0.919703 | -0.028158 | X | 19.02x | 5 |
| xgb_best_mgb30 | onehot | binary | Fitness_Club.csv | auc | 0.810823 | 0.810722 | -0.000101 | X | 0.80x | 5 |
| xgb_best_mgb30 | onehot | binary | Is-this-a-good-customer.csv | auc | 0.734754 | 0.733887 | -0.000867 | X | 5.36x | 5 |
| xgb_best_mgb30 | onehot | binary | blood-transfusion-service-center.csv | auc | 0.717556 | 0.698976 | -0.018580 | X | 0.40x | 5 |
| xgb_best_mgb30 | onehot | binary | credit-g.csv | auc | 0.787786 | 0.795429 | +0.007643 | M | 33.80x | 5 |
| xgb_best_mgb30 | onehot | binary | diabetes.csv | auc | 0.828799 | 0.828854 | +0.000055 | T | 0.41x | 5 |
| xgb_best_mgb30 | onehot | binary | hazelnut-spread-contaminant-detection.csv | auc | 0.979490 | 0.981687 | +0.002198 | M | 2.10x | 5 |
| xgb_best_mgb30 | onehot | binary | qsar-biodeg.csv | auc | 0.932492 | 0.933143 | +0.000651 | M | 4.84x | 5 |
| xgb_best_mgb30 | onehot | binary | seismic-bumps.csv | auc | 0.783599 | 0.780425 | -0.003174 | X | 0.68x | 5 |
| xgb_best_mgb30 | onehot | multiclass | anneal.csv | logloss | 0.0266 | 0.0262 | +1.5% | M | 99.75x | 5 |
| xgb_best_mgb30 | onehot | multiclass | maternal_health_risk.csv | logloss | 0.4004 | 0.3766 | +5.9% | M | 3.61x | 5 |
| xgb_best_mgb30 | onehot | multiclass | website_phishing.csv | logloss | 0.2455 | 0.2387 | +2.8% | M | 4.27x | 5 |

### 4.4 Drop/One-hot 覆盖边界

| 范围 | 任务 | Drop | One-hot | 差异 | 说明 |
| --- | --- | --- | --- | --- | --- |
| Other tabular Random300 | binary | 10 | 10 | 0 | 两种预处理 paired 集合一致 |
| Other tabular Random300 | multiclass | 4 | 4 | 0 | raw coverage 仍有 XGB-only cells |
| Other tabular Random300 | regression | 9 | 9 | 0 | 两种预处理 paired 集合一致 |
| TabArena main Random300 | binary | 17 | 13 | 4 | 缺 Diabetes130US、HR Analytics、Marketing Campaign、polish companies |
| TabArena main Random300 | multiclass | 4 | 3 | 2 (+1 only) | 缺 anneal、students dropout；onehot-only 为 website phishing |
| TabArena main Random600 | regression | 11 | 10 | 1 | 缺 Food Delivery Time |
| TabArena30 | binary | 10 | 8 | 集合不重合 | paired completion 差异，不能作为同集合 drop/onehot 对照 |

<a id="strong-tabular-baselines"></a>

## 5. 强 Tabular Baselines

0716 补全后的保留 baseline 为 Linear、ExtraTrees、RandomForest、LightGBM、CatBoost、TorchMLP 和 RealMLP。TabM 已从正式模型集合移除。Canonical union 使用新日志覆盖同 task id 的旧日志，并对缺失的少量旧日志回退。

| 模型 | 完成 folds | 完整 cells | 状态 |
| --- | --- | --- | --- |
| Linear | 270 | 54/54 | 完整 |
| ExtraTrees | 270 | 54/54 | 完整 |
| RandomForest | 270 | 54/54 | 完整 |
| LightGBM | 270 | 54/54 | 完整 |
| CatBoost | 270 | 54/54 | 完整 |
| TorchMLP | 270 | 54/54 | 完整 |
| RealMLP | 265 | 53/54 | electricity-onehot 缺 5 folds |

### 5.1 Baseline 模型角色与 54-cell 原始均值

| 模型 | 类别 | 在本实验中的作用 |
| --- | --- | --- |
| Linear | 全局线性模型 | 非深度学习；检验简单线性结构 |
| ExtraTrees | 随机化树集成 | 随机切分与 bagging 类参考 |
| RandomForest | 随机森林 | bagging 树集成参考 |
| LightGBM | 梯度提升树 | 高效 histogram boosting |
| CatBoost | 梯度提升树 | 强 gradient-boosting baseline |
| TorchMLP | 多层感知机 | 标准神经网络 baseline |
| RealMLP | 强化 MLP | 更强 tabular neural baseline；缺一个 cell |

| Prep. | 任务 | Dataset | Metric | Linear | Extra | RF | LGB | Cat | TMLP | RMLP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop | binary | banknote-authentication.csv | auc | 0.999796 | 1.000000 | 0.999892 | 0.999957 | 0.999925 | 1.000000 | 1.000000 |
| drop | binary | breast-w.csv | auc | 0.995137 | 0.993813 | 0.991780 | 0.994850 | 0.994280 | 0.994469 | 0.992729 |
| drop | binary | credit-approval.csv | auc | 0.825980 | 0.845751 | 0.855017 | 0.858974 | 0.857599 | 0.831634 | 0.832028 |
| drop | binary | electricity.csv | auc | 0.818990 | 0.948857 | 0.967938 | 0.963256 | 0.960520 | 0.906102 | 0.900345 |
| drop | binary | haberman.csv | auc | 0.692124 | 0.714052 | 0.693464 | 0.674355 | 0.722059 | 0.710653 | 0.687696 |
| drop | binary | ilpd.csv | auc | 0.754561 | 0.753177 | 0.740793 | 0.724986 | 0.759327 | 0.729136 | 0.719108 |
| drop | binary | kc1.csv | auc | 0.798718 | 0.835025 | 0.821807 | 0.813233 | 0.829278 | 0.799285 | 0.809197 |
| drop | binary | mozilla4.csv | auc | 0.886889 | 0.982009 | 0.982587 | 0.982382 | 0.980526 | 0.959220 | 0.958441 |
| drop | binary | pc1.csv | auc | 0.838340 | 0.865346 | 0.848927 | 0.848279 | 0.860437 | 0.849203 | 0.797734 |
| drop | binary | phoneme.csv | auc | 0.811552 | 0.969223 | 0.962189 | 0.957205 | 0.965313 | 0.946672 | 0.939594 |
| drop | multiclass | balance-scale.csv | logloss | 0.2630 | 0.4570 | 0.4247 | 0.2756 | 0.2269 | 0.0639 | 0.2415 |
| drop | multiclass | mfeat-fourier.csv | logloss | 0.4918 | 0.4451 | 0.5179 | 0.4151 | 0.3952 | 0.4528 | 0.4782 |
| drop | multiclass | mfeat-karhunen.csv | logloss | 0.2022 | 0.3168 | 0.3498 | 0.1305 | 0.0996 | 0.1387 | 0.2014 |
| drop | multiclass | optdigits.csv | logloss | 0.0943 | 0.1384 | 0.1705 | 0.0578 | 0.0473 | 0.0526 | 0.1315 |
| drop | multiclass | satimage.csv | logloss | 0.3569 | 0.2340 | 0.2605 | 0.2197 | 0.2140 | 0.2308 | 0.2843 |
| drop | multiclass | segment.csv | logloss | 0.1537 | 0.0643 | 0.0746 | 0.0531 | 0.0504 | 0.1019 | 0.1490 |
| drop | multiclass | vehicle.csv | logloss | 0.4058 | 0.4769 | 0.4983 | 0.4634 | 0.4599 | 0.3533 | 0.4225 |
| drop | multiclass | vowel.csv | logloss | 0.9150 | 0.2666 | 0.3792 | 0.1883 | 0.0559 | 0.0694 | 0.3214 |
| drop | regression | Moneyball.csv | rmse | 21.1835 | 23.8059 | 24.2998 | 28.7378 | 23.4692 | 117.0092 | 23.6589 |
| drop | regression | abalone.csv | rmse | 2.2359 | 2.1539 | 2.1459 | 2.1636 | 2.1711 | 2.1048 | 2.0909 |
| drop | regression | boston.csv | rmse | 4.8440 | 3.0451 | 3.3535 | 3.2395 | 2.9205 | 3.3215 | 3.7506 |
| drop | regression | cpu_activity.csv | rmse | 9.6882 | 2.2890 | 2.4425 | 2.2600 | 2.2212 | 2.5900 | 2.3163 |
| drop | regression | energy_efficiency.csv | rmse | 2.9410 | 0.4582 | 0.4677 | 0.3846 | 0.3280 | 0.6104 | 1.1406 |
| drop | regression | friedman1.csv | rmse | 2.5858 | 1.7456 | 1.8279 | 1.2682 | 1.1181 | 1.2198 | 1.0607 |
| drop | regression | kin8nm.csv | rmse | 0.2019 | 0.1338 | 0.1402 | 0.1090 | 0.0836 | 0.0712 | 0.0649 |
| drop | regression | naval_propulsion_plant.csv | rmse | 0.005846 | 0.000621 | 0.000899 | 0.000866 | 0.000832 | 0.001923 | 0.000260 |
| drop | regression | space_ga.csv | rmse | 0.1289 | 0.1177 | 0.1172 | 0.1055 | 0.1083 | 0.1004 | 0.1056 |
| onehot | binary | banknote-authentication.csv | auc | 0.999796 | 0.999968 | 0.999892 | 0.999957 | 0.999925 | 1.000000 | 1.000000 |
| onehot | binary | breast-w.csv | auc | 0.995137 | 0.993813 | 0.991780 | 0.994850 | 0.994280 | 0.994469 | 0.992729 |
| onehot | binary | credit-approval.csv | auc | 0.918178 | 0.923914 | 0.935024 | 0.939564 | 0.940689 | 0.924013 | 0.918418 |
| onehot | binary | electricity.csv | auc | 0.818990 | 0.948857 | 0.967938 | 0.963256 | 0.960520 | 0.906102 | -- |
| onehot | binary | haberman.csv | auc | 0.692124 | 0.714052 | 0.693464 | 0.674355 | 0.722059 | 0.710653 | 0.687696 |
| onehot | binary | ilpd.csv | auc | 0.753204 | 0.767457 | 0.743638 | 0.727351 | 0.757253 | 0.729242 | 0.739979 |
| onehot | binary | kc1.csv | auc | 0.798718 | 0.835025 | 0.821807 | 0.813233 | 0.829278 | 0.799285 | 0.809197 |
| onehot | binary | mozilla4.csv | auc | 0.886889 | 0.982009 | 0.982587 | 0.982382 | 0.980526 | 0.959220 | 0.958441 |
| onehot | binary | pc1.csv | auc | 0.838340 | 0.865346 | 0.848927 | 0.848279 | 0.860437 | 0.849203 | 0.797734 |
| onehot | binary | phoneme.csv | auc | 0.811552 | 0.969223 | 0.962189 | 0.957205 | 0.965313 | 0.946672 | 0.939594 |
| onehot | multiclass | balance-scale.csv | logloss | 0.2630 | 0.4570 | 0.4247 | 0.2756 | 0.2269 | 0.0639 | 0.2415 |
| onehot | multiclass | mfeat-fourier.csv | logloss | 0.4918 | 0.4451 | 0.5179 | 0.4151 | 0.3952 | 0.4528 | 0.4782 |
| onehot | multiclass | mfeat-karhunen.csv | logloss | 0.2022 | 0.3168 | 0.3498 | 0.1305 | 0.0996 | 0.1387 | 0.2014 |
| onehot | multiclass | optdigits.csv | logloss | 0.0943 | 0.1384 | 0.1705 | 0.0578 | 0.0473 | 0.0526 | 0.1315 |
| onehot | multiclass | satimage.csv | logloss | 0.3569 | 0.2340 | 0.2605 | 0.2197 | 0.2140 | 0.2308 | 0.2843 |
| onehot | multiclass | segment.csv | logloss | 0.1537 | 0.0643 | 0.0746 | 0.0531 | 0.0504 | 0.1019 | 0.1490 |
| onehot | multiclass | vehicle.csv | logloss | 0.4058 | 0.4769 | 0.4983 | 0.4634 | 0.4599 | 0.3533 | 0.4225 |
| onehot | multiclass | vowel.csv | logloss | 0.9151 | 0.2561 | 0.3720 | 0.2025 | 0.0514 | 0.0608 | 0.3131 |
| onehot | regression | Moneyball.csv | rmse | 21.7632 | 23.6816 | 24.5614 | 27.6924 | 23.3535 | 108.0258 | 29.3321 |
| onehot | regression | abalone.csv | rmse | 2.2113 | 2.1552 | 2.1331 | 2.1454 | 2.1647 | 2.1262 | 2.0934 |
| onehot | regression | boston.csv | rmse | 4.8440 | 3.0451 | 3.3535 | 3.2395 | 2.9205 | 3.3215 | 3.7506 |
| onehot | regression | cpu_activity.csv | rmse | 9.6882 | 2.2890 | 2.4425 | 2.2600 | 2.2212 | 2.5900 | 2.3163 |
| onehot | regression | energy_efficiency.csv | rmse | 2.9410 | 0.4582 | 0.4677 | 0.3846 | 0.3280 | 0.6104 | 1.1406 |
| onehot | regression | friedman1.csv | rmse | 2.5858 | 1.7456 | 1.8279 | 1.2682 | 1.1181 | 1.2198 | 1.0607 |
| onehot | regression | kin8nm.csv | rmse | 0.2019 | 0.1338 | 0.1402 | 0.1090 | 0.0836 | 0.0712 | 0.0649 |
| onehot | regression | naval_propulsion_plant.csv | rmse | 0.005846 | 0.000621 | 0.000899 | 0.000866 | 0.000832 | 0.001923 | 0.000260 |
| onehot | regression | space_ga.csv | rmse | 0.1289 | 0.1177 | 0.1172 | 0.1055 | 0.1083 | 0.1004 | 0.1056 |

### 5.2 七个 baseline 自身的 54-cell 排名

该排名仅比较 baseline 模型，并允许 RealMLP 在 electricity-onehot 缺席，因此每个模型的参与 cells 数不同。它适合描述 baseline family 的整体强度，不应与严格共同覆盖排名混用。

| 模型 | 平均 rank | 参与 cells | Fractional firsts |
| --- | --- | --- | --- |
| CatBoost | 2.259 | 54 | 22.000 |
| LightGBM | 3.444 | 54 | 1.000 |
| ExtraTrees | 3.667 | 54 | 7.333 |
| TorchMLP | 3.676 | 54 | 6.833 |
| RealMLP | 4.406 | 53 | 8.833 |
| RandomForest | 4.759 | 54 | 4.000 |
| Linear | 5.741 | 54 | 4.000 |

### 5.3 MetaGB 与每个实际模型的逐模型比较

| 对手 | Common cells | MetaGB 胜 | Tie | MetaGB 负 | MetaGB 胜率 |
| --- | --- | --- | --- | --- | --- |
| Linear | 46 | 36 | 0 | 10 | 78.3% |
| RandomForest | 46 | 30 | 3 | 13 | 65.2% |
| ExtraTrees | 46 | 28 | 2 | 16 | 60.9% |
| RealMLP | 45 | 25 | 2 | 18 | 55.6% |
| LightGBM | 46 | 24 | 2 | 20 | 52.2% |
| TorchMLP | 46 | 20 | 3 | 23 | 43.5% |
| CatBoost | 46 | 17 | 2 | 27 | 37.0% |
| XGBoost | 46 | 16 | 4 | 26 | 34.8% |

逐 cell strongest available baseline 是 oracle envelope，不是可训练模型。最新 46 个 aligned cells 上，envelope 对 MetaGB 为 42/2/2；MetaGB 的两个明确胜利均来自 ILPD 的 drop 和 one-hot cells。

### 5.4 当前论文八模型口径（46 cells）

当前 main.tex 的 manuscript-facing 排名排除 RealMLP，使用 MetaGB、XGBoost 和六个完整 baseline，共 46 个共同 cells。该口径中 MetaGB overall average rank 为 4.12，列第三，但只取得两个 first credits。

| 模型 | All | Binary | Multiclass | Regression | First credits |
| --- | --- | --- | --- | --- | --- |
| CatBoost | 2.93 | 3.40 | 2.50 | 2.61 | 11.00 |
| XGBoost | 3.61 | 4.00 | 3.00 | 3.44 | 7.00 |
| MetaGB (ours) | 4.12 | 4.28 | 4.25 | 3.89 | 2.00 |
| ExtraTrees | 4.37 | 3.40 | 6.50 | 4.50 | 8.50 |
| MLP | 4.38 | 5.17 | 2.75 | 4.22 | 11.50 |
| LightGBM | 4.57 | 4.80 | 4.50 | 4.33 | 0.00 |
| RandomForest | 5.39 | 4.40 | 7.00 | 5.78 | 2.00 |
| Linear | 6.63 | 6.55 | 5.50 | 7.22 | 4.00 |

### 5.5 含 RealMLP 的严格九模型口径（45 cells）

若把 RealMLP 纳入，则必须删除其未完成的 electricity-onehot，剩余 45 个九模型共同 cells。MetaGB 的平均 rank 为 4.59，仍列第三；其每类任务的组内位置均为第四。

| 模型 | 平均 rank | 中位 rank | Fractional firsts |
| --- | --- | --- | --- |
| CatBoost | 3.20 | 3.00 | 9.00 |
| XGBoost | 4.02 | 5.00 | 7.00 |
| MetaGB | 4.59 | 4.50 | 2.00 |
| ExtraTrees | 4.70 | 5.00 | 6.33 |
| TorchMLP | 4.70 | 5.00 | 6.83 |
| LightGBM | 4.98 | 5.00 | 0.00 |
| RealMLP | 5.48 | 7.00 | 8.83 |
| RandomForest | 6.00 | 6.00 | 1.00 |
| Linear | 7.33 | 9.00 | 4.00 |

| Prep. | 任务 | Dataset | MGB | XGB | Linear | Extra | RF | LGB | Cat | TMLP | RMLP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop | binary | banknote-authentication.csv | 5.0 | 4.0 | 9.0 | 2.0 | 8.0 | 6.0 | 7.0 | 2.0 | 2.0 |
| drop | binary | breast-w.csv | 6.0 | 5.0 | 1.0 | 7.0 | 9.0 | 2.0 | 4.0 | 3.0 | 8.0 |
| drop | binary | credit-approval.csv | 3.0 | 1.0 | 9.0 | 6.0 | 5.0 | 2.0 | 4.0 | 8.0 | 7.0 |
| drop | binary | electricity.csv | 3.0 | 2.0 | 9.0 | 6.0 | 1.0 | 4.0 | 5.0 | 7.0 | 8.0 |
| drop | binary | haberman.csv | 9.0 | 7.0 | 5.0 | 2.0 | 4.0 | 8.0 | 1.0 | 3.0 | 6.0 |
| drop | binary | ilpd.csv | 1.0 | 6.0 | 3.0 | 4.0 | 5.0 | 8.0 | 2.0 | 7.0 | 9.0 |
| drop | binary | kc1.csv | 3.0 | 5.0 | 9.0 | 1.0 | 4.0 | 6.0 | 2.0 | 8.0 | 7.0 |
| drop | binary | mozilla4.csv | 5.0 | 1.0 | 9.0 | 4.0 | 2.0 | 3.0 | 6.0 | 7.0 | 8.0 |
| drop | binary | pc1.csv | 7.0 | 6.0 | 8.0 | 1.0 | 4.0 | 5.0 | 2.0 | 3.0 | 9.0 |
| drop | binary | phoneme.csv | 2.0 | 5.0 | 9.0 | 1.0 | 4.0 | 6.0 | 3.0 | 7.0 | 8.0 |
| drop | multiclass | balance-scale.csv | 7.0 | 3.0 | 5.0 | 9.0 | 8.0 | 6.0 | 2.0 | 1.0 | 4.0 |
| drop | multiclass | segment.csv | 4.0 | 1.0 | 9.0 | 5.0 | 6.0 | 3.0 | 2.0 | 7.0 | 8.0 |
| drop | multiclass | vehicle.csv | 5.0 | 4.0 | 2.0 | 8.0 | 9.0 | 7.0 | 6.0 | 1.0 | 3.0 |
| drop | multiclass | vowel.csv | 3.0 | 5.0 | 9.0 | 6.0 | 8.0 | 4.0 | 1.0 | 2.0 | 7.0 |
| drop | regression | Moneyball.csv | 2.0 | 6.0 | 1.0 | 5.0 | 7.0 | 8.0 | 3.0 | 9.0 | 4.0 |
| drop | regression | abalone.csv | 3.0 | 6.0 | 9.0 | 5.0 | 4.0 | 7.0 | 8.0 | 2.0 | 1.0 |
| drop | regression | boston.csv | 7.0 | 3.0 | 9.0 | 2.0 | 6.0 | 4.0 | 1.0 | 5.0 | 8.0 |
| drop | regression | cpu_activity.csv | 8.0 | 1.0 | 9.0 | 4.0 | 6.0 | 3.0 | 2.0 | 7.0 | 5.0 |
| drop | regression | energy_efficiency.csv | 3.0 | 2.0 | 9.0 | 5.0 | 6.0 | 4.0 | 1.0 | 7.0 | 8.0 |
| drop | regression | friedman1.csv | 5.0 | 4.0 | 9.0 | 7.0 | 8.0 | 6.0 | 2.0 | 3.0 | 1.0 |
| drop | regression | kin8nm.csv | 4.0 | 6.0 | 9.0 | 7.0 | 8.0 | 5.0 | 3.0 | 2.0 | 1.0 |
| drop | regression | naval_propulsion_plant.csv | 3.0 | 5.0 | 9.0 | 2.0 | 7.0 | 6.0 | 4.0 | 8.0 | 1.0 |
| drop | regression | space_ga.csv | 6.0 | 5.0 | 9.0 | 8.0 | 7.0 | 2.0 | 4.0 | 1.0 | 3.0 |
| onehot | binary | banknote-authentication.csv | 4.5 | 3.0 | 9.0 | 4.5 | 8.0 | 6.0 | 7.0 | 1.5 | 1.5 |
| onehot | binary | breast-w.csv | 6.0 | 5.0 | 1.0 | 7.0 | 9.0 | 2.0 | 4.0 | 3.0 | 8.0 |
| onehot | binary | credit-approval.csv | 5.0 | 3.0 | 9.0 | 7.0 | 4.0 | 2.0 | 1.0 | 6.0 | 8.0 |
| onehot | binary | haberman.csv | 9.0 | 7.0 | 5.0 | 2.0 | 4.0 | 8.0 | 1.0 | 3.0 | 6.0 |
| onehot | binary | ilpd.csv | 1.0 | 5.0 | 4.0 | 2.0 | 6.0 | 9.0 | 3.0 | 8.0 | 7.0 |
| onehot | binary | kc1.csv | 3.0 | 5.0 | 9.0 | 1.0 | 4.0 | 6.0 | 2.0 | 8.0 | 7.0 |
| onehot | binary | mozilla4.csv | 5.0 | 1.0 | 9.0 | 4.0 | 2.0 | 3.0 | 6.0 | 7.0 | 8.0 |
| onehot | binary | pc1.csv | 7.0 | 6.0 | 8.0 | 1.0 | 4.0 | 5.0 | 2.0 | 3.0 | 9.0 |
| onehot | binary | phoneme.csv | 2.0 | 5.0 | 9.0 | 1.0 | 4.0 | 6.0 | 3.0 | 7.0 | 8.0 |
| onehot | multiclass | balance-scale.csv | 7.0 | 3.0 | 5.0 | 9.0 | 8.0 | 6.0 | 2.0 | 1.0 | 4.0 |
| onehot | multiclass | segment.csv | 4.0 | 1.0 | 9.0 | 5.0 | 6.0 | 3.0 | 2.0 | 7.0 | 8.0 |
| onehot | multiclass | vehicle.csv | 5.0 | 4.0 | 2.0 | 8.0 | 9.0 | 7.0 | 6.0 | 1.0 | 3.0 |
| onehot | multiclass | vowel.csv | 3.0 | 5.0 | 9.0 | 6.0 | 8.0 | 4.0 | 1.0 | 2.0 | 7.0 |
| onehot | regression | Moneyball.csv | 2.0 | 3.0 | 1.0 | 5.0 | 6.0 | 7.0 | 4.0 | 9.0 | 8.0 |
| onehot | regression | abalone.csv | 3.0 | 6.0 | 9.0 | 7.0 | 4.0 | 5.0 | 8.0 | 2.0 | 1.0 |
| onehot | regression | boston.csv | 7.0 | 3.0 | 9.0 | 2.0 | 6.0 | 4.0 | 1.0 | 5.0 | 8.0 |
| onehot | regression | cpu_activity.csv | 8.0 | 1.0 | 9.0 | 4.0 | 6.0 | 3.0 | 2.0 | 7.0 | 5.0 |
| onehot | regression | energy_efficiency.csv | 3.0 | 2.0 | 9.0 | 5.0 | 6.0 | 4.0 | 1.0 | 7.0 | 8.0 |
| onehot | regression | friedman1.csv | 5.0 | 4.0 | 9.0 | 7.0 | 8.0 | 6.0 | 2.0 | 3.0 | 1.0 |
| onehot | regression | kin8nm.csv | 4.0 | 6.0 | 9.0 | 7.0 | 8.0 | 5.0 | 3.0 | 2.0 | 1.0 |
| onehot | regression | naval_propulsion_plant.csv | 3.0 | 5.0 | 9.0 | 2.0 | 7.0 | 6.0 | 4.0 | 8.0 | 1.0 |
| onehot | regression | space_ga.csv | 6.0 | 5.0 | 9.0 | 8.0 | 7.0 | 2.0 | 4.0 | 1.0 | 3.0 |

<a id="synthetic-main"></a>

## 6. Synthetic 主 Benchmark

主 synthetic benchmark 使用 data seed 42，并在每个 scenario-dimension cell 上完成五个 outer folds。10/20 维共 38 cells。0716 新目录中的对应 380 个 XGBoost/MetaGB task JSON 与 0702 结果一致；LightGBM 新增 190/190 folds。

| 族 | 含义 | Cells | M/T/LGB | Median M-vs-LGB | M rank | X rank | LGB rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | All dim10/20 mechanisms | 38 | 28/0/10 | +11.5% | 1.42 | 2.21 | 2.37 |
| A | Interaction / XOR / multiplicative | 8 | 8/0/0 | +24.1% | 1.00 | 2.25 | 2.75 |
| B | Linear-in-leaves | 10 | 8/0/2 | +3.9% | 1.30 | 2.20 | 2.50 |
| C | Conditional-predictive separation | 6 | 6/0/0 | +34.8% | 1.00 | 2.50 | 2.50 |
| D | Sparse interaction | 6 | 4/0/2 | +47.6% | 1.33 | 2.50 | 2.17 |
| E | Negative controls | 6 | 0/0/6 | -6.2% | 2.83 | 1.67 | 1.50 |
| F | Noise pressure | 2 | 2/0/0 | +13.5% | 1.00 | 2.00 | 3.00 |

| 族 | Scenario | Dim | 任务 | Metric | XGB | LGB | MetaGB | Best | X rank | L rank | M rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | conditional_multiplicative | 10 | regression | rmse | 1.5504 | 1.6220 | 0.3748 | MetaGB | 2.0 | 3.0 | 1.0 |
| A | conditional_multiplicative | 20 | regression | rmse | 2.1523 | 2.5110 | 0.3551 | MetaGB | 2.0 | 3.0 | 1.0 |
| A | multiplicative | 10 | regression | rmse | 0.2614 | 0.2943 | 0.1802 | MetaGB | 2.0 | 3.0 | 1.0 |
| A | multiplicative | 20 | regression | rmse | 0.3028 | 0.3379 | 0.1804 | MetaGB | 2.0 | 3.0 | 1.0 |
| A | xor_3way | 10 | binary | auc | 0.995231 | 0.996060 | 0.997176 | MetaGB | 3.0 | 2.0 | 1.0 |
| A | xor_3way | 20 | binary | auc | 0.893339 | 0.901942 | 0.996293 | MetaGB | 3.0 | 2.0 | 1.0 |
| A | xor_classification | 10 | binary | auc | 0.997988 | 0.997547 | 0.998156 | MetaGB | 2.0 | 3.0 | 1.0 |
| A | xor_classification | 20 | binary | auc | 0.998043 | 0.997825 | 0.998310 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | feature_switch | 10 | regression | rmse | 0.2304 | 0.1689 | 0.2158 | LightGBM | 3.0 | 1.0 | 2.0 |
| B | feature_switch | 20 | regression | rmse | 0.2251 | 0.1692 | 0.2295 | LightGBM | 2.0 | 1.0 | 3.0 |
| B | global_linear | 10 | regression | rmse | 0.1603 | 0.1598 | 0.1536 | MetaGB | 3.0 | 2.0 | 1.0 |
| B | global_linear | 20 | regression | rmse | 0.1586 | 0.1602 | 0.1537 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | regime_logit | 10 | binary | auc | 0.909938 | 0.903949 | 0.919589 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | regime_logit | 20 | binary | auc | 0.898555 | 0.892202 | 0.918898 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | regime_slope | 10 | regression | rmse | 0.3399 | 0.3415 | 0.2375 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | regime_slope | 20 | regression | rmse | 0.4697 | 0.4819 | 0.2528 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | regime_slope_multi | 10 | regression | rmse | 0.3418 | 0.3703 | 0.2429 | MetaGB | 2.0 | 3.0 | 1.0 |
| B | regime_slope_multi | 20 | regression | rmse | 0.3780 | 0.4186 | 0.2258 | MetaGB | 2.0 | 3.0 | 1.0 |
| C | categorical_feature_switch | 10 | regression | rmse | 0.1767 | 0.1738 | 0.1525 | MetaGB | 3.0 | 2.0 | 1.0 |
| C | categorical_feature_switch | 20 | regression | rmse | 0.1717 | 0.1754 | 0.1548 | MetaGB | 2.0 | 3.0 | 1.0 |
| C | categorical_regime_slope | 10 | regression | rmse | 0.2448 | 0.2110 | 0.1565 | MetaGB | 3.0 | 2.0 | 1.0 |
| C | categorical_regime_slope | 20 | regression | rmse | 0.3008 | 0.2826 | 0.1585 | MetaGB | 3.0 | 2.0 | 1.0 |
| C | mixed_gate_regime | 10 | regression | rmse | 0.3014 | 0.3607 | 0.1919 | MetaGB | 2.0 | 3.0 | 1.0 |
| C | mixed_gate_regime | 20 | regression | rmse | 0.4661 | 0.5831 | 0.1955 | MetaGB | 2.0 | 3.0 | 1.0 |
| D | sparse_feature_switch | 10 | regression | rmse | 0.3159 | 0.2294 | 0.2410 | LightGBM | 3.0 | 1.0 | 2.0 |
| D | sparse_feature_switch | 20 | regression | rmse | 0.3028 | 0.2233 | 0.2522 | LightGBM | 3.0 | 1.0 | 2.0 |
| D | sparse_multiplicative | 10 | regression | rmse | 0.5014 | 0.5748 | 0.2355 | MetaGB | 2.0 | 3.0 | 1.0 |
| D | sparse_multiplicative | 20 | regression | rmse | 0.5578 | 0.6268 | 0.2494 | MetaGB | 2.0 | 3.0 | 1.0 |
| D | sparse_regime_slope | 10 | regression | rmse | 0.4793 | 0.4887 | 0.2499 | MetaGB | 2.0 | 3.0 | 1.0 |
| D | sparse_regime_slope | 20 | regression | rmse | 0.5624 | 0.5555 | 0.2978 | MetaGB | 3.0 | 2.0 | 1.0 |
| E | additive_nonlinear_control | 10 | regression | rmse | 0.1814 | 0.1848 | 0.1913 | XGBoost | 1.0 | 2.0 | 3.0 |
| E | additive_nonlinear_control | 20 | regression | rmse | 0.1857 | 0.1877 | 0.2149 | XGBoost | 1.0 | 2.0 | 3.0 |
| E | curved_control | 10 | regression | rmse | 0.1631 | 0.1619 | 0.1630 | LightGBM | 3.0 | 1.0 | 2.0 |
| E | curved_control | 20 | regression | rmse | 0.1617 | 0.1621 | 0.1647 | XGBoost | 1.0 | 2.0 | 3.0 |
| E | pure_step_control | 10 | regression | rmse | 0.2465 | 0.2339 | 0.2546 | LightGBM | 2.0 | 1.0 | 3.0 |
| E | pure_step_control | 20 | regression | rmse | 0.2453 | 0.2338 | 0.2564 | LightGBM | 2.0 | 1.0 | 3.0 |
| F | noise_regime_slope | 10 | regression | rmse | 0.8870 | 0.9010 | 0.7995 | MetaGB | 2.0 | 3.0 | 1.0 |
| F | noise_regime_slope | 20 | regression | rmse | 0.9487 | 0.9540 | 0.8043 | MetaGB | 2.0 | 3.0 | 1.0 |

三模型逐 cell 最优为 MetaGB 28/38、XGBoost 3/38、LightGBM 7/38。六个 negative-control cells 全由 LightGBM 获胜；因此该表支持机制选择性，而不是 synthetic 全局优势。

### 6.1 Dim50 不完整增量

0716 dim50 原始目录中可解析出 130 条 XGBoost/MetaGB fold rows。严格要求双方五折完整后得到 11 个 cells；其 dataset-level M/T/X 为 11/0/0。该完成子集没有覆盖 sparse、negative controls 和 noise families。

| Scenario | Dim | Metric | XGB | MetaGB | Delta | W | Folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| categorical_feature_switch__dim50 | 50 | rmse | 0.1694 | 0.1541 | +9.0% | M | 5 |
| categorical_regime_slope__dim50 | 50 | rmse | 0.5375 | 0.1576 | +70.7% | M | 5 |
| conditional_multiplicative__dim50 | 50 | rmse | 2.7871 | 0.3751 | +86.5% | M | 5 |
| feature_switch__dim50 | 50 | rmse | 0.2207 | 0.2113 | +4.3% | M | 5 |
| global_linear__dim50 | 50 | rmse | 0.1584 | 0.1536 | +3.0% | M | 5 |
| multiplicative__dim50 | 50 | rmse | 0.3352 | 0.1850 | +44.8% | M | 5 |
| regime_logit__dim50 | 50 | auc | 0.858304 | 0.917931 | +0.059627 | M | 5 |
| regime_slope__dim50 | 50 | rmse | 0.5822 | 0.2594 | +55.5% | M | 5 |
| regime_slope_multi__dim50 | 50 | rmse | 0.4220 | 0.2369 | +43.9% | M | 5 |
| xor_3way__dim50 | 50 | auc | 0.574337 | 0.994271 | +0.419934 | M | 5 |
| xor_classification__dim50 | 50 | auc | 0.996407 | 0.998074 | +0.001668 | M | 5 |

<a id="formal-ablation"></a>

## 7. 正式 Fixed-Backbone Synthetic 消融

正式消融为 7 scenarios × 2 dimensions × 3 data seeds × 5 outer folds × 16 variants。完整性审计为 210/210 outer tasks、3360/3360 fold rows 和 2,016,000/2,016,000 predictions。development seed 选择 XGBoost backbone 后，所有变体共享 backbone、split 和 model seed。

主效应定义：Delta split = SplitOnly(p=1, greedy) vs Bias；Delta leaf = Full(p=1, greedy) vs SplitOnly；Delta total = Full(p=1, greedy) vs Bias。百分比是三个 seed-level 相对改善的均值，括号为正向 seeds 数。

### 7.1 十六个变体清单

| Variant | Family | Leaf mode | Pred. fraction | Aggregation |
| --- | --- | --- | --- | --- |
| bias_p00_sa0 | bias | bias_only | 0.0 | greedy |
| full_p02_sa0 | full | predictive | 0.2 | greedy |
| full_p04_sa0 | full | predictive | 0.4 | greedy |
| full_p06_sa0 | full | predictive | 0.6 | greedy |
| full_p08_sa0 | full | predictive | 0.8 | greedy |
| full_p10_sa0 | full | predictive | 1.0 | greedy |
| full_p10_sa1 | full | predictive | 1.0 | mean |
| full_p10_sa2 | full | predictive | 1.0 | median |
| split_p02_sa0 | split_only | split_only | 0.2 | greedy |
| split_p04_sa0 | split_only | split_only | 0.4 | greedy |
| split_p06_sa0 | split_only | split_only | 0.6 | greedy |
| split_p08_sa0 | split_only | split_only | 0.8 | greedy |
| split_p10_sa0 | split_only | split_only | 1.0 | greedy |
| split_p10_sa1 | split_only | split_only | 1.0 | mean |
| split_p10_sa2 | split_only | split_only | 1.0 | median |
| xgb | baseline | constant | -- | not_applicable |

### 7.2 四个主变体的原始 loss

| Scenario | Dim | Metric | XGB | Bias | SplitOnly | Full |
| --- | --- | --- | --- | --- | --- | --- |
| categorical_regime_slope | 10 | rmse | 0.3778 | 0.3258 | 0.3803 | 0.3231 |
| categorical_regime_slope | 20 | rmse | 0.3918 | 0.4338 | 0.3282 | 0.3208 |
| curved_control | 10 | rmse | 0.1594 | 0.1591 | 0.1692 | 0.1612 |
| curved_control | 20 | rmse | 0.1595 | 0.1593 | 0.1706 | 0.1625 |
| feature_switch | 10 | rmse | 0.2188 | 0.2257 | 0.2313 | 0.2049 |
| feature_switch | 20 | rmse | 0.2084 | 0.2131 | 0.2296 | 0.1985 |
| pure_step_control | 10 | rmse | 0.2578 | 0.2422 | 0.2501 | 0.2510 |
| pure_step_control | 20 | rmse | 0.2612 | 0.2448 | 0.2524 | 0.2552 |
| regime_logit | 10 | logloss | 0.4027 | 0.4017 | 0.3811 | 0.4219 |
| regime_logit | 20 | logloss | 0.4309 | 0.4310 | 0.4096 | 0.4352 |
| regime_slope | 10 | rmse | 0.5443 | 0.5380 | 0.3224 | 0.3020 |
| regime_slope | 20 | rmse | 0.5581 | 0.6043 | 0.3096 | 0.2828 |
| sparse_multiplicative | 10 | rmse | 0.6404 | 0.6651 | 1.1529 | 1.8670 |
| sparse_multiplicative | 20 | rmse | 0.7684 | 0.7863 | 0.9029 | 1.7440 |

### 7.3 配对机制效应

| Scenario | Dim | Metric | Delta split | Delta leaf | Delta total | Full-XGB |
| --- | --- | --- | --- | --- | --- | --- |
| categorical_regime_slope | 10 | rmse | -29.2% (1/3) | +15.1% (3/3) | -8.8% (2/3) | +0.1% (2/3) |
| categorical_regime_slope | 20 | rmse | +22.2% (3/3) | +2.4% (2/3) | +23.0% (2/3) | +15.5% (2/3) |
| curved_control | 10 | rmse | -6.4% (0/3) | +4.7% (3/3) | -1.3% (0/3) | -1.1% (0/3) |
| curved_control | 20 | rmse | -7.1% (0/3) | +4.7% (3/3) | -2.0% (0/3) | -1.9% (0/3) |
| feature_switch | 10 | rmse | -3.5% (1/3) | +11.3% (3/3) | +8.9% (2/3) | +5.6% (2/3) |
| feature_switch | 20 | rmse | -7.7% (0/3) | +13.6% (3/3) | +7.0% (2/3) | +4.6% (2/3) |
| pure_step_control | 10 | rmse | -3.3% (0/3) | -0.4% (1/3) | -3.6% (0/3) | +2.4% (2/3) |
| pure_step_control | 20 | rmse | -3.3% (1/3) | -1.0% (1/3) | -4.0% (1/3) | +2.5% (2/3) |
| regime_logit | 10 | logloss | +5.2% (3/3) | -10.7% (0/3) | -5.0% (0/3) | -4.7% (0/3) |
| regime_logit | 20 | logloss | +4.9% (3/3) | -6.2% (0/3) | -1.0% (1/3) | -1.0% (0/3) |
| regime_slope | 10 | rmse | +38.9% (3/3) | +6.5% (2/3) | +43.1% (3/3) | +43.6% (3/3) |
| regime_slope | 20 | rmse | +48.6% (3/3) | +8.6% (3/3) | +53.2% (3/3) | +48.0% (3/3) |
| sparse_multiplicative | 10 | rmse | -73.5% (0/3) | -61.9% (0/3) | -181.3% (0/3) | -194.1% (0/3) |
| sparse_multiplicative | 20 | rmse | -15.0% (0/3) | -93.2% (0/3) | -122.2% (0/3) | -128.9% (0/3) |

最强正证据来自 regime_slope：Full-Bias 为 +43.1%/+53.2%，主要由 split 的 +38.9%/+48.6% 驱动。Feature switch 的独立 leaf 效应为 +11.3%/+13.6%。Regime logit 中 split 正、leaf 负；sparse multiplicative 在 p=1 下两个组件均失败。两个 controls 的 Full-Bias 均为负。

### 7.4 Predictive-feature fraction

| Scenario | Dim | p=.2 | p=.4 | p=.6 | p=.8 | p=1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| categorical_regime_slope | 10 | +22.7% | -6.3% | -21.6% | -39.0% | -8.8% |
| categorical_regime_slope | 20 | +14.7% | +4.1% | +6.2% | +23.8% | +23.0% |
| curved_control | 10 | -20.2% | -13.9% | -3.4% | -2.2% | -1.3% |
| curved_control | 20 | -14.8% | -8.7% | -5.3% | -3.4% | -2.0% |
| feature_switch | 10 | -0.8% | -1.3% | +4.0% | +6.3% | +8.9% |
| feature_switch | 20 | -7.8% | -1.6% | +3.3% | +3.9% | +7.0% |
| pure_step_control | 10 | +1.5% | -0.7% | -2.7% | -2.6% | -3.6% |
| pure_step_control | 20 | -0.0% | -1.2% | -1.8% | -3.7% | -4.0% |
| regime_logit | 10 | -0.0% | -1.4% | -3.0% | -4.1% | -5.0% |
| regime_logit | 20 | +7.6% | +7.9% | +7.7% | +6.1% | -1.0% |
| regime_slope | 10 | +31.6% | +37.8% | +42.5% | +45.2% | +43.1% |
| regime_slope | 20 | +37.9% | +45.5% | +50.6% | +53.7% | +53.2% |
| sparse_multiplicative | 10 | +4.3% | -28.4% | -67.7% | -101.9% | -181.3% |
| sparse_multiplicative | 20 | -23.6% | -59.7% | -72.0% | -84.1% | -122.2% |

Pred_fraction 不是跨场景统一单调剂量。Regime slope 上升后平台；feature switch 随覆盖率提高转正；regime logit 的 leaf 负效应随 p 放大；sparse multiplicative 在高 p 明显失稳。

### 7.5 Score aggregation

| Scenario | Dim | Full mean | Full median | Split mean | Split median |
| --- | --- | --- | --- | --- | --- |
| categorical_regime_slope | 10 | +21.3% | -192.3% | +27.0% | -298.4% |
| categorical_regime_slope | 20 | -33.5% | -389.4% | +13.3% | -779.0% |
| curved_control | 10 | -0.9% | -192.9% | +0.2% | -219.8% |
| curved_control | 20 | -1.5% | -383.9% | -0.9% | -433.1% |
| feature_switch | 10 | +2.2% | -173.6% | -6.9% | -202.3% |
| feature_switch | 20 | +1.3% | -539.3% | -1.7% | -782.2% |
| pure_step_control | 10 | -2.3% | -113.7% | +0.9% | -106.1% |
| pure_step_control | 20 | -2.0% | -223.7% | +3.2% | -183.9% |
| regime_logit | 10 | +0.4% | +0.9% | -5.0% | -12.9% |
| regime_logit | 20 | -4.7% | -3.8% | -7.5% | -14.0% |
| regime_slope | 10 | -5.3% | -211.0% | -9.7% | -287.4% |
| regime_slope | 20 | -6.5% | -366.3% | -7.5% | -426.2% |
| sparse_multiplicative | 10 | +49.5% | +59.7% | +42.5% | +29.3% |
| sparse_multiplicative | 20 | +54.3% | +45.4% | +31.9% | -57.6% |

Mean 的效果依赖场景与 leaf mode。Median 在 Full 中有 11/14 个 cells 的 paired CI 完全为负，在 SplitOnly 中有 13/14 个，不能作为稳健默认替代。

<a id="time-series"></a>

## 8. Time-Series

完整计划为 7 datasets × 2 seq_len × 3 horizons × 4 models = 168 cells。当前 72 cells 完成：三种常规树均完成 seq_len=24 的 21/21；MetaGB 完成 9/21；seq_len=48 为 0/84。所有结果使用 fixed split 和 seed 42。

| Dataset | Seq | H | LGB RMSE | XGB RMSE | Cat RMSE | MGB RMSE | Best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 24 | 1 | 0.0712 | 0.0713 | 0.0782 | 0.0735 | LightGBM |
| ETTh1 | 24 | 6 | 0.1244 | 0.1236 | 0.1338 | -- | XGBoost |
| ETTh1 | 24 | 12 | 0.1768 | 0.1648 | 0.1613 | -- | CatBoost |
| ETTh2 | 24 | 1 | 0.0655 | 0.0668 | 0.0922 | 0.0779 | LightGBM |
| ETTh2 | 24 | 6 | 0.1950 | 0.2010 | 0.2039 | -- | LightGBM |
| ETTh2 | 24 | 12 | 0.2707 | 0.2769 | 0.2808 | -- | LightGBM |
| ETTm1 | 24 | 1 | 0.0375 | 0.0379 | 0.0413 | 0.0362 | MetaGB |
| ETTm1 | 24 | 6 | 0.0683 | 0.0684 | 0.0728 | -- | LightGBM |
| ETTm1 | 24 | 12 | 0.0916 | 0.0909 | 0.0945 | -- | XGBoost |
| ETTm2 | 24 | 1 | 0.0273 | 0.0389 | 0.0316 | 0.0219 | MetaGB |
| ETTm2 | 24 | 6 | 0.0598 | 0.0660 | 0.0724 | -- | LightGBM |
| ETTm2 | 24 | 12 | 0.1107 | 0.1051 | 0.1175 | -- | XGBoost |
| ILI | 24 | 1 | 1.9544 | 1.7782 | 1.9056 | 0.5401 | MetaGB |
| ILI | 24 | 6 | 1.8800 | 1.8807 | 1.8953 | 0.8044 | MetaGB |
| ILI | 24 | 12 | 1.9622 | 1.9653 | 1.9737 | 1.0917 | MetaGB |
| Weather | 24 | 1 | 0.0382 | 0.3074 | 0.1068 | -- | LightGBM |
| Weather | 24 | 6 | 0.2015 | 0.4090 | 0.3557 | -- | LightGBM |
| Weather | 24 | 12 | 0.2550 | 0.6420 | 0.2558 | -- | LightGBM |
| exchange_rate | 24 | 1 | 0.1793 | 0.0690 | 0.2777 | 0.0507 | MetaGB |
| exchange_rate | 24 | 6 | 0.1710 | 0.1178 | 0.2284 | 0.0548 | MetaGB |
| exchange_rate | 24 | 12 | 0.1884 | 0.1758 | 0.4245 | -- | XGBoost |

### 8.1 MetaGB 的九个四模型共同 cells

| Dataset | Seq | H | Strongest | Base RMSE | MGB RMSE | Delta | Runtime | W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 24 | 1 | LightGBM | 0.0712 | 0.0735 | -3.2% | 113.85x | B |
| ETTh2 | 24 | 1 | LightGBM | 0.0655 | 0.0779 | -18.9% | 111.92x | B |
| ETTm1 | 24 | 1 | LightGBM | 0.0375 | 0.0362 | +3.4% | 100.31x | M |
| ETTm2 | 24 | 1 | LightGBM | 0.0273 | 0.0219 | +19.7% | 93.81x | M |
| ILI | 24 | 1 | XGBoost | 1.7782 | 0.5401 | +69.6% | 74.54x | M |
| ILI | 24 | 6 | LightGBM | 1.8800 | 0.8044 | +57.2% | 123.90x | M |
| ILI | 24 | 12 | LightGBM | 1.9622 | 1.0917 | +44.4% | 193.50x | M |
| exchange_rate | 24 | 1 | XGBoost | 0.0690 | 0.0507 | +26.5% | 93.92x | M |
| exchange_rate | 24 | 6 | XGBoost | 0.1178 | 0.0548 | +53.4% | 89.65x | M |

MetaGB 相对逐 cell 最强常规树为 7/2，平均和中位 RMSE 改善为 +28.02%/+26.50%，端到端 elapsed ratio 中位数为 100.31x。负向 cells 为 ETTh1 h=1 和 ETTh2 h=1。Coverage 非随机且不足一半，因此只能作为 targeted evidence。

<a id="scope-history"></a>

## 9. 历史口径与当前口径对照

| 版本 | 口径 | 结果 | 使用说明 |
| --- | --- | --- | --- |
| 0702 总报告 | 真实 MGB-XGB | 125 cells: 49/8/68 | 含 TabArena main、TabArena30、Other tabular；协议分开解释 |
| 0714 baseline 报告 | Strongest baseline vs MetaGB | 41/3/2 | 历史快照；CatBoost/RealMLP 尚未完全补齐 |
| 0716 最新补全 | Strongest available envelope vs MetaGB | 42/2/2 | 当前权威 oracle diagnostic |
| 当前 main.tex | 八模型共同排名 | MetaGB 4.12, 第 3/8 | 46 cells；不含 RealMLP |
| 0716 九模型严格排名 | 九模型共同排名 | MetaGB 4.59, 第 3/9 | 45 cells；含 RealMLP，删除 electricity-onehot |
| 0702 time-series | MetaGB common coverage | 3/5 | 旧 68-row snapshot |
| 0716 time-series | MetaGB common coverage | 7/9 | 当前 72-row snapshot |

<a id="integrity-limitations"></a>

## 10. 数据完整性、已知差异和限制

Canonical builder 记录了五个 task_status 顶层计数与 task map/terminal log 不一致的文件。处理原则是忽略不一致的顶层计数，使用 task map 和终止日志状态。

| Status file | Reported total | Reported ok | Task-map total | Task-map ok | Resolution |
| --- | --- | --- | --- | --- | --- |
| task_status.json | 300 | 75 | 300 | 65 | coverage uses task map and terminal log status; top-level counters ignored |
| task_status.json | 80 | 75 | 80 | 45 | coverage uses task map and terminal log status; top-level counters ignored |
| task_status.json | 80 | 45 | 80 | 35 | coverage uses task map and terminal log status; top-level counters ignored |
| task_status.json | 110 | 100 | 110 | 105 | coverage uses task map and terminal log status; top-level counters ignored |
| task_status.json | 355 | 240 | 355 | 225 | coverage uses task map and terminal log status; top-level counters ignored |

- Synthetic 10/20 主 benchmark 只有一个 data seed；outer folds 不能替代跨 seed 稳健性。
- 正式消融只有三个独立 seeds，bootstrap CI 用于描述方向和尺度，不用于细粒度确认性显著性。
- 正式消融固定 XGBoost 优选 backbone，回答组件贡献，不回答各变体独立完全调优后的最优性能。
- TabArena30 是 fixed-backbone tail protocol，不能与完整 Random300/Random600 搜索合并。
- RealMLP 缺 electricity-onehot；七 baseline 的非严格排名与九模型严格排名使用不同 coverage。
- Strongest available baseline 是 oracle envelope，不是单一可训练模型。
- Time-series 使用 fixed split，MetaGB coverage 非随机且只有 9/42 完整计划 cells。
- Runtime 是当前共享环境中的端到端任务耗时，缺少完整硬件和软件锁定。
- Retained real-tabular preprocessing 在 outer split 前执行，不能描述为 fold-local 或完全 leakage-free。
- Dim50 只完成机制匹配子集，缺 sparse、negative controls、noise 和 LightGBM 完整对齐。

<a id="data-entrypoints"></a>

## 11. 推荐的数据入口

| 用途 | 路径 |
| --- | --- |
| 所有 MetaGB-XGB cell | description_project/project_paperwriting/artifacts/canonical/dataset_level_results.csv |
| 协议分组汇总 | description_project/project_paperwriting/artifacts/canonical/group_summary.csv |
| Synthetic 三模型 cell | description_project/project_paperwriting/artifacts/canonical/synthetic_three_model_cells.csv |
| Synthetic family 汇总 | description_project/project_paperwriting/artifacts/canonical/synthetic_three_model_summary.csv |
| 七 baseline dataset means | description_project/project_paperwriting/artifacts/canonical/baseline_dataset_results.csv |
| 当前八模型 rank | description_project/project_paperwriting/artifacts/canonical/real_baseline_rank_matrix.csv |
| Time-series 全部 cells | description_project/project_paperwriting/artifacts/canonical/timeseries_results.csv |
| Time-series MetaGB 对照 | description_project/project_paperwriting/artifacts/canonical/timeseries_metagb_comparison.csv |
| Formal ablation 原始主变体 | dataAddition0716/metagb_ablation_syn/reanalysis/syn_formal_analysis/main_ablation.csv |
| Formal ablation 主效应 | dataAddition0716/metagb_ablation_syn/reanalysis/syn_formal_analysis/main_effects.csv |
| Pred_fraction 效应 | dataAddition0716/metagb_ablation_syn/reanalysis/syn_formal_analysis/pred_effects.csv |
| Aggregation 效应 | dataAddition0716/metagb_ablation_syn/reanalysis/syn_formal_analysis/scoreagg_effects.csv |
| Canonical hashes/counts | description_project/project_paperwriting/artifacts/provenance/canonical_build_report.json |
| Canonical discrepancy ledger | description_project/project_paperwriting/artifacts/provenance/canonical_discrepancy_ledger.csv |
| 0716 最新解释 | dataAddition0716/LATEST_CONCLUSIONS_20260718.zh copy.md |
| Formal ablation 中文总结 | dataAddition0716/metagb_ablation_syn/ABLATION_RESULT_SUMMARY_ZH.md |

后续修改表格时，优先从上述 CSV 重新筛选和透视，而不是从 PDF 或 Markdown 反向抄数。本手册的表格用于检索与核对；canonical CSV 和 ablation reanalysis CSV 才是可编程数据入口。
