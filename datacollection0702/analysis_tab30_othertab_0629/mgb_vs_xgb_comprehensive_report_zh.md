# MGB vs XGB 阶段性结果汇报稿（0629 中文版）

## 1. 本轮汇报目的

本轮主要不是继续补代码，而是把目前已有的 `datacollection0624` 和
`mgb_datacollection_by0629` 结果统一整理，回答三个问题：

1. 当前 MGB 相比 XGB 的总体效果如何？
2. MGB 的优势是否符合方法设计理念，即“条件特征定义区域，预测特征在叶内提供局部线性响应”？
3. 对于 TabArena 中训练极慢的长尾数据，使用 `XGB-best backbone + MGB 30-grid` 是否可以作为合理的补表方案？

本报告只统计 XGB 与 current_metagb 在同一数据集、同一 outer fold 下都完成的 paired folds。未完成或只有单边结果的任务只进入 coverage 和后续时间估计，不进入胜负统计。

## 2. 数据来源与统计口径

当前分析覆盖四类结果：

| 数据来源 | 内容 | 协议 | 作用 |
|---|---|---|---|
| `datacollection0624` | TabArena 主体实验 | Random300/Random600 | 主实验真实数据证据 |
| `mgb_datacollection_by0629/tabarena30_*` | TabArena 长尾后续实验 | XGB-best + 10 pred_fraction x 3 score_agg | 补长尾表格，降低 MGB 搜索成本 |
| `mgb_datacollection_by0629/tabular_otherdata_*` | 额外中等规模 tabular 数据 | Random300 | 寻找时间经济、MGB 匹配的数据 |
| `mgb_datacollection_by0629/synthetic_main102050_random300_seed42` | 合成机制实验 | Random300 single seed | 验证 MGB 设计理念 |

指标口径：

- Regression：RMSE，越低越好。
- Binary：AUC，越高越好。
- Multiclass：logloss，越低越好。
- AUC 绝对差不超过 `1e-4` 记为 tie。
- RMSE/logloss 相对差不超过 `0.1%` 记为 tie。

当前共解析：

| 项目 | 数量 |
|---|---:|
| model/fold 原始记录 | 1779 |
| paired folds | 699 |
| dataset-level 可比较单元 | 142 |
| 总体 MGB / Tie / XGB | 74 / 7 / 61 |

## 3. 总体结论

目前结果支持一个比较稳妥的判断：

> MGB 不是在所有真实数据上全面超过 XGB，但它在机制匹配的数据上有明确优势；synthetic 实验强力支持其方法理念，TabArena 和 other tabular 结果则显示真实数据中存在正向 pocket，同时也存在 XGB 更适合的负向场景。

因此论文或汇报中更适合强调：

1. **机制创新成立**：在条件切分、叶内线性响应、conditional/predictive feature 分离、稀疏交互等结构上，MGB 明显优于常数叶 XGB。
2. **真实数据表现是选择性优势**：TabArena 主表不是全面赢，而是部分任务、部分数据集上有收益。
3. **运行时间需要诚实解释**：MGB 在完整 Random300/600 下明显更慢，卖点不是速度，而是“额外表达能力是否值得额外计算”。
4. **TabArena30 是合理补表策略**：对于长尾慢数据，用 XGB-best backbone 固定共享参数，只搜索 MGB 特有的 30 个组合，可以显著降低补表成本，但不能和完整 Random300/600 混为同一个实验协议。

## 4. 分组 Win/Tie/Loss

| 数据族 | 协议 | 预处理 | 任务 | 数据集数 | paired folds | MGB / Tie / XGB | 中位 runtime ratio |
|---|---|---|---|---:|---:|---:|---:|
| Synthetic | Random300 single seed | synthetic | binary | 6 | 30 | 6 / 0 / 0 | 46.40x |
| Synthetic | Random300 single seed | synthetic | regression | 29 | 145 | 24 / 1 / 4 | 15.23x |
| TabArena30 | XGB-best + MGB30 | drop | binary | 9 | 38 | 3 / 0 / 6 | 2.62x |
| TabArena30 | XGB-best + MGB30 | onehot | binary | 8 | 40 | 3 / 1 / 4 | 2.02x |
| TabArena30 | XGB-best + MGB30 | onehot | multiclass | 3 | 11 | 3 / 0 / 0 | 4.27x |
| TabArena main | Random300 | drop | binary | 17 | 85 | 8 / 0 / 9 | 22.01x |
| TabArena main | Random300 | onehot | binary | 13 | 65 | 5 / 0 / 8 | 58.98x |
| TabArena main | Random300 | drop | multiclass | 4 | 20 | 2 / 1 / 1 | 26.23x |
| TabArena main | Random300 | onehot | multiclass | 3 | 15 | 2 / 0 / 1 | 21.48x |
| TabArena main | Random600 | drop | regression | 11 | 55 | 3 / 0 / 8 | 10.30x |
| TabArena main | Random600 | onehot | regression | 10 | 50 | 5 / 0 / 5 | 13.81x |
| Other tabular | Random300 | drop | binary | 8 | 40 | 2 / 2 / 4 | 14.02x |
| Other tabular | Random300 | onehot | binary | 5 | 25 | 2 / 2 / 1 | 10.37x |
| Other tabular | Random300 | drop | regression | 7 | 35 | 3 / 0 / 4 | 10.09x |
| Other tabular | Random300 | onehot | regression | 6 | 30 | 2 / 0 / 4 | 22.37x |

## 5. Synthetic：最能体现 MGB 理念的证据

Synthetic 结果是目前最清楚的机制证据。当前可比较的 synthetic dataset-cell 中，MGB/Tie/XGB 为：

| 机制类型 | MGB | Tie | XGB | 解读 |
|---|---:|---:|---:|---|
| A：交互 / XOR / multiplicative | 8 | 0 | 0 | MGB 稳定胜出，说明局部线性叶能更有效表达交互 |
| B：linear-in-leaves | 9 | 0 | 1 | 大多数区域斜率、特征切换、logit slope 场景 MGB 胜出 |
| C：conditional-predictive separation | 6 | 0 | 0 | 类别/条件变量决定区域，连续变量负责预测时，MGB 很匹配 |
| D：sparse interaction | 6 | 0 | 0 | 大量无关特征下，MGB 仍能抓住局部线性结构 |
| E：negative control | 0 | 1 | 3 | 负对照没有普遍偏向 MGB，说明不是“所有 synthetic 都硬赢” |
| F：noise pressure | 1 | 0 | 0 | 加强噪声的区域斜率结构下仍有优势 |

这部分可以作为论文中最核心的机制叙述：

> XGBoost 的常数叶需要用更多切分去近似区域内的线性响应；MGB 在叶子中允许一个预测特征提供局部线性项，因此当数据真实机制是“先由 conditional feature 定义 regime，再由 predictive feature 在 regime 内线性影响目标”时，表达效率和泛化都更好。

代表性正向 case：

| Case | 任务 | XGB test | MGB test | 改善 |
|---|---|---:|---:|---:|
| `conditional_multiplicative__dim20` | RMSE | 2.1523 | 0.3551 | 83.5% |
| `conditional_multiplicative__dim10` | RMSE | 1.5504 | 0.3748 | 75.8% |
| `mixed_gate_regime__dim20` | RMSE | 0.4661 | 0.1955 | 58.1% |
| `sparse_multiplicative__dim20` | RMSE | 0.5578 | 0.2494 | 55.3% |
| `categorical_regime_slope__dim20` | RMSE | 0.3008 | 0.1585 | 47.3% |

## 6. TabArena 主体：真实数据上是混合结果

TabArena 主体结果不支持“全面优于 XGB”的强说法，但支持“存在明确正向场景”的说法。

Regression：

- Drop Random600：MGB/Tie/XGB = 3/0/8。
- OneHot Random600：MGB/Tie/XGB = 5/0/5。
- 正向数据包括 `QSAR_fish_toxicity.csv`、`physiochemical_protein.csv`、`miami_housing.csv` 等。
- 负向数据包括 `houses.csv`、`airfoil_self_noise.csv`、`diamonds.csv` 等。

Binary：

- Drop Random300：MGB/Tie/XGB = 8/0/9。
- OneHot Random300：MGB/Tie/XGB = 5/0/8。
- 正向数据包括 `credit-g.csv`、`churn.csv`、`hazelnut-spread-contaminant-detection.csv`、`blood-transfusion-service-center.csv` 等。
- 负向数据包括 `polish_companies_bankruptcy.csv`、`seismic-bumps.csv` 等。

Multiclass：

- Drop Random300：MGB/Tie/XGB = 2/1/1。
- OneHot Random300：MGB/Tie/XGB = 2/0/1。
- 当前 multiclass 更偏正向，但数据集数量较少，因此适合做支持性证据，不宜单独做过强结论。

这部分对老师可以这样讲：

> 真实 TabArena 上 MGB 的优势不是均匀分布的。它在部分数据集上有效，但在另一些标准树模型已经很适合的数据上会输。因此主论文要避免“平均全面 SOTA”的叙事，更适合做“结构条件下的表达能力增强 + 真实数据 positive pockets + 机制实验验证”。

## 7. TabArena30：长尾数据的补表策略

TabArena30 的核心是：对于 MGB 完整 Random300 很慢的长尾数据，先使用 XGB 选择好的 backbone，再只搜索 MGB 的两个特有维度：

```text
pred_feature_fraction = 0.1, 0.2, ..., 1.0
metagb_score_agg = 0, 1, 2
```

也就是每个 fold 只跑 30 个 MGB candidates，而不是完整 300 个。

当前结果：

| 任务 | 预处理 | MGB/Tie/XGB | 解读 |
|---|---|---:|---|
| binary | drop | 3/0/6 | 效果偏混合，很多 AUC 差距很小 |
| binary | onehot | 3/1/4 | 接近五五开，适合作为补表而非强结论 |
| multiclass | onehot | 3/0/0 | 当前偏正向，但样本数仍少 |

TabArena30 的价值主要有两点：

1. **工程上可行**：大幅减少 MGB 搜索成本，使长尾数据可以进入表格。
2. **解释上要分开**：它不是完整 Random300 MGB，而是 fixed-backbone MGB30，因此论文表格中应单独标注 protocol。

## 8. Tabular other data：用于寻找 positive pockets

额外 tabular 数据当前总体 MGB/Tie/XGB = 10/4/15，仍然是混合结果。

正向较明显的数据：

| Dataset | 任务 | 指标 | XGB | MGB | 改善 |
|---|---|---|---:|---:|---:|
| `vowel.csv` | multiclass | logloss | 0.2282 | 0.1320 | 42.1% |
| `kin8nm.csv` | regression | RMSE | 0.1125 | 0.1003 | 10.9% |
| `Moneyball.csv` | regression | RMSE | 23.34 | 22.26 | 4.6% |
| `ilpd.csv` | binary | AUC | 0.7518 | 0.7696 | +0.0178 |
| `phoneme.csv` | binary | AUC | 0.9592 | 0.9656 | +0.0064 |

负向较明显的数据：

| Dataset | 任务 | 指标 | XGB | MGB | 变化 |
|---|---|---|---:|---:|---:|
| `balance-scale.csv` | multiclass | logloss | 0.2338 | 0.3429 | -46.7% |
| `boston.csv` | regression | RMSE | 3.0492 | 3.4493 | -13.1% |
| `energy_efficiency.csv` | regression | RMSE | 0.3461 | 0.3574 | -3.3% |

这说明额外数据集有价值，但需要继续按“机制匹配度 + 时间经济性”筛选，而不是盲目增加数据。

## 9. 数据规模与运行时间

MGB 的计算压力与特征数强相关，粗略代理是：

```text
pressure_proxy = rows * features^2
```

当前高压力数据包括：

| 数据集 | rows | features | pressure_proxy | 当前胜者 |
|---|---:|---:|---:|---|
| `coil2000_insurance_policies.csv` | 9822 | 80 | 62.86M | XGB |
| `taiwanese_bankruptcy_prediction.csv` | 6819 | 94 | 60.25M | XGB |
| `NATICUSdroid.csv` | 7491 | 86 | 55.40M | MGB |
| `diamonds.csv` onehot | 53940 | 26 | 36.46M | XGB |
| `APSFailure.csv` | 756 | 170 | 21.85M | XGB |

运行时间结论：

- 完整 Random300/600 中，MGB 通常比 XGB 慢 10x-60x。
- onehot 高维数据会显著放大 MGB 成本。
- TabArena30 把部分 binary 任务中位 runtime ratio 降到约 2x-3x，但宽表和长尾任务仍会很慢。
- 因此后续实验应优先跑中低 feature、机制更匹配的数据，而不是继续硬跑所有高压长尾。

## 10. 后续任务时间估计

基于当前已完成任务的 median elapsed time，粗略串行任务小时如下：

| Run | 剩余任务 | median 每任务小时 | 串行剩余小时 |
|---|---:|---:|---:|
| `tabarena30_drop/tabarena30_binary_drop_20260628_023950` | 17 | 0.68h | 11.61h |
| `tabarena30_onehot` | 104 | 0.56h | 57.84h |
| `tabular_otherdata_drop` | 85 | 0.56h | 47.53h |
| `tabular_otherdata_onehot` | 135 | 0.29h | 38.70h |

注意：这是串行任务小时，不等于实际 wall time。实际耗时要除以有效并行 worker 数，但还要考虑慢尾任务会拖长总时间。

## 11. 汇报主线

回顾总结以下：

1. **先讲问题**：MGB 的目标不是替代所有 XGB，而是增强 XGB 常数叶在局部线性结构上的表达能力。
2. **再讲 synthetic**：机制实验强烈支持这一点，A-D/F 机制类几乎全部 MGB 胜出，负对照没有普遍偏向 MGB。
3. **再讲真实数据**：TabArena 主体是混合结果，说明真实数据中 MGB 有 positive pockets，但不是全面优势。
4. **解释运行时间**：MGB 慢是预期内的，因为它增加了叶内预测特征搜索；因此重点是 runtime-quality tradeoff。
5. **说明 TabArena30**：对于长尾数据，用 XGB-best + MGB30 是合理补表策略，但需要和完整 Random300/600 分开标注。
6. **后续计划**：补齐重点未完成任务，优先机制匹配和低 pressure 数据；同时做更清晰的 positive/negative case 分析。

一句话版本：

> 当前结果显示，MGB 的方法理念在合成机制实验中非常清楚，在真实 TabArena 中体现为选择性优势而非全面胜利；后续论文应围绕“局部线性叶增强 XGBoost 表达能力、在机制匹配数据上获得收益、以额外运行时间换取性能提升”来组织。

