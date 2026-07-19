# 0716 数据补全后的最新实验结论

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + paper-oriented synthesis
- Origin Date: 2026-07-18
- Verification Status: ANALYZED
- Version Label: `0716_addition_conclusions_v2_with_formal_ablation`
- Snapshot Time: 2026-07-18 23:37 CST (+0800)
- Overall Confidence: CAUTION

`ANALYZED` 表示本报告重新解析了现有日志、JSON、CSV 和状态文件，并按统一口径
完成描述性比较；没有重跑训练，因此不标记为 `VERIFIED`。既有 synthetic 主
benchmark 仍为单 seed，正式 synthetic 消融包含 3 个独立 data seeds，其他
结果还包含部分覆盖和大量逐 cell 比较，不能把胜负计数或探索性 bootstrap CI
写成充分的统计显著性结论。

## 1. 更新范围与统一口径

本报告是
`../datacollection0702/analysis_aaai_0702/timeseries_and_random300_baseline_analysis_20260714.md`
及 synthetic 0702 分析的增量更新，不改写旧报告。

四个证据块使用不同统计单位，不能直接相加：

- tabular baseline：同一 dataset、preprocessing、task 下完整 5 outer folds 的
  dataset-level mean；
- synthetic 主 benchmark：同一 scenario、dimension、model 下完整 5 outer folds 的
  scenario cell；
- synthetic 正式消融：同一 scenario、dimension、data seed 下由 5 个 outer
  folds 的 OOF predictions 重算一个 seed-level metric，再在 3 个 seeds 内
  做严格配对；
- timeseries：固定 split、单 outer/inner fold 的
  dataset × `seq_len` × `pred_len` cell。

统一 tie 规则：

- AUC：绝对差不超过 `1e-4`；
- logloss/RMSE：对称相对差
  `abs(a-b) / max(abs(a), abs(b)) <= 1e-3`。

指标方向：

- binary 使用 AUC，越高越好；
- multiclass 使用 logloss，越低越好；
- regression 和 timeseries 使用 RMSE，越低越好。

## 2. 执行摘要

最新数据使论文故事更清晰，而不是使“全面领先”更可信：

1. **Synthetic 主 benchmark 仍是主要效果证据。** 10/20 维 XGBoost–MetaGB 结果与 0702
   快照完全一致；加入完整 LightGBM 后，MetaGB 仍在 38 个 cells 中取得
   28 个逐 cell 最优，且优势继续集中在 interaction、local-linear、
   conditional-predictive 和部分 sparse 机制。
2. **正式消融把机制贡献进一步定位，但也收紧了线性叶主张。** 在 fixed-XGB-
   backbone、3 个独立 seeds 的配对设计下，`regime_slope` 的 Full 相对 Bias
   改善 43.1%/53.2%，主要由 split 贡献；`feature_switch` 的线性叶独立改善
   11.3%/13.6%。但 `regime_logit` 中 split 的正效应被 leaf 抵消，
   `sparse_multiplicative` 在高覆盖率下系统失败，因此线性叶只能主张为
   **结构依赖的独立表达贡献**，不能主张为普遍增益来源。
3. **强 tabular baseline 的压力进一步增强。** CatBoost 补全后仍是总体最稳健
   baseline，RealMLP 补全后在多个 regression cells 成为新的最优模型。
   Strongest available baseline 在严格对齐 cells 上相对 MetaGB 为
   `42/2/2`，因此真实数据不能写成 MetaGB 广泛击败强 tabular 方法。
4. **Timeseries 的正向证据明显增强，但仍不完整。** 新增 4 个 MetaGB cells
   均击败对应 strongest baseline，使配对结果由 3/5 提高到 7/9；与此同时，
   coverage 仍不足一半，端到端计算成本约为 strongest baseline 的 100 倍。
5. **统一结论仍应是条件性优势。** MetaGB 是一种将结构划分与叶内预测解耦的
   structure-aware XGBoost extension；它在部分局部线性结构中获得强机制
   证据，在真实 tabular 和 timeseries 中表现为 selective transfer，而不是
   universal benchmark dominance。

## 3. Tabular 强 baseline 补全

### 3.1 Coverage 与 0714 快照差异

最终结果必须使用旧目录与 0716 增量的 task union：

1. 先读取
   `../datacollection0702/datacollection0707_baselinemodel_random300/tasks/`；
2. 同 task id 若在
   `tab_otherBaselinemodel0716/logs/tasks/` 中有更新，则使用新日志；
3. 0716 副本缺少的 5 个 TorchMLP 日志从旧目录回退；
4. 只有出现 `finish ... status=ok` 的日志才进入指标聚合。

| 模型 | Started folds | Completed folds | 完整 model-cells | 状态 |
| --- | ---: | ---: | ---: | --- |
| Linear | 270 | 270 | 54/54 | 完整 |
| ExtraTrees | 270 | 270 | 54/54 | 完整 |
| RandomForest | 270 | 270 | 54/54 | 完整 |
| LightGBM | 270 | 270 | 54/54 | 完整 |
| CatBoost | 270 | 270 | 54/54 | 完整 |
| TorchMLP | 270 | 270 | 54/54 | 完整；5 个日志从旧目录回退 |
| RealMLP | 270 | 265 | 53/54 | 仅 `electricity-onehot` 五折未 finish |
| **保留模型合计** | **1890** | **1885** | **377/378** | 0 failed |

相对 0714 报告所用的旧状态，最新 `completed` 从 1662 增至 1922，增加 260：

- CatBoost 新增完成 13 folds；
- RealMLP 新增完成 210 folds；
- 已弃用 TabM 新增 37 folds。

论文保留模型实际增加 223 个完成 folds。TabM 的 62 个 started、37 个 completed
残留任务仅用于审计；按照当前研究决定，它已从模型集合、分母、排名、胜负统计和
“待补实验”叙述中完全移除。因此，后续不应再写“还需补齐 TabM”。

### 3.2 模型排名

54 个 dataset/preprocessing cells 均至少有 6 个完整 baseline；其中 53 个有
全部 7 个 baseline，只有 `electricity-onehot` 缺 RealMLP。

| 模型 | 平均 rank | 参与 cells | 唯一最佳 cells |
| --- | ---: | ---: | ---: |
| CatBoost | **2.259** | 54 | **22** |
| LightGBM | 3.444 | 54 | 1 |
| ExtraTrees | 3.667 | 54 | 7 |
| TorchMLP | 3.676 | 54 | 6 |
| RealMLP | 4.406 | 53 | 8 |
| RandomForest | 4.759 | 54 | 4 |
| Linear | 5.741 | 54 | 4 |

精确同分采用 average rank。`banknote-authentication` 有两个并列 cells：

- drop：ExtraTrees、RealMLP、TorchMLP 三方并列；
- onehot：RealMLP、TorchMLP 并列。

按 fractional wins 分摊并列后，CatBoost、RealMLP、ExtraTrees、TorchMLP、
Linear、RandomForest、LightGBM 分别为
22.000、8.833、7.333、6.833、4、4、1，总和为 54。

相较旧报告，最重要的变化不是总体叙事反转，而是回归 baseline 被明显加强：

- `abalone`、`friedman1`、`kin8nm`、`naval_propulsion_plant` 的 drop 与
  onehot 共 8 个 cells 均改由 RealMLP 最优；
- `mfeat-fourier` 的 drop/onehot 和 `mfeat-karhunen-onehot` 改由 CatBoost
  最优。

### 3.3 与 XGBoost/MetaGB 的严格对齐比较

这里的 `best available baseline` 是每个 cell 在保留 baseline 中取得的最优值，
属于用于诊断竞争压力的 **oracle envelope**，不是一个可训练或可部署的单一模型，
也不应在论文模型表中伪装成一行算法。

| 对照对象 | 对齐 cells | Best baseline 胜 | Tie | Best baseline 负 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 51 | **42** | 2 | 7 |
| MetaGB | 46 | **42** | 2 | 2 |

按任务分层：

| 对照 | 任务 | Cells | Best baseline 胜/Tie/负 |
| --- | --- | ---: | ---: |
| XGBoost | binary | 20 | 15 / 2 / 3 |
| XGBoost | multiclass | 13 | 11 / 0 / 2 |
| XGBoost | regression | 18 | 16 / 0 / 2 |
| MetaGB | binary | 20 | 16 / 2 / 2 |
| MetaGB | multiclass | 8 | 8 / 0 / 0 |
| MetaGB | regression | 18 | 18 / 0 / 0 |

旧报告为 XGBoost `41/2/8`、MetaGB `41/3/2`。更新来自：

- CatBoost 使 `mfeat-fourier-drop` 从 XGBoost 胜变为 baseline 胜；
- RealMLP 使 `abalone-onehot` 从与 MetaGB tie 变为 baseline 胜。

MetaGB 仅在两个 ILPD cells 明确优于 strongest available baseline：

| Preprocessing | MetaGB AUC | Strongest baseline | Baseline AUC | MetaGB 差值 |
| --- | ---: | --- | ---: | ---: |
| drop | 0.759708 | CatBoost | 0.759327 | +0.000381 |
| onehot | 0.769598 | ExtraTrees | 0.767457 | +0.002141 |

`banknote-authentication` 的 drop/onehot 为 ties。强 baseline 相对 MetaGB 的
代表性大优势包括：

- `balance-scale`：logloss 降低 81.35%；
- `naval_propulsion_plant`：RealMLP RMSE 从 MetaGB 的 0.000804 降至
  0.000260，降低 67.66%，drop/onehot 均成立；
- `vowel`：logloss 分别降低 59.54% 和 57.69%；
- `cpu_activity`：RMSE 降低 43.55%；
- `kin8nm`：RealMLP RMSE 从 0.100273 降至 0.064933，降低 35.24%，
  drop/onehot 均成立。

### 3.4 MetaGB 相对每个模型及综合排名

为了公平回答“MetaGB 排第几”，这里使用两个互补统计：

- **逐模型 W/T/L**：MetaGB 与每个模型在双方都有完整 5 folds 的 cells 上比较；
- **严格综合排名**：只使用全部 9 个模型都完整的 45 个 cells。相比 MetaGB
  可用的 46 个 cells，仅排除 RealMLP 未完成的 `electricity-onehot`。

逐模型比较如下，胜/平/负均从 MetaGB 视角报告：

| 对手 | Common cells | MetaGB 胜 | Tie | MetaGB 负 | MetaGB 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 46 | **36** | 0 | 10 | 78.3% |
| RandomForest | 46 | **30** | 3 | 13 | 65.2% |
| ExtraTrees | 46 | **28** | 2 | 16 | 60.9% |
| RealMLP | 45 | **25** | 2 | 18 | 55.6% |
| LightGBM | 46 | **24** | 2 | 20 | 52.2% |
| TorchMLP | 46 | 20 | 3 | **23** | 43.5% |
| CatBoost | 46 | 17 | 2 | **27** | 37.0% |
| XGBoost | 46 | 16 | 4 | **26** | 34.8% |

因此，MetaGB 在逐模型意义上对 8 个对手中的 5 个取得多数胜利：
Linear、RandomForest、ExtraTrees、RealMLP 和 LightGBM；对 TorchMLP 略处
下风，对 CatBoost 和 XGBoost 明显处于下风。

在全部 9 个模型严格共同完成的 45 个 cells 上，按每个 cell 内的名次计算平均
rank，精确同分取 average rank，得到：

| 综合名次 | 模型 | 平均 rank | 中位 rank | Fractional firsts |
| ---: | --- | ---: | ---: | ---: |
| 1 | CatBoost | **3.20** | 3.0 | 9.00 |
| 2 | XGBoost | **4.02** | 5.0 | 7.00 |
| 3 | **MetaGB** | **4.59** | 4.5 | **2.00** |
| 4= | ExtraTrees | 4.70 | 5.0 | 6.33 |
| 4= | TorchMLP | 4.70 | 5.0 | 6.83 |
| 6 | LightGBM | 4.98 | 5.0 | 0.00 |
| 7 | RealMLP | 5.48 | 7.0 | 8.83 |
| 8 | RandomForest | 6.00 | 6.0 | 1.00 |
| 9 | Linear | 7.33 | 9.0 | 4.00 |

MetaGB 因而是 **综合第 3/9**，但只在 2/45 个严格共同覆盖 cells 中排名第一。
这两个第一名均为 ILPD：

- `ilpd-drop`：MetaGB AUC 0.759708；
- `ilpd-onehot`：MetaGB AUC 0.769598。

按任务类型分层，MetaGB 的平均 rank 和组内位置为：

| 任务 | Cells | MetaGB 平均 rank | 组内综合位置 |
| --- | ---: | ---: | ---: |
| Binary | 19 | 4.55 | 4/9 |
| Multiclass | 8 | 4.75 | 4/9 |
| Regression | 18 | 4.56 | 4/9 |

MetaGB 在每类任务中都是第 4，但总体为第 3，并不意味着某个分层计算错误：
ExtraTrees、TorchMLP、RealMLP 等模型在特定任务很强、在其他任务明显下滑；
MetaGB 的三个任务 rank 都稳定在 4.55–4.75，跨任务一致性使其总体平均 rank
略优于这些波动更大的模型。

因此，MetaGB 在真实数据上的优势应分成三层理解：

1. **稳定性优势**：跨 binary、multiclass、regression 都保持中上游，没有某一
   任务类型出现整体崩塌；这是综合排名第 3 的主要来源。
2. **相对中等/传统 baseline 的广泛优势**：对 Linear、RandomForest、
   ExtraTrees、RealMLP、LightGBM 均取得多数胜利。
3. **相对顶级竞争者的选择性优势**：对 CatBoost、XGBoost 和 TorchMLP 没有
   总体优势；真正击败逐 cell strongest baseline 的只有两个 ILPD cells。
   相对 XGBoost 的代表性 positive pockets 包括 `vowel`、`kin8nm`、
   `Moneyball` 和 `naval_propulsion_plant`，但这些收益不足以转化为全局第一。

这意味着真实数据最合适的表述不是“MetaGB 最强”，而是：

> MetaGB 在九模型严格共同覆盖的真实 tabular cells 上取得第三名的综合平均
> 排名，并在三类任务中均保持第四名，显示出较稳定的跨任务竞争力。它相对
> Linear、RandomForest、ExtraTrees、RealMLP 和 LightGBM 取得多数胜利，
> 但仍整体落后于 CatBoost 和 XGBoost，且仅在两个 ILPD cells 中成为逐 cell
> 最优。因此，MetaGB 的真实数据优势体现为稳定的中上游表现和少数结构匹配
> 场景中的明确收益，而不是普遍的最优性能。

### 3.5 更新后的 tabular 结论

可以写：

> 在 27 个 random300 tabular 数据集和两种预处理设置上，补全后的强基线比较
> 显示 CatBoost 是总体最稳健的 baseline，在 54 个 cells 上取得 2.26 的平均
> rank 和 22 个唯一最佳；RealMLP 则明显强化了回归任务的竞争水平。逐 cell
> strongest available baseline 在与 MetaGB 严格对齐的 46 个 cells 中取得
> 42 胜、2 平、2 负，表明 MetaGB 在真实 tabular 数据上的收益具有高度选择性，
> 更适合被定位为 XGBoost 的机制感知结构扩展，而不是广义 tabular SOTA。

不能写：

- “MetaGB 在真实 tabular 上优于强 baseline”；
- “TabM 是当前唯一待补实验”——TabM 已弃用，不属于待补集合；
- 把 `best available baseline` 当成一个实际算法；
- 忽略 RealMLP 比其他模型少 1 个完整 cell。

## 4. Synthetic：LightGBM、dim50 与正式机制消融

### 4.1 10/20 维主实验没有发生漂移

`syn102050part0716/synthetic_main_random300_seed42/tasks/` 中的 10/20 维
380 个 XGBoost/MetaGB task JSON 与
`../datacollection0702/synthetic_main1020full_random300/tasks/` 对应结果一致。
因此，0716 数据没有改写旧的 MetaGB vs XGBoost 主结论：

- 38 个完整 scenario cells、190 paired outer folds；
- dataset level：MetaGB/XGBoost = `31/1/6`；
- fold level：MetaGB/tie/XGBoost = `155/1/34`。

### 4.2 完整 LightGBM 补充

`synbaseline_lgbm0716/` 的计划矩阵为
`38 scenarios × 5 folds × 3 models = 570 tasks`。其中：

- LightGBM：190/190，覆盖 38 cells × 5 folds；
- CatBoost：30/190，仅覆盖 6 个 dim10 cells；
- RandomForest：0/190。

因此用户指定的 LightGBM baseline 是完整的，但整个 runner 没有完成。根据本轮
“只补一个模型”的实验口径，partial CatBoost 和 zero-coverage RandomForest
均不进入正式 rank 或 strongest-baseline 统计。

| 比较 | Dataset-level W/T/L | Fold-level W/T/L |
| --- | ---: | ---: |
| MetaGB vs LightGBM | **28 / 0 / 10** | 149 / 4 / 37 |
| MetaGB vs XGBoost | **31 / 1 / 6** | 155 / 1 / 34 |
| LightGBM vs XGBoost | 14 / 0 / 24 | 82 / 3 / 105 |

三模型逐 cell 最优分布：

| 模型 | 最优 cells | 平均 rank |
| --- | ---: | ---: |
| MetaGB | **28/38** | **1.42** |
| XGBoost | 3/38 | 2.21 |
| LightGBM | 7/38 | 2.37 |

LightGBM 是必要的补充：它把 MetaGB 相对 strongest conventional tree 的优势
从旧的 XGBoost-only `31/1/6` 收紧为三模型 envelope 下的 `28/0/10`，
但没有推翻机制结论。

协议层面，LightGBM 与主实验的 190 个 split-manifest entries 完全一致，
scenario、seed=42、CV seed、5×4 nested folds 和每模型 300 candidates 的预算
均对齐，因此可以做 paired comparison。不过，LightGBM 使用独立的 Simple-HPO
空间，XGBoost/MetaGB 使用 shared-backbone 设计；它是预算匹配的跨模型 baseline，
不是严格的叶机制消融。

### 4.3 机制分层

下表的胜负均为 MetaGB vs LightGBM；signed improvement 对 AUC 为
`MetaGB - LightGBM`，对 RMSE 为从 LightGBM 到 MetaGB 的相对降低。

| 机制族 | Cells | MetaGB/Tie/LightGBM | 中位 signed improvement |
| --- | ---: | ---: | ---: |
| A: interaction / XOR / multiplicative | 8 | **8/0/0** | +0.2411 |
| B: linear-in-leaves | 10 | **8/0/2** | +0.0395 |
| C: conditional-predictive separation | 6 | **6/0/0** | +0.3485 |
| D: sparse interaction | 6 | **4/0/2** | +0.4762 |
| E: negative controls | 6 | **0/0/6** | -0.0619 |
| F: Gaussian-noise pressure | 2 | **2/0/0** | +0.1348 |

dim10 和 dim20 上均为 MetaGB/LightGBM = `14/0/5`。最重要的模式是：

- A、C、F 机制族保持全胜；
- B、D 中存在少量 LightGBM 胜利，说明机制匹配不是充分条件；
- LightGBM 揭示了 XGBoost-only 对照没有充分暴露的边界：
  `feature_switch` dim10/20 与 `sparse_feature_switch` dim10/20 均由
  LightGBM 获胜；
- 6 个 negative controls 全由 LightGBM 获胜，使“generator 天然偏向 MetaGB”
  的解释更不可信，也强化了方法优势的机制特异性。

### 4.4 dim50 只能作为不完整增量

0716 synthetic 主实验目录比 0702 原始目录新增 115 个 tasks，全部来自 dim50。
当前 synthetic 主目录共有 510/570 个 task JSON；dim50 为 130/190。

严格只统计 XGBoost 与 MetaGB 都完成 5 folds 的 cells：

- 11 个完整 paired cells、55 paired folds；
- dataset level MetaGB/XGBoost = `11/0/0`；
- fold level MetaGB/tie/XGBoost = `53/0/2`；
- 中位 signed improvement = `+0.4199`。

此外，`mixed_gate_regime__dim50` 只有 3 个 partial paired folds，不进入
dataset-level 统计。当前 11 个完整 cells 仅覆盖：

- A interaction：4；
- B linear-in-leaves：5；
- C conditional-predictive：2。

D sparse、E negative controls、F noise pressure 均没有完整 dim50 cells，
LightGBM 也没有 dim50 全矩阵。因此，`11/0/0` 只能写成
“已完成的机制匹配 dim50 子集保持强正向”，不能写成：

- “MetaGB 在 dim50 全部 synthetic 场景上获胜”；
- “优势随维度升高而普遍增强”；
- “三模型 dim50 比较完成”。

### 4.5 主 benchmark 更新结论

可以写：

> 在完整的 10/20 维 synthetic benchmark 上，MetaGB 在 38 个
> XGBoost–LightGBM–MetaGB 对齐 cells 中取得 28 个逐 cell 最优，平均 rank
> 为 1.42。优势集中于 interaction、local-linear-in-leaves、
> conditional-predictive separation 和 Gaussian-noise regime，而 6 个
> negative-control cells 全部由 LightGBM 获胜。这种随机制类别出现和消失的
> 模式支持 MetaGB 的局部叶内线性归纳偏置，而不是对所有 synthetic 函数的
> 无条件优势。dim50 已完成的 11 个机制匹配 cells 均由 MetaGB 获胜，但由于
> sparse、negative-control 和 noise families 尚未完整覆盖，该结果仅作为
> 初步维度扩展证据。

### 4.6 正式 fixed-backbone 消融

`metagb_ablation_syn/` 新增了一组与 random300 主 benchmark 分离的正式机制
消融。它先在独立 development seed 上选择 XGBoost backbone，再将相同
backbone、数据划分和模型种子冻结给 16 个变体。完整矩阵为：

```text
7 scenarios × 2 dimensions × 3 data seeds × 5 outer folds × 16 variants
```

离线完整性审计通过：

- 210/210 个 outer tasks；
- 3360/3360 条成功 fold rows；
- 2,016,000/2,016,000 条 predictions；
- 每个 seed 由 5-fold OOF predictions 重算一次 RMSE 或 logloss；
- paired bootstrap 只作用于 3 个 seed-level 配对点，outer folds 不作为 iid
  重复。

主机制效应定义为：

```text
Δsplit = SplitOnly(p=1, greedy) vs Bias
Δleaf  = Full(p=1, greedy) vs SplitOnly(p=1, greedy)
Δtotal = Full(p=1, greedy) vs Bias
```

下表为三个 seed-level 相对改善的均值，正值表示 target loss 更低；每个单元格
依次报告 10 维/20 维：

| Scenario | Δsplit | Δleaf | Δtotal | 公允判断 |
| --- | ---: | ---: | ---: | --- |
| `regime_slope` | +38.9% / +48.6% | +6.5% / +8.6% | **+43.1% / +53.2%** | split 主导、leaf 增强；最强完整正证据 |
| `feature_switch` | -3.5% / -7.7% | **+11.3% / +13.6%** | +8.9% / +7.0% | 线性叶最干净的独立贡献 |
| `categorical_regime_slope` | -29.2% / +22.2% | +15.1% / +2.4% | -8.8% / +23.0% | 维度与候选覆盖高度敏感 |
| `regime_logit` | +5.2% / +4.9% | **-10.7% / -6.2%** | -5.0% / -1.0% | split 有效，但分类 leaf 抵消收益 |
| `sparse_multiplicative` | -73.5% / -15.0% | -61.9% / -93.2% | **-181.3% / -122.2%** | 当前机制的明确失败边界 |
| `pure_step_control` | -3.3% / -3.3% | -0.4% / -1.0% | -3.6% / -4.0% | 完整机制无净收益 |
| `curved_control` | -6.4% / -7.1% | +4.7% / +4.7% | -1.3% / -2.0% | leaf 有局部逼近收益，但不足以抵消 split |

最重要的机制判断是：

1. **整体理念获得有边界的支持。** `regime_slope` 中 split 与 leaf 的增益可
   分解，Full 总效应在两个维度均为 3/3 seeds 同向，符合“先识别局部响应
   region，再在叶内拟合线性响应”的设计思想。
2. **线性叶具有独立贡献，但不能强化为普遍主张。** `feature_switch` 中
   SplitOnly 本身变差，加入 predictive leaf 后却稳定改善 11.3%/13.6%，这是
   最直接的 leaf 归因证据。相反，10 个 aligned scenario-dimension cells 的
   `Δleaf` CI 为 4 个正、4 个负、2 个跨 0，不能写成“大多数机制匹配任务的
   线性叶均有效”。
3. **最强总体提升不能全部归因于线性叶。** `regime_slope` 的 43.1%/53.2%
   总改善中，split 贡献为 38.9%/48.6%，leaf 是 6.5%/8.6% 的增量增强。
4. **负对照支持的是完整机制边界，而非 leaf 的严格专属性。** 两个 controls
   的 Full 均未优于 Bias，但 `curved_control` 的 leaf-only 效应仍为正，说明
   predictive leaf 也提供一般性的局部线性逼近能力。

`pred_fraction` 曲线进一步否定了统一单调剂量：

- `regime_slope` 随 p 增加而改善，在 p≈0.8 后进入平台；
- `feature_switch` 在较高覆盖率下才稳定转正；
- `regime_logit` 的 split 始终为正，但 leaf 负效应随 p 放大；
- `sparse_multiplicative` 的高 p 产生严重失稳。

Aggregation 同样存在明显交互。Mean 相对 Greedy 没有统一优势，但可部分恢复
`sparse_multiplicative` 的 SplitOnly 路径；Median 在 Full 中有 11/14 个
cells 的 paired CI 完全为负，在 SplitOnly 中为 13/14，不能作为稳健替代。

因此，正式消融将 synthetic 主 benchmark 的相关性模式推进为更直接的组件证据，
同时把论文主张收缩为：

> MetaGB 的整体价值来自结构划分与叶内预测的可分解协同。Predictive-aware
> split 在 regime-sensitive 结构中可贡献主要增益，predictive leaf 在
> feature-switch 和部分局部线性回归结构中提供独立表达能力；分类 logit、
> 稀疏乘法、高 predictive-feature 覆盖率和 Median aggregation 则构成必须
> 主动报告的适用边界。

## 5. Timeseries 续跑更新

### 5.1 Coverage

理论矩阵为：

`7 datasets × 2 seq_len × 3 horizons × 4 models = 168 cells`

| 模型 | 完成 | Pending | seq_len=24 | seq_len=48 |
| --- | ---: | ---: | ---: | ---: |
| MetaGB | 9/42 | 33 | 9/21 | 0/21 |
| XGBoost | 21/42 | 21 | 21/21 | 0/21 |
| LightGBM | 21/42 | 21 | 21/21 | 0/21 |
| CatBoost | 21/42 | 21 | 21/21 | 0/21 |
| **合计** | **72/168** | **96** | **72/84** | **0/84** |

最新 `summary.csv` 有 72 条唯一记录，全部 `status=ok`；状态文件为
72 completed、96 pending、0 failed。静态状态没有 `running` 标记，因此本报告
只写“尚未完成”，不推断这些任务此刻是否仍在运行。

### 5.2 相对旧 68 行快照的新增结果

旧 68 行全部保留，新增 4 个 MetaGB cells，并且四个均击败该 cell 上的最强
常规树 baseline：

| 新增 cell | MetaGB RMSE | 最强 baseline | Baseline RMSE | 相对改善 | MetaGB elapsed |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTm1, h=1 | 0.036244 | LightGBM | 0.037507 | +3.37% | 167.08 h |
| ETTm2, h=1 | 0.021906 | LightGBM | 0.027295 | +19.75% | 162.21 h |
| ILI, h=12 | 1.091698 | LightGBM | 1.962222 | +44.36% | 152.68 h |
| exchange_rate, h=6 | 0.054827 | XGBoost | 0.117778 | +53.45% | 191.85 h |

MetaGB 的四模型 paired coverage 从 5 增至 9，战胜 strongest baseline 从
3/5 增至 7/9。

### 5.3 四模型 common coverage

相对改善定义为
`(baseline RMSE - MetaGB RMSE) / baseline RMSE`。

| 对照 | MetaGB 胜/负 | 平均相对改善 | 中位相对改善 | MetaGB/对照 elapsed 中位数 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 7/2 | +31.08% | +43.65% | 89.65× |
| LightGBM | 7/2 | +34.96% | +44.36% | 123.90× |
| CatBoost | 9/0 | +44.00% | +44.69% | 78.47× |
| 每 cell 最强常规树 | **7/2** | **+28.02%** | **+26.50%** | **100.31×** |

MetaGB 的 9-cell 中位 `task_elapsed_sec` 为 75.30 小时。这里是包含 Random300
搜索在内的端到端任务耗时，不是纯训练或纯推理时间。

按数据集看：

| Dataset | 配对 cells | MetaGB 胜/负 | 相对 strongest baseline 平均改善 |
| --- | ---: | ---: | ---: |
| ETTh1 | 1 | 0/1 | -3.18% |
| ETTh2 | 1 | 0/1 | -18.88% |
| ETTm1 | 1 | 1/0 | +3.37% |
| ETTm2 | 1 | 1/0 | +19.75% |
| ILI | 3 | 3/0 | +57.07% |
| exchange_rate | 2 | 2/0 | +39.97% |
| Weather | 0 | — | — |

三种常规树 baseline 的完整 21-cell 结论保持不变：

- LightGBM：13 wins；
- XGBoost：7 wins；
- CatBoost：1 win。

### 5.4 更新后的 timeseries 结论

可以写：

> 在相同 fixed-split、seed=42、`seq_len=24` 和 300-candidate 配置下，
> MetaGB 已完成的 9 个四模型配对任务中，有 7 个优于该任务上的最强
> LightGBM/XGBoost/CatBoost baseline；相对最强 baseline 的 RMSE 平均和
> 中位降低分别为 28.02% 和 26.50%。收益在 ILI 三个预测跨度、
> exchange-rate 的 h=1/6 以及 ETTm1/ETTm2 短期预测上出现，但在
> ETTh1/ETTh2 的 h=1 上仍分别落后 3.18% 和 18.88%。由于 MetaGB 仅覆盖
> `seq_len=24` 的 9/21 个任务，且端到端任务耗时中位数约为 strongest
> baseline 的 100 倍，这些结果应解释为明显增强但仍具有数据集选择性的
> targeted evidence，而不是完整时间序列 benchmark。

不能写：

- “MetaGB 在 timeseries 上整体领先”；
- “MetaGB 在中长 horizon 普遍更强”——h=6 只有 2 个配对 cells，h=12 只有 1 个；
- “MetaGB 已完成 9/21 的全部计划”——完整计划还包括 `seq_len=48`，
  MetaGB 实际为 9/42；
- 隐去约 100× 的端到端计算代价。

## 6. 面向论文的统一结论

### 6.1 主张层级

| 主张 | 最新证据 | 建议强度 |
| --- | --- | --- |
| 结构划分与线性叶在部分局部线性结构中协同有效 | 正式消融 `regime_slope`：Δtotal=+43.1%/+53.2%，两个维度均 3/3 seeds 同向 | **主文强主张，但限定结构** |
| 线性叶具有可分离的独立表达贡献 | `feature_switch`：Δleaf=+11.3%/+13.6%，两个维度均 3/3 seeds 同向 | **主文中等偏强主张** |
| 线性叶在所有机制匹配任务中普遍有效 | aligned cells 的 Δleaf CI 为 4 正/4 负/2 跨零；`regime_logit`、`sparse_multiplicative` 明确失败 | **证据反对，不应主张** |
| MetaGB 在 synthetic 机制 benchmark 上有选择性优势 | 10/20 维三模型 synthetic：28/38 cells 最优；负对照 0/6 | **主文强主张** |
| MetaGB 相对 XGBoost 是一种有条件收益的结构扩展 | Synthetic 31/1/6；真实 tabular 结果异质 | **主文中等主张** |
| MetaGB 能广泛击败强 tabular baseline | Strongest baseline vs MetaGB = 42/2/2 | **证据反对，不应主张** |
| MetaGB 在部分 timeseries 数据结构上有较大收益 | 9 个配对 cells 中 7 胜，平均 RMSE 降低 28.02% | **补充/扩展主张** |
| MetaGB 是高效的 timeseries 方法 | 端到端耗时中位约 100× strongest baseline | **证据反对，不应主张** |
| dim50 仍保持机制优势 | 已完成的 11 个机制匹配 cells 为 11/0/0 | **初步主张，必须附 coverage** |

### 6.2 推荐的主文组织

1. **机制 benchmark 表**：使用完整 10/20 维 XGBoost、LightGBM、MetaGB
   三模型表，并保留
   negative controls。主指标为逐 family W/T/L、平均 rank 和代表性幅度。
2. **正式消融表**：报告
   `XGB / Bias / SplitOnly p=1 / Full p=1` 以及
   `Δsplit / Δleaf / Δtotal`；配套展示 `regime_slope`、`feature_switch`、
   `regime_logit`、`sparse_multiplicative` 和 controls 的双 p 曲线。
3. **真实 tabular 强 baseline 表**：报告各实际模型的平均 rank/最佳次数；
   `best available baseline` 只在分析文字或诊断列中出现，不作为模型行。
4. **MetaGB vs XGBoost 表**：把它明确标为结构扩展的 family-level comparison，
   不暗示 XGBoost 代表 strongest tabular baseline。
5. **Timeseries**：作为 targeted extension 或附录实验；同时报告 coverage、
   负向 ETTh cases 和约 100× 端到端耗时。
6. **dim50**：在补齐 negative controls、sparse 和 noise families 前，只放在
   ongoing/partial evidence 或附录 coverage 表中。

### 6.3 可直接放在 Discussion 的总表述

> The expanded evidence sharpens, rather than broadens, the scope of MetaGB's
> advantage. On mechanism-controlled synthetic tasks, MetaGB remains the
> strongest of XGBoost, LightGBM, and MetaGB in 28 of 38 fully paired
> 10/20-dimensional cells, while losing all six negative-control cells to
> LightGBM. A paired fixed-backbone ablation further localizes this evidence:
> on regime-slope tasks, the full mechanism improves over the bias-only
> updater by 43.1%/53.2%, primarily through predictive-aware splitting,
> whereas predictive leaves independently improve feature-switch tasks by
> 11.3%/13.6%. However, predictive leaves reverse the split gain on
> regime-logit and fail strongly on sparse-multiplicative tasks. The evidence
> therefore supports a decomposable but bounded local-linear-in-leaves
> inductive bias, rather than a universally beneficial linear-leaf module.
> On real tabular data, however, the
> strongest available baseline exceeds MetaGB in 42 of 46 aligned cells,
> indicating selective rather than universal transfer. Time-series results
> provide additional targeted evidence—MetaGB wins 7 of 9 currently paired
> cells against the strongest conventional tree—but remain incomplete and
> computationally expensive, with roughly 100× median end-to-end task time.

对应中文：

> 扩充后的证据收紧了 MetaGB 优势的适用范围，而不是扩大了普遍性。在机制可控
> 的 synthetic 任务上，MetaGB 在 38 个完整对齐的 10/20 维 cells 中有 28 个
> 优于 XGBoost 和 LightGBM，同时在 6 个负对照上全部落后于 LightGBM；这种
> 随机制匹配出现和消失的模式支持 local-linear-in-leaves 归纳偏置。正式
> fixed-backbone 配对消融进一步定位了该证据：`regime_slope` 上完整机制相对
> Bias 改善 43.1%/53.2%，且主要由 predictive-aware split 贡献；
> `feature_switch` 上 predictive leaf 则独立改善 11.3%/13.6%。但是，
> predictive leaf 在 `regime_logit` 中抵消 split 收益，并在
> `sparse_multiplicative` 中明显失败。因此，数据支持的是可分解但有边界的
> 局部线性叶归纳偏置，而不是普遍有益的线性叶模块。在真实 tabular 数据上，
> strongest available baseline 在 46 个对齐 cells 中有 42 个
> 超过 MetaGB，说明方法迁移是选择性的而不是普遍的。时间序列结果提供了额外
> 的定向证据：当前 9 个四模型配对 cells 中 MetaGB 有 7 个获胜，但覆盖仍不
> 完整，且端到端任务耗时中位数约为最强常规树的 100 倍。

## 7. 统计解释与 11 类误区检查

除正式 synthetic 消融外，本报告没有发现可报告的 p-value、置信区间或标准化
效应量，因此其他百分比、胜负数和 rank 均为描述性统计。正式消融提供了基于
3 个 seed-level 配对点的 10,000 次 bootstrap CI，但该 CI 仅适合描述大效应
和方向一致性，不构成充分的确认性显著性证据。整体置信等级仍为 `CAUTION`。

- Fallacy Scan Coverage: **11/11 checked**

| 误区 | 状态 | 本报告中的处理 |
| --- | --- | --- |
| Simpson's paradox | CAUTION | 总体计数按 task、family、dimension、dataset/horizon 分层；不得用跨协议总数覆盖分层反转 |
| Ecological fallacy | NOTE | dataset/cell 级结论不外推为单样本或所有同类数据集的个体结论 |
| Berkson's paradox | CAUTION | benchmark 与完成任务均为选择性样本，不视为总体任务分布的随机抽样 |
| Collider bias | NOTE | 未做控制变量回归；当前不适用，但不把事后按完成状态筛选解释成因果效应 |
| Base-rate neglect | CAUTION | binary 主指标为 AUC，不据此声称校准、PPV/NPV 或真实部署收益 |
| Regression to the mean | NOTE | 无按极端结果选组的 pre/post 设计；当前未触发 |
| Survivorship bias | CAUTION | timeseries、dim50 和 RealMLP 有未完成 cells；所有结论显式报告 coverage，禁止只报完成的正向子集为全局结果 |
| Look-elsewhere effect | CAUTION | 比较 cells 很多且未做多重比较校正；正式消融的逐 cell bootstrap CI 也不使用显著性星号 |
| Garden of forking paths | CAUTION | 存在多协议、两种预处理和逐 cell best envelope；固定 tie/过滤/聚合规则并分协议报告 |
| Correlation ≠ causation | CAUTION | 单 seed benchmark 提供机制一致性模式；3-seed fixed-backbone 消融加强组件归因，但仍不构成对真实数据收益的普遍因果证明 |
| Reverse causality | NOTE | 不是观察性变量关系研究，当前不适用 |

## 8. 仍需保留的限制

- Synthetic 10/20 维只有一个 data seed；5 outer folds 不能替代跨 seed 稳健性。
- 正式 synthetic 消融包含 3 个独立 data seeds，但仍不足以支撑精细尾部概率或
  多重比较后的确认性结论；outer folds 只用于 seed-level OOF metric。
- 正式消融固定 XGBoost 优选 backbone 给所有变体，回答的是机制贡献，不是每个
  变体各自完全调优后的最佳可达性能。
- 正式消融只有一个分类 aligned scenario，且缺少逐树实际 predictive-candidate
  telemetry，不能把 `regime_logit` 或非单调 p 曲线外推为一般分类结论。
- Synthetic LightGBM 与 XGBoost/MetaGB 的搜索空间属于各模型 Random300，
  候选数对齐不等于超参数空间难度完全等价。
- LightGBM 参数空间传入了 `subsample`，但实现是否同时启用
  `subsample_freq`/`bagging_freq` 需要在定稿前核查；若默认频率为 0，
  该搜索维度可能没有实际生效。
- dim50 当前完成顺序与机制覆盖不均衡，存在明显 survivorship/selection 风险。
- 两个 synthetic 新增目录都没有最终 `summary.csv` 或 `task_status.json`，
  runner 日志也没有正常结束标志；现有 task JSON 指标完整可解析，但不应写成
  整个 orchestrator 已完成。
- Tabular strongest-baseline envelope 是 oracle diagnostic，不是可部署模型。
- RealMLP 少 1 个完整 cell；平均 rank 必须连同 `N=53` 报告。
- Timeseries 使用 fixed split、单 outer/inner fold，不能提供跨切分方差。
- Timeseries MetaGB 完成顺序不是随机抽样，当前 7/9 可能受选择性覆盖影响。
- Runtime 来自任务端到端 elapsed time，受搜索空间、机器和调度影响；它适合说明
  当前计算负担，不等同于纯算法训练复杂度或推理吞吐。

## 9. 权威证据路径

### 0716 增量

- `tab_otherBaselinemodel0716/logs/task_status.json`
- `tab_otherBaselinemodel0716/logs/tasks/*.log`
- `synbaseline_lgbm0716/synthetic_baseline_random300_seed42_20260718_052755/tasks/*.json`
- `synbaseline_lgbm0716/synthetic_baseline_random300_seed42_20260718_052755/protocol.json`
- `syn102050part0716/synthetic_main_random300_seed42/tasks/*.json`
- `syn102050part0716/synthetic_main_random300_seed42/protocol.json`
- `metagb_ablation_syn/reanalysis/syn_formal_analysis/completeness_audit.json`
- `metagb_ablation_syn/reanalysis/syn_formal_analysis/unit_results.csv`
- `metagb_ablation_syn/reanalysis/syn_formal_analysis/main_effects.csv`
- `metagb_ablation_syn/reanalysis/syn_formal_analysis/pred_effects.csv`
- `metagb_ablation_syn/reanalysis/syn_formal_analysis/scoreagg_effects.csv`
- `metagb_ablation_syn/ABLATION_RESULT_SUMMARY_ZH.md`
- `ts0716/task_status.json`
- `ts0716/ts_tree_tiered_20260708_153501__seed42__non_pems/summary.csv`

### 旧结果与对齐参考

- `../datacollection0702/datacollection0707_baselinemodel_random300/tasks/*.log`
- `../datacollection0702/tabular_drop_20260628_030502/tasks/*.log`
- `../datacollection0702/tabular_onehot_20260628_025442/tasks/*.log`
- `../datacollection0702/synthetic_main1020full_random300/tasks/*.json`
- `../datacollection0702/datacollection0707_ts_tree_seed42__non_pems/summary.csv`
- `../datacollection0702/analysis_aaai_0702/mgb_vs_xgb_neurips_analysis_0702.zh.md`
- `../datacollection0702/analysis_aaai_0702/timeseries_and_random300_baseline_analysis_20260714.md`
