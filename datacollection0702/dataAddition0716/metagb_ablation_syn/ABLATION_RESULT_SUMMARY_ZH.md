# MetaGB 正式 Synthetic 消融结果总结

> 目的：依据 `metagb_ablation/EXPERIMENT_DESIGN_ZH.md` 的原始设计思想，
> 总结 `syn_formal_formal16_v1` 的完整结果，为论文正文、消融表格和讨论部分
> 提供可审计的写作依据。
>
> 结论版本：2026-07-18；分析单位为 3 个独立 data seeds，outer folds 仅用于
> 构造每个 seed 的 OOF 指标，不作为独立重复。

## 1. 一句话结论

这组消融支持的是 **有明确边界的、场景依赖的机制贡献**，而不是“打开
predictive-aware split 和 predictive leaf 就会普遍提升性能”：

- `regime_slope` 给出最强且最完整的正证据：predictive-aware split 是主要
  增益来源，predictive leaf 提供额外增益，Full 相对 Bias 的平均相对改善在
  10/20 维分别为 **43.1%/53.2%**，且两个维度均为 **3/3 seeds 同向**。
- `feature_switch` 的增益主要来自 predictive leaf，而不是 split scoring：
  `Δleaf` 为 **11.3%/13.6%**，均为 3/3 seeds 同向；`Δsplit` 反而为
  **-3.5%/-7.7%**。
- `regime_logit` 出现清晰的组件冲突：split scoring 稳定改善 logloss
  **5.2%/4.9%**，但 predictive leaf 恶化 **10.7%/6.2%**，使 Full 的总效应
  转为 **-5.0%/-1.0%**。
- `sparse_multiplicative` 是明确失败边界：在 `p=1, greedy` 下 split 和 leaf
  都恶化，Full 相对 Bias 恶化 **181.3%/122.2%**。这否定了“所有机制匹配
  synthetic 场景都受益”的强表述。
- 两个负对照没有显示完整机制相对 Bias 的收益：`pure_step_control` 为
  **-3.6%/-4.0%**，`curved_control` 为 **-1.3%/-2.0%**。因此，正向结果
  不是所有 synthetic 数据上的统一平移，但机制特异性只能写成“在部分局部
  线性结构上成立”。
- `pred_fraction` 不是统一单调的剂量变量：`regime_slope` 呈上升后平台，
  `feature_switch` 随覆盖率提高而改善，`regime_logit` 和
  `sparse_multiplicative` 则因 leaf 负效应随覆盖率放大而恶化。
- score aggregation 不是无关紧要的实现细节。Mean 的作用取决于场景和 leaf
  mode；Median 在 Full 中有 11/14 个 cells 的 paired bootstrap CI 完全为负，
  在 SplitOnly 中有 13/14 个 cells 完全为负，不能作为稳健替代。

因此，论文最稳妥的核心叙事是：

> MetaGB 的机制收益可以被分解，但不同组件解决不同结构。Predictive-aware
> split 在符号翻转的局部斜率和 regime-logit 结构中有效，predictive leaf
> 在 feature-switch 与部分局部线性回归结构中提供额外表达能力；这些收益不
> 能外推到稀疏乘法、分类 leaf 或任意 aggregation。

## 2. 数据身份与完整性

离线重建分析通过全部硬审计：

| 项目 | 结果 |
| --- | ---: |
| Protocol | `formal`, version `1.3-seed3-grid5-scoreagg16` |
| Confirmatory | `True` |
| Variants per task | 16 |
| Scenarios × dimensions × seeds × outer folds | 7 × 2 × 3 × 5 |
| Tasks | 210 / 210 |
| Fold rows | 3360 / 3360 |
| Prediction rows | 2,016,000 / 2,016,000 |
| Completeness audit | Passed |

证据源：

- `reanalysis/syn_formal_analysis/completeness_audit.json`：完整性硬审计；
- `reanalysis/syn_formal_analysis/unit_results.csv`：每个
  `scenario × dimension × seed × variant` 的 OOF 指标，共 672 行；
- `reanalysis/syn_formal_analysis/main_effects.csv`：主机制配对效应；
- `reanalysis/syn_formal_analysis/pred_effects.csv`：`pred_fraction`
  配对效应；
- `reanalysis/syn_formal_analysis/scoreagg_effects.csv`：aggregation
  配对效应；
- `reanalysis/syn_formal_analysis/ANALYSIS_EVIDENCE.md`：离线分析器生成的
  完整证据表。

`syn_formal_analysis/` 与 `reanalysis/syn_formal_analysis/` 当前逐文件一致。
本文固定引用后者，以强调这些表是从已保存 task shards 离线重建所得，未重新训练。
独立地从 `unit_results.csv` 按
`(scenario, dimension, unit_id)` 重算三组配对效应后，与
`main_effects.csv`、`pred_effects.csv`、`scoreagg_effects.csv` 的最大绝对
误差分别为 `3.2e-16`、`3.2e-16`、`2.7e-15`，即仅存在浮点舍入误差。

## 3. 统计口径

所有主指标均为 loss，数值越小越好：

- 回归：RMSE；
- 分类：logloss。

对 target 方法 A 与 baseline 方法 B，正值表示 A 更好：

```text
relative improvement = (loss_B - loss_A) / |loss_B|
```

表中的百分比是三个 seed-level 配对相对改善的均值；括号中的 `k/3` 表示三个
独立 seeds 中有多少个方向为正。95% CI 是对 3 个 seed-level 配对差做 10,000
次 bootstrap 的探索性区间。由于只有 3 个 seeds，不能把“CI 未跨 0”等同于
充分的确认性显著性证据。

主效应定义：

```text
Δsplit = SplitOnly(p=1, greedy) - Bias
Δleaf  = Full(p=1, greedy) - SplitOnly(p=1, greedy)
Δtotal = Full(p=1, greedy) - Bias
```

其中“减号”表示 target 相对 baseline 的 signed improvement，不是原始 loss
直接相减。

## 4. 主消融：四类变体的机制分解

| Scenario | Dim | Δsplit | Δleaf | Δtotal | Full−XGB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `categorical_regime_slope` | 10 | -29.2% (1/3) | +15.1% (3/3) | -8.8% (2/3) | +0.1% (2/3) |
| `categorical_regime_slope` | 20 | +22.2% (3/3) | +2.4% (2/3) | +23.0% (2/3) | +15.5% (2/3) |
| `feature_switch` | 10 | -3.5% (1/3) | +11.3% (3/3) | +8.9% (2/3) | +5.6% (2/3) |
| `feature_switch` | 20 | -7.7% (0/3) | +13.6% (3/3) | +7.0% (2/3) | +4.6% (2/3) |
| `regime_logit` | 10 | +5.2% (3/3) | -10.7% (0/3) | -5.0% (0/3) | -4.7% (0/3) |
| `regime_logit` | 20 | +4.9% (3/3) | -6.2% (0/3) | -1.0% (1/3) | -1.0% (0/3) |
| `regime_slope` | 10 | +38.9% (3/3) | +6.5% (2/3) | +43.1% (3/3) | +43.6% (3/3) |
| `regime_slope` | 20 | +48.6% (3/3) | +8.6% (3/3) | +53.2% (3/3) | +48.0% (3/3) |
| `sparse_multiplicative` | 10 | -73.5% (0/3) | -61.9% (0/3) | -181.3% (0/3) | -194.1% (0/3) |
| `sparse_multiplicative` | 20 | -15.0% (0/3) | -93.2% (0/3) | -122.2% (0/3) | -128.9% (0/3) |
| `pure_step_control` | 10 | -3.3% (0/3) | -0.4% (1/3) | -3.6% (0/3) | +2.4% (2/3) |
| `pure_step_control` | 20 | -3.3% (1/3) | -1.0% (1/3) | -4.0% (1/3) | +2.5% (2/3) |
| `curved_control` | 10 | -6.4% (0/3) | +4.7% (3/3) | -1.3% (0/3) | -1.1% (0/3) |
| `curved_control` | 20 | -7.1% (0/3) | +4.7% (3/3) | -2.0% (0/3) | -1.9% (0/3) |

### 4.1 MetaGB updater 本身：Bias 与 XGB

原设计中的第一个问题是 `p=0` 是否只是 XGB 改名。利用 `unit_results.csv`
补算 `Bias vs XGB` 后：

- 14 个 scenario-dimension cells 中，平均方向为 7 正、7 负；
- paired bootstrap CI 为 3 个正、1 个负、10 个跨 0；
- 跨 cells 的相对改善中位数仅 **+0.05%**。

这说明 Bias updater 与 XGB 的确不是数值等价实现，但没有稳定的总体增益。它在
`pure_step_control` 上相对 XGB 改善约 **5.8%/6.3%**，解释了为何该负对照中
Full 可以略优于 XGB、却仍劣于 Bias。机制写作应优先使用 `Full vs Bias`，
不能仅用 `Full vs XGB` 归因 predictive components。

### 4.2 Predictive-aware split

Split 的作用高度结构化：

- 最强正效应出现在 `regime_slope`，10/20 维分别改善 **38.9%/48.6%**，
  且均为 3/3 seeds 同向；
- `regime_logit` 也稳定受益，改善 **5.2%/4.9%**，均为 3/3 seeds 同向；
- `categorical_regime_slope` 在 10 维为负、20 维为正，说明其效果受维度和
  候选覆盖影响；
- `feature_switch`、两个 controls 和 `sparse_multiplicative` 在 `p=1`
  下不受益。

因此可以写“predictive-aware split 在某些需要先识别 regime、再进行局部预测的
结构中贡献主要增益”，不能写“split scoring 是 MetaGB 的普遍增益来源”。

### 4.3 Predictive leaf

Leaf 的作用与 split 明显可分：

- `feature_switch` 是最稳定的 leaf 正例，10/20 维分别为
  **+11.3%/+13.6%**，均为 3/3 seeds 同向；
- `regime_slope` 提供 **+6.5%/+8.6%** 的附加收益；
- `categorical_regime_slope` 的 leaf 在 10 维明显有益，在 20 维接近零；
- `regime_logit` 和 `sparse_multiplicative` 中 leaf 稳定恶化，且在
  `sparse_multiplicative` 中是 p=1 失败的主要放大器；
- `curved_control` 中 leaf 单独改善约 4.7%，但不足以抵消 split 的损失，
  因而 Full 总效应仍为负。

最合适的表述是“predictive leaf 对局部线性回归表达有选择性贡献”，而不是把
该结论外推到二分类 logit 或乘法结构。

### 4.4 完整机制

在 10 个预标记为 aligned 的 scenario-dimension cells 中：

- `Δtotal` 的 CI 完全为正：2 个，均来自 `regime_slope`；
- CI 完全为负：3 个，来自 `regime_logit` dim10 和
  `sparse_multiplicative` 两个维度；
- CI 跨 0：5 个。

若只看四个主变体的平均 loss 最优者，aligned cells 中 Full、SplitOnly、XGB
分别赢得 6/10、2/10、2/10，Bias 为 0/10；四个 negative-control cells 则
全部由 Bias 最优。这一计数进一步说明 Full 有选择性优势，但不是统一赢家。

所以 Full 机制最强的可重复正证据集中在 `regime_slope`，`feature_switch`
给出方向一致性较弱但机制分解清晰的次级证据；其他 aligned 场景要求收缩或反向
报告。

## 5. `pred_fraction`：不是统一单调剂量

下表报告 `Δtotal(p) = Full(p) vs Bias`。正值越大越好：

| Scenario | Dim | p=.2 | p=.4 | p=.6 | p=.8 | p=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `categorical_regime_slope` | 10 | +22.7% | -6.3% | -21.6% | -39.0% | -8.8% |
| `categorical_regime_slope` | 20 | +14.7% | +4.1% | +6.2% | +23.8% | +23.0% |
| `feature_switch` | 10 | -0.8% | -1.3% | +4.0% | +6.3% | +8.9% |
| `feature_switch` | 20 | -7.8% | -1.6% | +3.3% | +3.9% | +7.0% |
| `regime_logit` | 10 | -0.0% | -1.4% | -3.0% | -4.1% | -5.0% |
| `regime_logit` | 20 | +7.6% | +7.9% | +7.7% | +6.1% | -1.0% |
| `regime_slope` | 10 | +31.6% | +37.8% | +42.5% | +45.2% | +43.1% |
| `regime_slope` | 20 | +37.9% | +45.5% | +50.6% | +53.7% | +53.2% |
| `sparse_multiplicative` | 10 | +4.3% | -28.4% | -67.7% | -101.9% | -181.3% |
| `sparse_multiplicative` | 20 | -23.6% | -59.7% | -72.0% | -84.1% | -122.2% |
| `pure_step_control` | 10 | +1.5% | -0.7% | -2.7% | -2.6% | -3.6% |
| `pure_step_control` | 20 | -0.0% | -1.2% | -1.8% | -3.7% | -4.0% |
| `curved_control` | 10 | -20.2% | -13.9% | -3.4% | -2.2% | -1.3% |
| `curved_control` | 20 | -14.8% | -8.7% | -5.3% | -3.4% | -2.0% |

可提炼为四类曲线：

1. **上升后平台：** `regime_slope` 从 p=.2 到 p=.8 持续增强，p=.8 到 p=1
   基本平台，说明增益不是由 p=1 单点偶然驱动。
2. **覆盖率促进 leaf 表达：** `feature_switch` 在两个维度中均从低 p 的负/近零
   逐步转正，且 p=1 最好；其 `Δleaf(p)` 同样总体随 p 上升。
3. **组件冲突随剂量放大：** `regime_logit` 的 `Δsplit(p)` 在所有 p 上为正，
   但 `Δleaf(p)` 随 p 增大而更负，导致总曲线在高 p 恶化。
4. **高覆盖率失稳：** `sparse_multiplicative` 的 SplitOnly 在 p=.2 时仍为
   **+15.2%/+5.6%**，但随 p 增大转负；leaf 在所有 p 上为负，最终产生
   p=1 的大幅崩坏。

`categorical_regime_slope` 的曲线在 10/20 维明显不同，表明 p 还与有效候选的
数量、类别展开后的特征结构交互。按设计，p=.2 在 10/20 维分别对应期望
2/4 个 predictive candidates，理论零候选概率约为 10.7%/1.15%。因此不能把
同一 p 简单解释为跨维度相同强度的机制剂量。

这些结果不支持从测试曲线选择一个“全局最佳 p”。论文应展示完整双曲线，并把
p=1 保留为预注册的强机制开关，而不是部署最优参数。

## 6. Score aggregation：Mean 有交互，Median 明显失稳

### 6.1 总体证据计数

| Effect | CI 正向 | CI 负向 | CI 跨 0 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Full Mean vs Greedy | 3 | 4 | 7 | 场景依赖，无统一优势 |
| Full Median vs Greedy | 2 | 11 | 1 | 大多数 cells 明显恶化 |
| SplitOnly Mean vs Greedy | 4 | 6 | 4 | 与 leaf mode、场景强交互 |
| SplitOnly Median vs Greedy | 1 | 13 | 0 | 几乎系统性恶化 |

### 6.2 关键交互

- 在 `regime_slope` 中，Mean 相对 Greedy 使 Full 恶化 **5.3%/6.5%**，
  SplitOnly 恶化 **9.7%/7.5%**；Greedy 对最强正例仍是更合适的默认设置。
- 在 `feature_switch` 中，Full Mean 仅小幅改善 **2.2%/1.3%**，但
  SplitOnly Mean 恶化 **6.9%/1.7%**，再次说明 aggregation 与 leaf mode
  不可混为一个主效应。
- 在 `sparse_multiplicative` 中，Mean 相对 Greedy 大幅缓解失败：
  Full 改善 **49.5%/54.3%**，SplitOnly 改善 **42.5%/31.9%**。但“相对
  Greedy 改善”不等于已经优于 Bias：
  - Full Mean 相对 Bias 仍为 **-41.6%/-1.3%**；
  - SplitOnly Mean 相对 Bias 为 **+0.3%/+21.8%**。
  这表明该场景的失败同时包含 aggregation fragility 与 predictive-leaf
  fragility，Mean 主要修复了 SplitOnly 路径。
- Median 在回归 cells 中常产生数倍 loss 放大。它只在
  `sparse_multiplicative` 的 Full 路径中相对已经很差的 Greedy 有所改善，
  但仍不能恢复为稳定正效应。

因此可以写“Greedy 的正结论并非完全由 aggregation 偶然决定”，但必须同时
承认：aggregation 对困难结构有巨大影响，尤其 Median 不是稳健替代，Mean
也不能作为跨场景默认最优策略。

## 7. 机制匹配场景与负对照的逐场景判断

| 场景 | 判断 | 写作强度 |
| --- | --- | --- |
| `regime_slope` | split 主导、leaf 增益、剂量上升后平台；最完整正证据 | 强支持 |
| `feature_switch` | split 无益，leaf 稳定有益，Full 小幅净改善 | 支持 leaf 的选择性贡献 |
| `categorical_regime_slope` | 维度与 p 高度敏感；dim20 正、dim10 非单调 | 有条件支持 |
| `regime_logit` | split 稳定有益，leaf 稳定有害，Full 不改善 | 支持 split，否定当前 leaf 泛化 |
| `sparse_multiplicative` | p=1 Greedy 明显失败；低 p/Mean 可缓解 SplitOnly | 明确失败边界 |
| `pure_step_control` | Full 相对 Bias 小幅恶化；Full 略优于 XGB 来自 Bias updater | 符合负对照预期 |
| `curved_control` | leaf 小幅正、split 更大幅负，Full 净负 | 符合完整机制负对照预期 |

负对照的结果支持“收益不是 synthetic 数据上的普遍平移”，但不能据此声称严格
的机制专属性，因为五个 aligned scenarios 中仍有一个完全失败、两个只支持部分
组件。

## 8. 证据、推断与未知项

### 8.1 直接证据

- 所有 210 个任务和 3360 条 fold rows 完整，且 16 个变体均存在；
- 所有效应在同一 scenario、dimension、seed 内严格配对；
- `regime_slope` 的 split/total/Full-vs-XGB 大效应在两个维度和三个 seeds
  中一致；
- `feature_switch` 的 leaf 正效应、`regime_logit` 的 split 正/leaf 负效应、
  `sparse_multiplicative` 的高 p 失败均在两个维度复现；
- 两个 negative controls 的 `Full vs Bias` 均无正向净效应；
- Median aggregation 在绝大多数 cells 中稳定恶化。

### 8.2 基于证据的推断

- MetaGB 不是单一机制整体起作用，而是 split scoring 与 leaf expression
  在不同结构上分别贡献、抵消或放大；
- `regime_slope` 的大幅改善最符合“先以 gate 划分 regime，再在叶内拟合局部
  线性响应”的设计假设；
- 分类 `regime_logit` 的结果提示当前 predictive leaf 与 logloss/概率建模
  路径存在适配边界，但本实验不能定位具体工程原因；
- `sparse_multiplicative` 的失败不是单纯由一种 aggregation 导致，因为 Mean
  只能部分恢复 SplitOnly，不能稳定恢复 Full。

### 8.3 未知项与结论边界

- 只有 3 个独立 seeds；bootstrap CI 适合描述大效应和方向一致性，不适合精细
  显著性或尾部概率主张；
- 没有为每个变体独立调参；这是 fixed-XGB-backbone mechanism effect，
  不是四个方法各自最佳可达性能；
- p 曲线全部使用 outer-test 结果，不能据此回填“最优 p”到主结果；
- 只有一个分类 aligned scenario，不能把 `regime_logit` 的 leaf 失败外推为
  所有分类任务都失败；
- 理论候选数量和零候选概率已保存，但没有逐树实际候选 telemetry，不能精确
  解释每条非单调曲线；
- 本实验不支持“MetaGB 普遍优于 XGBoost”或“所有机制匹配任务都受益”。

## 9. 论文可直接使用的中文表述

### 9.1 正文精简版

> 在固定 XGBoost backbone、相同数据划分与随机种子的条件下，我们将 MetaGB
> 分解为 updater、predictive-aware split 和 predictive leaf。结果表明，各
> 组件的贡献具有明显的结构依赖性。在符号翻转的 `regime_slope` 场景中，
> predictive-aware split 是主要增益来源，predictive leaf 进一步降低误差，
> 使完整模型相对 Bias 在 10/20 维分别改善 43.1% 和 53.2%，且三个独立数据
> seeds 均同向。`feature_switch` 的收益则主要来自 predictive leaf，其相对
> SplitOnly 改善 11.3% 和 13.6%。相反，在 `regime_logit` 中 split scoring
> 稳定改善 logloss，但 predictive leaf 抵消了该收益；在
> `sparse_multiplicative` 中，高 predictive-feature 覆盖率进一步放大误差。
> 两个负对照上 Full 均未优于 Bias。以上结果说明 MetaGB 的优势来自可分解但
> 有边界的机制匹配，而非对任意任务的统一性能提升。

### 9.2 剂量与 aggregation 版

> `pred_fraction` 敏感性进一步显示，机制强度并不存在统一的单调规律。
> `regime_slope` 随覆盖率增加而改善并在 p≈0.8 后进入平台，
> `feature_switch` 在较高覆盖率下才稳定获益；而 `regime_logit` 和
> `sparse_multiplicative` 中 predictive leaf 的负效应随覆盖率放大。
> Aggregation 消融同样呈现显著交互：Mean 可缓解
> `sparse_multiplicative` 的 SplitOnly 失败，但在 `regime_slope` 中弱于
> Greedy；Median 则在绝大多数 cells 中显著恶化。因而 p 和 aggregation
> 应被视为机制边界与稳健性变量，而不是可跨任务复用的固定最优选择。

### 9.3 不应使用的表述

- “MetaGB 的 split 和 leaf 在所有机制匹配任务中均带来增益”；
- “p 越大效果越好”；
- “p=1 是最佳部署配置”；
- “Median aggregation 更稳健”；
- “Synthetic 结果证明 MetaGB 普遍优于 XGBoost”；
- 把 5 个 outer folds 当成 5 个独立重复或显著性样本。

## 10. 建议的论文呈现

1. 主表保留 `XGB / Bias / SplitOnly p=1 / Full p=1` 四行，并在旁表报告
   `Δsplit / Δleaf / Δtotal / Full−XGB`。
2. 剂量图同时画 SplitOnly 与 Full，重点展示：
   - `regime_slope`：正向平台；
   - `feature_switch`：leaf coverage 收益；
   - `regime_logit`：split 与 leaf 冲突；
   - `sparse_multiplicative`：高 p 失稳；
   - 两个 controls：完整机制无净收益。
3. 每个代表 cell 展示三个 seed-level paired points；不要只放 bootstrap
   error bar。
4. Score aggregation 使用 `SplitOnly/Full × greedy/mean/median` 六格表，
   单独标出 Median 的系统性失败和 Mean 在 sparse 场景中的部分恢复。

## 11. 综合判断

按原 Claim–Evidence Matrix：

| 预注册问题 | 结果 |
| --- | --- |
| updater 与 XGB 是否等价 | 数值上不等价，但无稳定总体优势 |
| predictive-aware split 是否有独立贡献 | 有；最强于 `regime_slope`，在 `regime_logit` 也稳定为正 |
| predictive leaf 是否有独立贡献 | 有但选择性强；`feature_switch`/`regime_slope` 为正，分类与 sparse 场景为负 |
| p 是否呈剂量响应 | 仅部分场景成立；总体非单调且存在反转 |
| Greedy 是否只是偶然选择 | 不是；其强正例可复现，但 Mean 在 sparse 场景显示明显交互 |
| 是否存在机制边界 | 是；controls 无净收益，`regime_logit` 和 `sparse_multiplicative` 暴露组件失败 |

最终应把 Synthetic 结论定位为：

> **部分支持、需要收缩的机制验证。** 最可信的贡献是 regime-slope 上由 split
> 主导、leaf 增强的强效应，以及 feature-switch 上 leaf 的独立贡献；同时，
> 分类 leaf、稀疏乘法、高覆盖率和 Median aggregation 构成必须主动报告的
> 失败边界。
