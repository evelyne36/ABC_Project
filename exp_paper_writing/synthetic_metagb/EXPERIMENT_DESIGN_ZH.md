# Synthetic MetaGB 主实验设计

## 1. 主问题与比较模型

主实验回答：

> 在条件切分、叶内线性响应、条件/预测特征分离和稀疏交互结构下，
> MetaGB 相比标准 XGBoost 的优势是否稳定，并如何随特征维度变化？

主实验只比较：

- `xgb_cpu`
- `current_metagb`

`current_metagb_pred0` 不进入主实验。它将在独立消融实验中用于验证：
关闭 predictive-feature leaf 后，MetaGB updater 是否退化到接近常数叶
XGBoost 的表现。

## 2. 统一实验轴

### 2.1 特征维度

每个 A-E 基础 case 均展开三个子 case：

| 子 case | 总特征数 |
| --- | ---: |
| `dim10` | 10 |
| `dim50` | 50 |
| `dim100` | 100 |

生成器先创建结构所需的有效特征，再补充独立无关特征：

```text
x_noise ~ Normal(0, 1)
```

### 2.2 高斯目标噪声

所有回归 case 的默认目标噪声均为高斯噪声：

```text
epsilon ~ Normal(0, 0.15^2)
```

分类 case 在分类 margin/logit 中加入高斯噪声：

```text
epsilon ~ Normal(0, 0.05^2)
```

噪声强度不在每个 case 内重复展开。独立噪声压力 case
`noise_regime_slope` 使用更强高斯噪声 `sigma=0.75`，并只展开三个维度。

### 2.3 首轮协议

- `data_seed=42`
- `search_seed=42`
- 5 outer folds × 4 inner folds
- 每模型 300 个 Random300 候选
- 15 个独立任务并行：`task_workers=15, n_jobs=1`
- 单 seed 仅用于结构趋势和容量验证，不声称统计显著性

### 2.4 Framework Smoke 初步观察

已完成的 framework smoke 只运行每个基础 case 的 `dim10` 代表 cell：

```text
protocol = synthetic_main_framework_smoke
19 cells
76 / 76 tasks
2,280 / 2,280 candidate rows
models = xgb_cpu, current_metagb
```

该 smoke 用于验证生成器、nested CV、候选公平性、分片恢复和输出格式。
它只有一个 `data_seed=42`，并且样本数低于正式 Random300，因此只能作为
初步方向观察，不能作为显著性或最终优势结论。

Type 级初步表现：

| Type | Smoke 观察 | 解释 |
| --- | --- | --- |
| A：XOR 与乘法交互 | 4/4 个 dim10 cells 中 **MetaGB 优于 XGB** | 交互结构符合“先切分条件，再在叶内使用预测特征”的设计预期 |
| B：Linear-in-Leaves | 4/5 个 dim10 cells 中 **MetaGB 优于 XGB** | 区域依赖斜率、特征切换和 logit 斜率符合预期；`global_linear` 在小样本 smoke 中 XGB 更好 |
| C：Conditional-Predictive Separation | 3/3 个 dim10 cells 中 **MetaGB 优于 XGB** | one-hot 条件区域与连续预测特征分离时，MetaGB 初步表现稳定 |
| D：Sparse Interaction | 3/3 个 dim10 cells 中 **MetaGB 优于 XGB** | 少量有效特征与无关特征混合时，MetaGB 仍能利用局部线性响应 |
| E：负对照 | 2/3 个 dim10 cells 中 XGB 优于 MetaGB | `curved_control` 和 `additive_nonlinear_control` 按预期偏向 XGB；`pure_step_control` 中 **MetaGB 小幅领先**，需在正式 Random300 中复查 |
| F：高斯噪声压力 | `noise_regime_slope__dim10` 中 **MetaGB 优于 XGB** | 在高斯噪声加强的区域斜率结构下，**MetaGB 仍保持初步优势** |

Case 级 smoke 结果：

| Case | 结构内涵 | Smoke 表现 |
| --- | --- | --- |
| `xor_2way` / `xor_classification` | `sign(x1) * x2` 决定二分类 margin；切分 `x1` 后 `x2` 成为叶内线性分类特征 | **MetaGB AUC 高于 XGB，优势约 +0.139** |
| `xor_3way` | `sign(x1*x2) * x3` 决定二分类 margin；交互阶数更高 | **MetaGB AUC 高于 XGB，优势约 +0.266** |
| `multiplicative_2way` / `multiplicative` | `y = 4*x1*x2 + epsilon`；树近似一个乘法因子，叶内斜率表达另一个因子 | **MetaGB RMSE 低于 XGB，相对改善约 42.3%** |
| `conditional_multiplicative` | gate 区域决定乘法项 `x1*x2` 的系数和符号 | **MetaGB RMSE 低于 XGB，相对改善约 19.5%** |
| `global_linear` | 全局线性关系 `y = 3*x + epsilon` | Smoke 中 XGB RMSE 更低；说明低样本/当前候选下全局线性不一定自动转化为 MetaGB 优势 |
| `regime_slope_2` / `regime_slope` | 两个 gate 区域分别使用相反斜率 | **MetaGB RMSE 低于 XGB，相对改善约 50.7%** |
| `regime_slope_multi` | 四个区域使用不同斜率 | **MetaGB RMSE 低于 XGB，相对改善约 20.3%** |
| `feature_switch` | gate 决定使用哪个预测特征 | **MetaGB RMSE 低于 XGB，相对改善约 27.8%** |
| `regime_logit` | gate 决定 predictive feature 的 logit 系数符号和大小 | **MetaGB AUC 高于 XGB，优势约 +0.217** |
| `categorical_regime_slope` | one-hot 类别决定连续变量斜率 | **MetaGB RMSE 低于 XGB，相对改善约 29.7%** |
| `categorical_feature_switch` | one-hot 类别决定使用哪个连续预测特征 | **MetaGB RMSE 低于 XGB，相对改善约 24.4%** |
| `mixed_gate_regime` | 类别 gate 与连续 gate 共同定义区域斜率 | **MetaGB RMSE 低于 XGB，相对改善约 37.3%** |
| `sparse_multiplicative` | 少量有效特征构成乘法交互，其余为无关高斯特征 | **MetaGB RMSE 低于 XGB，相对改善约 46.7%** |
| `sparse_regime_slope` | 少量 gate/predictive 特征构成区域斜率 | **MetaGB RMSE 低于 XGB，相对改善约 55.1%** |
| `sparse_feature_switch` | 少量有效特征控制区域和预测特征切换 | **MetaGB RMSE 低于 XGB，相对改善约 28.4%** |
| `pure_step_control` | 分段常数函数，理论上常数叶 XGB 应较合适 | Smoke 中 **MetaGB 小幅领先**，需在正式 Random300 和更多维度中复查 |
| `curved_control` | 光滑二次曲线，不直接符合叶内单特征线性响应 | XGB RMSE 明显低于 MetaGB，符合负对照预期 |
| `additive_nonlinear_control` | 多变量加性非线性函数，不强调区域内单特征线性响应 | XGB RMSE 低于 MetaGB，符合负对照预期 |
| `noise_regime_slope` | `regime_slope_2` 结构加更强高斯噪声 | **MetaGB RMSE 低于 XGB，相对改善约 39.7%** |

## 3. Type A：XOR 与乘法交互

验证 MetaGB 是否能利用“条件切分 + 叶内预测特征”更高效地表达交互。

| Case | 任务 | 数据结构 | 设计目的 |
| --- | --- | --- | --- |
| `xor_2way` | 二分类 | `margin = sign(x1) * x2 + epsilon` | 一次条件切分后，`x2` 可作为叶内线性分类特征 |
| `xor_3way` | 二分类 | `margin = sign(x1*x2) * x3 + epsilon` | 提高交互阶数，检验 MetaGB 优势是否衰减 |
| `multiplicative_2way` | 回归 | `y = 4*x1*x2 + epsilon` | 树近似一个因子，叶内斜率表达另一个因子 |
| `conditional_multiplicative` | 回归 | gate 区域决定 `x1*x2` 的系数与符号 | 同时要求区域识别和区域内乘法响应 |

重点分析：

- 交互阶数提高后 MetaGB 优势是否稳定。
- 最佳候选的深度、树数和训练时间是否显示结构效率。

## 4. Type B：Linear-in-Leaves

这是最直接对应 MetaGB 架构假设的数据类型。

| Case | 任务 | 数据结构 | 设计目的 |
| --- | --- | --- | --- |
| `global_linear` | 回归 | `y = 3*x + epsilon` | 检验全局线性信号 |
| `regime_slope_2` | 回归 | 两个 gate 区域分别使用 `-3*x` 与 `3*x` | 检验最简单的区域依赖斜率 |
| `regime_slope_multi` | 回归 | 4 个区域使用不同斜率 | 检验区域数增加后的稳定性 |
| `feature_switch` | 回归 | gate 决定使用哪个预测特征 | 检验区域内预测特征切换 |
| `regime_logit` | 二分类 | gate 决定 predictive feature 的 logit 斜率 | 检验分类中的区域依赖线性响应 |

重点分析：

- 多区域结构是否仍保持优势。
- 最佳候选是否倾向更浅树或更少树。

## 5. Type C：Conditional-Predictive Separation

模拟类别或状态变量负责定义区域，连续变量负责区域内预测。

| Case | 任务 | 数据结构 | 设计目的 |
| --- | --- | --- | --- |
| `categorical_regime_slope` | 回归 | one-hot 类别决定连续变量斜率 | 明确分离条件变量与预测变量 |
| `categorical_feature_switch` | 回归 | one-hot 类别决定使用哪个连续预测特征 | 检验类别区域中的预测特征选择 |
| `mixed_gate_regime` | 回归 | 类别 gate 与连续 gate 共同决定斜率 | 检验混合条件区域 |

类别变量统一 one-hot 编码，避免将结果与 XGBoost categorical 实现差异
混淆。

当前 MetaGB 没有显式 conditional/predictive feature mask。主实验检验
当前采样机制能否自动发现这种分离，而不是 oracle 特征角色分配。

## 6. Type D：High-Dimensional Sparse Interaction

验证存在大量无关特征时，MetaGB 是否仍能发现少量有效交互。

| Case | 任务 | 数据结构 | 设计目的 |
| --- | --- | --- | --- |
| `sparse_multiplicative` | 回归 | 5 个有效特征中少量特征构成乘法交互 | 检验稀疏乘法信号 |
| `sparse_regime_slope` | 回归 | 少量 gate/predictive 特征构成区域斜率 | 检验稀疏区域线性信号 |
| `sparse_feature_switch` | 回归 | 少量特征控制区域和预测特征切换 | 检验稀疏特征切换 |

额外记录：

- 最佳 `pred_feature_fraction`。
- 性能随总维度 `10 → 50 → 100` 的变化。
- 训练时间、预测时间和性能收益比。

## 7. Type E：负对照

用于证明实验没有只选择 MetaGB 容易获胜的数据。

| Case | 任务 | 数据结构 | 预期 |
| --- | --- | --- | --- |
| `pure_step_control` | 回归 | 分段常数函数 | XGB 占优或持平 |
| `curved_control` | 回归 | 光滑二次曲线 | XGB 占优或持平 |
| `additive_nonlinear_control` | 回归 | 多变量加性非线性函数 | 不应出现明显 MetaGB 结构优势 |

## 8. 独立高斯噪声压力 Case

`noise_regime_slope` 使用 `regime_slope_2` 的结构和更强高斯目标噪声：

```text
y = slope(gate) * predictive + epsilon
epsilon ~ Normal(0, 0.75^2)
```

仅展开：

- `noise_regime_slope__dim10`
- `noise_regime_slope__dim50`
- `noise_regime_slope__dim100`

它用于回答高斯噪声与特征维度共同增加时，MetaGB 优势是否仍存在。

## 9. 实验矩阵与计算量

A-E 共 18 个基础 case，每个展开三个维度：

```text
18 × 3 = 54 个结构 cells
```

再加入独立高斯噪声 case 的三个维度：

```text
54 + 3 = 57 个主实验 cells
```

单 seed Random300：

```text
57 cells × 1 seed × 5 outer folds × 2 models = 570 tasks
570 tasks × 300 candidates = 171,000 个候选评估
171,000 × 4 inner folds + 570 = 684,570 次模型拟合
```

## 10. 主实验输出

### Cell 级

- MetaGB 与 XGB 的 outer-fold 平均 RMSE/AUC。
- 相对性能改善、outer-fold 胜率。
- 最佳候选参数、`pred_feature_fraction`。
- 训练时间与预测时间。

### Case 级

- 维度从 10、50 到 100 时的性能趋势。
- MetaGB 优势是否跨三个维度稳定。

### Type 级

- 各类型中 MetaGB/XGB 获胜的 cell 数量。
- 平均相对改善和最稳定优势结构。
- 负对照是否符合预期。

首轮只有一个 data seed，所有输出均标记为探索性，不进行 Wilcoxon 或
显著性声明。

## 11. Pred0 消融分支

Pred0 消融不属于本主实验。后续选取代表 cells：

- 常数叶对照：`pure_step_control__dim10/dim100`、`curved_control__dim50`
- MetaGB 优势结构：`regime_slope_2__dim10`、`feature_switch__dim50`、
  `sparse_multiplicative__dim100`

比较 `xgb_cpu`、`current_metagb_pred0`、`current_metagb`，验证：

```text
负对照中：pred0 ≈ XGB
优势结构中：MetaGB > pred0 ≈ XGB
```

Pred0 与 XGB 的“接近”应通过预定义等价区间和 paired bootstrap/TOST
验证，而不是仅依赖“不显著”。
