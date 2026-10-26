# T4 · N4b 结构诊断表（为什么 CPU 崩、kin8nm 赢）

> 对应：Rev1-3「CPU 为何大幅退化」、Rev3-2「额外成本何时值得」、Rev5-2。
> 状态：**20/20 selected refit 完成**（10 CGB + 10 XGB，含叶结构导出），待聚合。
> 纪律：本表只能写**相关**，不能写因果（"可能原因"必须在措辞上限定）。

## 1. 对照设计

| 维度 | 负例 `cpu_activity`（CGB 3.934 vs XGB 2.187，+79.92%，胜折 0/5） | 正例 `kin8nm`（CGB 0.1003 vs XGB 0.1125，−10.90%，胜折 5/5） |
|---|---|---|
| 特征数 / one-hot 比例 | [待填] | [待填] |
| 叶内候选特征数 P | [待填] | [待填] |
| 被选中的叶特征（频次 top-5） | [待填] | [待填] |
| 叶样本量中位数（叶子数/样本数） | [待填] | [待填] |
| 叶内线性拟合的 shrinkage | [待填] | [待填] |
| 局部残差形态（符号偏移/异方差） | [待填] | [待填] |
| 与 XGB 的叶划分差异（split 命中率） | [待填] | [待填] |

## 2. 中文内部表（待填，按数据集）

| 数据集 | 方法 | 叶数 | 叶样本中位 | 单特征叶占比 | 选中特征重复率 | shrinkage 均值 | test RMSE |
|---|---|---|---|---|---|---|---|
| cpu_activity | CrossGB | | | | | | 3.9344 |
| cpu_activity | XGBoost | | | | | | 2.1868 |
| kin8nm | CrossGB | | | | | | 0.1003 |
| kin8nm | XGBoost | | | | | | 0.1125 |

（另可加 energy_efficiency 作中间态。）

## 3. English appendix draft（骨架）

**Table S4.** Post-hoc structural diagnostics of the selected configurations for the worst case (cpu_activity) and the best case (kin8nm): per-leaf candidate count, selected predictive feature, leaf support, linear-coefficient shrinkage, and agreement of split points with XGBoost.

*Notes.* These statistics are **descriptive post-hoc measurements on the selected models**; they identify candidate mechanisms (e.g. predictive features that are near-constant within leaves of a heavily one-hot-encoded design matrix) but do not establish a causal account of the performance gap.

## 4. 填数要点

- 若诊断指向"one-hot 下叶内几乎无变化 ⇒ 线性项的估计方差吃掉增益"，就写成**可检验的假设**并说明如何在修改稿中验证，不能写成已证结论。
- 若诊断指向实现细节（如 shrinkage 步长、叶刷新频率），要区分"我们的实现"与"方法的固有性质"，前者可直接修，后者要承认。
