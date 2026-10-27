# T4 · N4b 结构诊断表（为什么 CPU 崩、kin8nm 赢）

> 对应：Rev1-3「CPU 为何大幅退化」、Rev3-2「额外成本何时值得」、Rev5-2。
> 状态：**✅ 全部补齐**（10-27 17:05 重抓 `leaf_structure.csv`，20/20 折齐全；归档 `证据_远端/N4_leaf_structure.tar.gz`（14 MB，被 `*.tar.gz` 忽略，未入 git））。
> 纪律：本表只能写**相关**，不能写因果（"可能原因"必须在措辞上限定）。

## 1. 终测指标（20/20）

| 数据集 | 方法 | 五折 test RMSE | 均值 | 逐折胜负 |
|---|---|---|---|---|
| cpu_activity | CrossGB | 3.3459 / 3.5649 / 3.6084 / 4.0918 / 5.0612 | **3.934443** | 0/5 |
| cpu_activity | XGBoost | 2.2199 / 2.1418 / 2.0919 / 2.3359 / 2.1446 | **2.186817** | — |
| kin8nm | CrossGB | 0.1035 / 0.0980 / 0.1031 / 0.0985 / 0.0984 | **0.100273** | 5/5 |
| kin8nm | XGBoost | 0.1155 / 0.1117 / 0.1152 / 0.1085 / 0.1118 | **0.112534** | — |

逐折配对差 CGB−XGB：cpu = [+1.1260, +1.4231, +1.5165, +1.7559, +2.9166]（五折全正）；kin = [−0.0120, −0.0137, −0.0122, −0.0100, −0.0134]（五折全负）。
与归档一致：本次 refit 复现归档五折均值（<1e-6），说明重放链路与原文同一条，N4a 结论可直接引用。

## 2. 叶结构诊断（五折齐全）

| 指标 | cpu_activity · CrossGB（负例） | kin8nm · CrossGB（正例） | cpu · XGBoost | kin · XGBoost |
|---|---|---|---|---|
| 覆盖折数 | 5/5 | 5/5 | 5/5 | 5/5 |
| 叶总数 | 88,303 | 455,916 | 91,263 | 262,038 |
| 常数叶占比 | 24.4% | 21.5% | 100% | 100% |
| 单特征预测叶占比 | 75.6% | 78.5% | —（无预测叶） | —（无预测叶） |
| 其中选中**路径外**特征 | **68.7%** | **19.3%** | — | — |
| 叶内训练样本中位数 | 11 | 8 | 15 | 10 |
| 选中特征 top3 | 20(19,921)、7(5,583)、18(3,995) | 5(49,402)、6(48,830)、3(46,524) | — | — |

（XGBoost 的叶是常数叶，`predictive_feature` 字段按定义为空/占位，不是"100% 用 bias 做预测"。）

**中性事实句**
- 中：两个数据集的叶都很大且叶内样本都很小（中位 11 与 8 行）；差别不在叶大小，而在**预测特征是否位于该叶的分裂路径上**——cpu_activity 为 68.7%，kin8nm 为 19.3%。
- 中：cpu_activity 中被选最多的预测特征是第 20 号特征（19,921 次），kin8nm 是第 5/6/3 号。
- En: Leaf support is similarly small in both datasets (median 11 vs 8 training rows). The datasets differ in whether the selected predictive feature lies outside the leaf's split path: 68.7% for cpu_activity versus 19.3% for kin8nm.

## 3. 能回答 Rev1-3 到什么程度

**可以给出机制候选，但不能给因果**：
- ✅ 能说：退化不是噪声（五折全部同向，+1.13 到 +2.92）。
- ✅ 能说：正负例的区分变量是"路径外预测特征占比"（68.7% vs 19.3%），不是叶样本量（11 vs 8）。
- ⚠️ 只能作为假设：路径外特征是所在叶从未分裂过的方向，叶内该特征的变化幅度小、线性外推的方差大；这与 N3 中 cpu_activity 的候选验证误差出现 76–4,117 的数值爆炸现象一致（见 T3），但两者都还是相关性证据。
- ❌ 不能说：已证明"路径外特征导致退化"。要做因果需要受控干预（例如限制候选特征只在路径内），属于新实验。

## 4. English appendix draft

**Table S4.** Post-hoc structural diagnostics of the selected configurations, five outer folds per dataset.

| Dataset | Method | Leaves | Constant-leaf share | Single-feature leaves | Of which outside split path | Median leaf support |
|---|---|---|---|---|---|---|
| cpu_activity | CrossGB | 88,303 | 24.4% | 75.6% | **68.7%** | 11 |
| cpu_activity | XGBoost | 91,263 | 100% | — | — | 15 |
| kin8nm | CrossGB | 455,916 | 21.5% | 78.5% | **19.3%** | 8 |
| kin8nm | XGBoost | 262,038 | 100% | — | — | 10 |

*Notes.* Descriptive post-hoc measurements on the selected models; they identify candidate mechanisms but do not establish a causal account of the performance gap. A leaf's predictive feature is "outside the split path" when the feature was not used in any split leading to that leaf.

## 5. 填数要点

- 若诊断指向"one-hot 下叶内几乎无变化 ⇒ 线性项的估计方差吃掉增益"，写成**可检验的假设**并说明如何在修改稿验证，不能写成已证结论。
- 若诊断指向实现细节（shrinkage 步长、叶刷新频率），区分"我们的实现"与"方法固有性质"：前者可直接修，后者要承认。
- ✅ 覆盖率问题已解决（20/20）；剩余只有"因果"这一层不能越过。
