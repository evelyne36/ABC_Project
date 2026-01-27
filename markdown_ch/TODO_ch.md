# ABC_Project

一个通用梯度提升库，通过解耦条件特征(分裂)和预测特征(叶子)来扩展 XGBoost。

## 验证计划: MetaGB vs XGBoost

本文档概述了一个系统化计划，用于验证通用梯度提升 (MetaGB) 在适当任务上能够优于标准 XGBoost。

---

### 阶段 1: 人工数据集

**目标**: 设计合成数据集，使 MetaGB 的架构优势理论上应该有所帮助。

#### 1.1 XOR 类交互数据集
- **理由**: MetaGB 可以在叶子节点学习 `β·x` 而不是常数，可能更高效地处理特征交互
- **数据集**: 2D/3D XOR 模式，分类依赖于特征乘积
- **假设**: 使用适当 `pred_feature_fraction` 的 MetaGB 应该需要更浅的树

#### 1.2 叶子线性数据集 (Linear-in-Leaves Datasets)
- **理由**: 当真实模型在决策边界处具有线性函数时
- **数据集**: 分段线性回归，每个区域具有不同的斜率
- **示例**: 当 `x1 > 0` 时 `y = x1 * x2`，否则 `y = -x1 * x2`
- **假设**: MetaGB 应该用更少的树达到更低的误差

#### 1.3 条件-预测分离数据集 (Conditional-Predictive Separation Datasets)
- **理由**: 当某些特征更适合分裂而非预测时
- **数据集**: 生成数据，其中分类特征定义区域，连续特征定义区域内的斜率
- **示例**: 星期几决定机制，数值特征决定预测幅度
- **假设**: 具有显式特征分离的 MetaGB 应该表现出色

#### 1.4 高维稀疏交互数据集 (High-Dimensional Sparse Interaction Datasets)
- **理由**: 许多无关特征，少数重要交互
- **数据集**: 100+ 特征，只有 5-10 个相关，具有乘法交互
- **假设**: MetaGB 的独立特征采样可能更高效地发现交互

#### 1.5 噪声鲁棒性测试 (Noise Robustness Tests)
- **数据集**: 清晰信号 + 不同水平的噪声特征
- **假设**: MetaGB 的线性叶子可能与常数叶子有不同的正则化效果

---

### 阶段 2: 来自 TabArena 的真实数据集

**目标**: 在来自 TabArena 基准的多样化、真实的表格数据集上进行评估。

**来源**: [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2505.22252)

#### 2.1 数据集选择标准
选择 MetaGB 可能具有优势的属性的数据集:
- **混合特征类型**: 分类特征(用于条件) + 连续特征(用于预测)
- **已知特征交互**: 交互重要的领域(例如金融、医疗)
- **大型特征集**: 在树的不同组件进行特征选择有帮助的地方
- **回归任务**: 线性叶子预测有意义的地方

#### 2.2 候选数据集
| 数据集 | 领域 | 大小 | 特征数 | 任务 | 相关性 |
|---------|--------|------|----------|------|--------------|
| California Housing | 房地产 | 20K | 8 | 回归 | 基准，现有脚本 |
| Adult/Census Income | 人口统计 | 48K | 14 | 分类 | 混合类型 |
| Covertype | 生态学 | 581K | 54 | 多分类 | 高维，交互 |
| Higgs | 物理学 | 11M | 28 | 二分类 | 特征交互 |
| NYC Taxi | 交通运输 | 大型 | 混合 | 回归 | 时间条件 |
| Medical datasets | 医疗 | 不定 | 混合 | 各种 | 领域知识分离 |

#### 2.3 评估协议
- 使用官方的训练/测试分割(如可用)
- 报告多个随机种子的均值 ± 标准差
- 跟踪准确率/RMSE 和计算成本

#### 2.4 数据预处理: 归一化 (Normalization)

**重要**: 对于真实数据集，特征尺度对 MetaGB 的广义叶子预测影响显著。

**为什么归一化很重要**:
- MetaGB 计算叶子预测为 `β·x`，其中 `x` 是特征值
- 如果特征的尺度差异很大(例如，收入以千为单位 vs 年龄以十年为单位)，学习到的 `β` 系数将被大尺度特征主导
- 这会损害优化稳定性和预测性能

**推荐的预处理**:
```python
# 选项 1: StandardScaler (零均值，单位方差)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 选项 2: MinMaxScaler (缩放到 [0, 1])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# 选项 3: RobustScaler (对异常值鲁棒)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

**最佳实践**:
- 训练前始终检查特征分布 (`df.describe()`，直方图)
- 仅在训练数据上拟合 scaler，然后转换验证/测试集
- 对于严重偏斜的特征，在缩放前考虑对数转换
- 记录使用了哪种归一化方法以确保可重现性
- 比较有无归一化的结果以量化影响

#### 2.5 特征类型分析: 条件 vs 预测

**重要**: 对于真实数据集，理解哪些特征应该用于条件(分裂) vs 预测(叶子值)对 MetaGB 的有效性至关重要。

**特征类型指南**:

| 特征类型 | 更适合 | 理由 |
|--------------|------------|-----------|
| **分类 / 离散** | 条件 | 自然用于分裂；创建不同的机制 |
| **连续 / 数值** | 预测 | 线性关系由 `β·x` 捕获 |
| **低基数** | 条件 | 唯一值少 → 干净的分裂 |
| **高基数** | 预测 | 值多 → 更适合作为线性预测器 |
| **领域"机制"指标** | 条件 | 例如，地区、类别、时间段 |
| **领域"幅度"指标** | 预测 | 例如，金额、计数、测量值 |

**按领域的示例**:

| 领域 | 条件特征 | 预测特征 |
|--------|---------------------|---------------------|
| **电子商务** | 产品类别、用户细分、设备类型 | 价格、数量、页面停留时间 |
| **金融** | 账户类型、地区、风险等级 | 交易金额、余额、收入 |
| **医疗** | 诊断代码、科室、年龄组 | 实验室值、生命体征、剂量 |
| **时间序列** | 星期几、月份、假日标志 | 滞后值、滚动平均 |

**分析工作流**:
```python
import pandas as pd

# 1. 检查特征类型和基数
for col in df.columns:
    n_unique = df[col].nunique()
    dtype = df[col].dtype
    print(f"{col}: {dtype}, {n_unique} 个唯一值")

# 2. 分类特征
conditional_candidates = []  # 低基数，分类
predictive_candidates = []   # 高基数，连续

for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        conditional_candidates.append(col)
    else:
        predictive_candidates.append(col)

# 3. 使用领域知识进行细化
# - 将"机制"特征移至条件
# - 将"幅度"特征移至预测
```

**实验策略**:
- 从基于基数/类型的自动分类开始
- 使用 `cond_pred_share=0` 允许两个集合的独立采样
- 使用 Version 2 的 `cond_prior_features` 实验显式指定条件特征
- 与 `cond_pred_share=1` (预测从条件派生) 作为消融对比

---

### 阶段 3: 系统化实验

**目标**: 通过受控和广泛的超参数调优进行公平、可重现的比较。

#### 3.0 前提: 超参数审计 (Hyperparameter Audit)

**关键**: 在运行实验之前，彻底理解 XGBoost 和 MetaGB 中的所有超参数。不要想当然地接受默认值或假设。

**需要审计的 XGBoost 超参数**:
| 参数 | 默认值 | 影响 | 注意事项 |
|-----------|---------|--------|-------|
| `max_depth` | 6 | 树的复杂度 | 更深的树 = 更大容量，更高过拟合风险 |
| `learning_rate` (eta) | 0.3 | 步长 | 越低 = 需要更多树，通常泛化更好 |
| `n_estimators` | 100 | 树的数量 | 更多树配合更低的 LR 通常有帮助 |
| `subsample` | 1.0 | 行采样 | <1.0 增加正则化 |
| `colsample_bytree` | 1.0 | 每棵树的列采样 | 特征随机化 |
| `colsample_bylevel` | 1.0 | 每层的列采样 | 经常被忽视 |
| `colsample_bynode` | 1.0 | 每个分裂的列采样 | 最细粒度 |
| `reg_lambda` (L2) | 1.0 | 权重的 L2 正则化 | **XGBoost 默认是 1.0，不是 0!** |
| `reg_alpha` (L1) | 0.0 | L1 正则化 | 诱导稀疏性 |
| `min_child_weight` | 1.0 | 子节点中 hessian 的最小和 | 控制分裂创建 |
| `gamma` | 0.0 | 分裂的最小损失减少 | 剪枝参数 |
| `max_delta_step` | 0 | 最大 delta 步长 | 对不平衡数据有用 |
| `tree_method` | 'auto' | 算法 | 'hist' vs 'exact' 可能不同 |
| `grow_policy` | 'depthwise' | 如何生长树 | 'lossguide' 是按叶子生长 |

**需要审计的 MetaGB 超参数**:
| 参数 | 默认值 | 影响 | 注意事项 |
|-----------|---------|--------|-------|
| `max_depth` | ? | 树的复杂度 | 验证与 XGBoost 含义相同 |
| `learning_rate` | ? | 步长 | 检查应用方式是否相同 |
| `n_estimators` | ? | 树的数量 | - |
| `subsample` | ? | 行采样 | 验证实现与 XGBoost 匹配 |
| `feature_fraction` | ? | 列采样 | **这是每棵树、每层还是每个节点?** |
| `reg_lambda` | ? | L2 正则化 | **检查默认值! XGBoost 使用 1.0** |
| `min_child_weight` | ? | 最小 hessian 和 | 验证公式匹配 |
| `pred_feature_fraction` | ? | 预测特征采样 | MetaGB 特有 |
| `cond_pred_share` | ? | 条件/预测特征关系 | MetaGB 特有 |
| `leaf_only_use_bias` | ? | 仅偏置叶子 | MetaGB 特有 |
| `n_bins` | ? | 直方图分箱 | 检查默认值和影响 |

**验证清单**:
- [ ] 阅读每个参数的 XGBoost 文档
- [ ] 阅读 MetaGB 源代码以验证参数含义
- [ ] **检查默认值** - MetaGB 默认值是否与 XGBoost 对齐?
- [ ] **检查参数交互** - `feature_fraction` 是否与 `pred_feature_fraction` 交互?
- [ ] 运行消融: 一次变化一个参数，观察影响
- [ ] 记录 XGBoost 和 MetaGB 参数行为之间的任何差异

**要避免的常见陷阱**:
1. **Lambda 默认值不匹配**: XGBoost 默认 `reg_lambda=1.0`，许多实现默认为 0
2. **列采样范围**: XGBoost 有 3 个级别 (bytree, bylevel, bynode)；MetaGB 使用哪个?
3. **直方图分箱**: 不同的分箱数量会影响结果
4. **树生长策略**: 深度优先 vs 叶子优先可能产生非常不同的树
5. **梯度/Hessian 计算**: 验证使用相同的公式

**推荐的审计流程**:
```python
# 1. 打印所有默认值
import xgboost as xgb
print(xgb.XGBRegressor().get_params())

# 2. 与 MetaGB 默认值比较
# 阅读 metagb_utils.py 和 metagb.c 的默认值

# 3. 健全性检查: 相同参数应给出相同结果
# 用完全相同的参数训练 XGBoost 和 MetaGB v0
# 比较预测 - 如果实现匹配应该非常接近
```

#### 3.1 方法 A: 受控超参数比较 (Controlled Hyperparameter Comparison)

固定共享超参数，仅改变方法特定的参数:

**固定参数** (所有方法相同):
```python
shared_params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [0.0, 1.0],
    'min_child_weight': [1, 5, 10]
}
```

**方法特定参数**:
```python
# XGBoost: 无额外参数

# MetaGB Version 0 (基准): 无额外参数

# MetaGB Version 3 (广义化):
metagb_v3_params = {
    'pred_feature_fraction': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'cond_pred_share': [0, 1],
    'leaf_only_use_bias': [0, 1]  # 1 退化为标准 GB
}
```

#### 3.2 方法 B: 通过交叉验证进行广泛调优 (Extensive Tuning via Cross-Validation)

**协议**:
1. 在训练集上进行 5 折或 10 折交叉验证
2. 网格搜索或贝叶斯优化 (Optuna)
3. 为每个方法独立选择最佳超参数
4. 在留出的测试集上评估

**搜索空间**:
- XGBoost: ~1000 个配置
- MetaGB v0: ~1000 个配置(与 XGBoost 相同)
- MetaGB v3: ~3000 个配置(额外的 pred_feature_fraction 变体)

**比较指标**:
| 指标 | 目的 |
|--------|---------|
| 最佳 CV 分数 | 样本内性能 |
| 测试分数 | 泛化能力 |
| 训练时间 | 计算成本 |
| 收敛时的树数量 | 模型复杂度 |

#### 3.3 统计显著性
- 跨数据集的配对 t 检验或 Wilcoxon 符号秩检验
- 报告 胜/平/负 计数
- 使用临界差异图进行多数据集比较

---

### 阶段 4: 基准方法 (Baselines)

**目标**: 通过使用 XGBoost 和 MetaGB v0 作为基准来确保公平比较。

#### 4.1 基准方法

| 方法 | 实现 | 注意事项 |
|--------|----------------|-------|
| **XGBoost** | `xgboost` 包 | 行业标准 |
| **MetaGB v0** | `metagb.train_original()` | 我们的 XGBoost 等效实现 |
| **MetaGB v3** | `metagb.train_simple()` | 广义化版本 |

#### 4.2 为什么需要两个基准?

1. **XGBoost 基准**: 确保我们与经过优化、广泛使用的实现进行比较
2. **MetaGB v0 基准**: 控制实现差异(C 代码、树结构等)

如果 MetaGB v3 击败 MetaGB v0 但未击败 XGBoost，表明广义化有帮助但实现需要优化。
如果 MetaGB v3 击败两者，广义化提供了真正的算法改进。

#### 4.3 基准等效性验证
- 首先在标准数据集上验证 MetaGB v0 ≈ XGBoost
- 记录任何性能差距及其原因
- 使用 MetaGB v0 作为消融研究的"公平"基准

---

### 实验执行清单

- [ ] **阶段 1**: 创建合成数据集生成器
  - [ ] XOR 交互数据集
  - [ ] 分段线性数据集
  - [ ] 条件-预测分离数据集
  - [ ] 高维稀疏数据集
  - [ ] 噪声鲁棒性数据集

- [ ] **阶段 2**: 设置 TabArena 数据集
  - [ ] 下载并预处理选定的数据集
  - [ ] 实现数据加载工具
  - [ ] 验证训练/测试分割

- [ ] **阶段 3**: 实现实验框架
  - [ ] 受控超参数比较脚本
  - [ ] 交叉验证网格搜索脚本
  - [ ] Optuna 贝叶斯优化集成
  - [ ] 结果记录和可视化

- [ ] **阶段 4**: 基准验证
  - [ ] 验证 MetaGB v0 ≈ XGBoost 等效性
  - [ ] 记录任何差异
  - [ ] 设置三方比较管道

---

### 预期交付成果

1. **合成数据集分析**: 识别 MetaGB 在理论上和实证上获胜的场景
2. **TabArena 基准结果**: 跨真实数据集的性能比较表
3. **超参数敏感性分析**: `pred_feature_fraction` 如何影响性能?
4. **统计摘要**: 带显著性检验的 胜/平/负 计数
5. **推荐建议**: 何时使用 MetaGB vs XGBoost 的指南