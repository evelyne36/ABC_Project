# 通用梯度提升库

## XGBoost 回顾

损失函数在第 $t$ 步的泰勒展开为:

$$\sum_{i=1}^n \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t)$$

在大多数情况下，我们假设 $f_t(x_i)$ 对左右子树都是常数。令 $w_j = f_t(x_i)$ 为叶子 $j$ 中样本的预测值，可得:

$$
\begin{align}
L &= \sum_{i=1}^n \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t) \\
&= \sum_{j=1}^T \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i\right) w_j^2\right] + \Omega(f_t)
\end{align}
$$

其中 $I_j$ 表示叶子 $j$ 中的样本集合。

对于固定的树结构，我们可以计算叶子 $j$ 的最优权重 $w_j^*$:

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

在无加权样本的回归中，$h_i = 1$ 且 $g_i = y_i - \hat{y}_i^{(t-1)}$，即简单的残差。因此，给定叶子的最优权重就是残差的均值。

## 条件变量 (Conditional Variables) vs. 预测变量 (Predictive Variables)

我们可以将树节点分裂概念化为使用两类变量:

- **条件变量 (Conditional variables)**: 用于决定分裂点
- **预测变量 (Predictive variables)**: 用于在叶子节点评估和计算预测值

在标准梯度提升中，预测变量仅仅是每个叶子 $j$ 的偏置项(常数) $w_j$。

### 动机: 当偏置项不够用时

在某些情况下，条件变量单独没有预测能力，使得仅用偏置的预测变量过于薄弱。

考虑这个经典的 XOR 类分类示例:

```
y-axis
  ^
  | x o
  | o x
  +-------> x-axis
```

我们想要将 `x` 和 `o` 分类。单独的 x 轴或 y 轴都不具有信息量。如果单独使用 x 或 y 作为决策变量，损失不会显著降低。标准梯度提升会产生一个两层树:

```
      x < 1
      /   \
   y<1     y<1
   / \     / \
  o   x   x   o
```

然而，如果我们可以使用变量作为预测函数，我们只需要一个单层树:

```
    x < 1
    /   \
   y    1-y
```

这使得决策边界更加高效和表达力更强！

## 通用梯度提升 (Generalized Gradient Boosting)

在 XGBoost 中，$f_t(x_i)$ 是一个常数偏置，这在树学习算法中提供了显著的加速。我们提出将其泛化为:

$$f_t(x_i) = \beta \cdot x_{ij}$$

其中 $x_{ij}$ 是样本 $x_i$ 的特征 $j$。注意 $f_t(x_i)$ 可以是任何函数，但我们需要在速度与表达能力之间保持权衡——更复杂的函数会降低速度并增加过拟合风险。

### 修改后的目标函数 (Modified Objective)

通过这种泛化，目标函数变为:

$$
\begin{align}
L &= \sum_{i=1}^n \left[g_i \beta x_{ij} + \frac{1}{2}h_i \beta^2 x_{ij}^2\right] + \Omega(f_t) \\
&= \sum_{j=1}^T \left[\left(\sum_{i \in I_j} g_i x_{ij}\right) \beta_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i x_{ij}^2\right) \beta_j^2\right] + \Omega(f_t)
\end{align}
$$

对于固定的树结构，我们可以计算叶子 $j$ 的最优系数 $\beta^*$:

$$
\beta_j^* = -\frac{\sum_{i \in I_j} g_i x_{ij}}{\sum_{i \in I_j} h_i x_{ij}^2 + \lambda}
$$

### 向后兼容性 (Backward Compatibility)

对于每个数据集，我们可以添加一个偏置特征(常数 1)并将其用作每次分裂的预测变量候选。当选择偏置时，算法退化为原始的 XGBoost 公式，确保向后兼容性。

## 主要优势

- **更强的表达能力**: 线性预测函数可以捕捉常数预测无法捕捉的模式
- **高效性**: 保持与标准梯度提升相当的计算效率
- **灵活性**: 自然处理单个特征边际效应弱但交互效应强的情况
- **向后兼容**: 将标准梯度提升作为特例包含在内


## MetaGB API

### 版本选择

- **version=0**: 标准梯度提升(等同于 XGBoost)。使用相同参数时产生相似结果。
- **version=1**: 已弃用，可忽略。
- **version=2**: 完整的通用梯度提升算法，支持深度分离和先验特征。
- **version=3**: 简化的通用梯度提升。**推荐用于大多数使用场景。**

### 核心算法

该算法将特征分为两个不同的角色:

- **条件变量 (Conditional variables)** ($c_1, c_2, \ldots, c_n$): 用于确定树的分裂
- **预测变量 (Predictive variables)** ($p_1, p_2, \ldots, p_m$): 用于计算叶子预测值

对于每次树分裂，算法:

1. 枚举条件变量 $c_i$ 和预测变量 $p_j$ 的所有组合
2. 对于条件变量 $c_i$ 的每个分裂点 $s_k$，计算分裂增益:
   $$\text{Gain} = L_{\text{left}} + L_{\text{right}} - L_{\text{overall}}$$
   其中叶子预测是 $\beta \cdot p_j$ 而不是常数偏置(见上面的[修改后的目标函数](#修改后的目标函数-modified-objective))
3. 选择最小化增益的最优组合 $(c_i^*, p_j^*, s_k^*)$
4. 使用 $c_i^*$ 在阈值 $s_k^*$ 处分裂树，叶子使用 $p_j^*$ 进行预测

### 向后兼容性

当预测变量集仅包含**偏置项**(值为 1 的常数特征)时，算法退化为标准梯度提升，其中每个叶子预测常数值 $w_j$。这通过向输入数据附加偏置特征自动处理。


## Version 3: 简化的通用梯度提升

Version 3 (`train_simple`) 是一个简化的实现，移除了深度分离和先验特征的复杂性，同时保留了核心的通用梯度提升功能。

### 何时使用 Version 3

- **大多数使用场景**: Version 3 更简单，足以应对大多数应用
- **快速实验**: 需要调整的参数更少
- **无先验特征知识**: 当你没有领域特定的先验特征时

### 与 Version 2 的主要区别

| 特性 | Version 2 | Version 3 |
|------|-----------|-----------|
| 先验特征 | 通过 `cond_prior_features` 支持 | 不支持 |
| 深度分离 | `depth_sep`, `cond_depth_sep`, `pred_depth_sep` | 不支持(所有深度使用所有特征) |
| 特征选择 | 复杂的层次化选择 | 简单的随机采样 |
| 参数 | 许多与深度相关的参数 | 最少的参数 |

### Version 3 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `feature_fraction` | 必需 | 用于条件判断的特征采样比例(同 `cond_feature_fraction`) |
| `pred_feature_fraction` | 同 `feature_fraction` | 用于预测的特征采样比例。设为 `0` 则仅使用偏置。 |
| `leaf_use_all_pred_feature` | `1` (True) | 是否在叶子节点使用所有采样的预测特征 |
| `leaf_only_use_bias` | `0` (False) | 当为 `1` 时，叶子预测仅使用偏置特征(在叶子处退化为标准 GB) |
| `cond_pred_share` | `0` | `0`: 独立采样，`1`: 预测特征从条件特征派生 |

### 使用示例

```python
from metagb_utils import MetaGB

# 基本用法 - 条件和预测都使用 feature_fraction
model = MetaGB(
    learning_rate=0.1,
    num_trees=150,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    min_data_in_leaf=20,
    max_depth=5,
    max_bin=64,
    version=3,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 使用单独的 pred_feature_fraction
model = MetaGB(
    learning_rate=0.1,
    num_trees=150,
    feature_fraction=0.8,        # 条件特征
    pred_feature_fraction=0.4,   # 预测特征(可以不同)
    bagging_fraction=0.8,
    min_data_in_leaf=20,
    max_depth=5,
    max_bin=64,
    version=3,
)

# 叶子节点仅使用偏置预测(类似标准 GB，但条件/预测采样分离)
model = MetaGB(
    learning_rate=0.1,
    num_trees=150,
    feature_fraction=0.8,
    pred_feature_fraction=0.0,   # 不采样预测特征
    leaf_only_use_bias=1,        # 叶子节点仅使用偏置
    version=3,
)
```

---

## Version 2: 完整的通用梯度提升

Version 2 (`train_memory`) 提供完整的特性集，包括深度分离和先验特征支持。

### 参数

#### 1. 先验条件特征 (Prior Conditioning Features)

当你有先验知识认为某些特征应作为条件变量使用时使用这些参数。例如，如果你知道"星期几"不直接预测股票收益，但是一个强大的条件特征(即其他特征与收益的关系因星期几而异)，你可以将其设置为先验条件特征。

| 参数 | 描述 |
|------|------|
| `prior_cond_feature_idx` | 特征索引数组，例如 `[0, 1, 2]` |
| `prior_cond_feature_fraction` | 采样概率数组(长度相同)，例如 `[0.5, 0.6, 0.9]`。指定每棵树采样每个特征作为条件特征的概率。 |
| `num_prior_cond_features` | 特征索引数组的长度 |

#### 2. 深度分离参数 (Depth Separation Parameters)

这些参数控制在不同树深度使用哪些特征，实现**层次化特征选择**策略。

##### `depth_sep` (输入参数)

用户指定的深度阈值，将树分为**浅层** (`depth < depth_sep`) 和**深层** (`depth >= depth_sep`) 区域。

##### `cond_depth_sep` (内部计算)

`cond_feature_idx[]` 数组中的索引，分隔**先验条件特征**和**其他条件特征**。

```
cond_feature_idx = [prior_0, prior_1, ..., other_0, other_1, ...]
                   |<--- 0 到 cond_depth_sep --->|<--- cond_depth_sep 到末尾 --->|
                         (先验特征)                    (其他特征)
```

- 如果 `cond_shallow_use_prior_only=1`: `cond_depth_sep = num_prior_cond_features_sampled`
- 否则: `cond_depth_sep = num_frac_features_cond` (所有特征)

##### `pred_depth_sep` (内部计算)

`pred_feature_idx[]` 数组中的索引，控制在不同深度使用哪些预测特征。

```
pred_feature_idx = [feat_0, feat_1, ..., feat_n, bias]
                   |<--- 0 到 pred_depth_sep --->|<--- pred_depth_sep 到末尾 --->|
                        (仅深层区域)                    (浅层区域)
```

- 如果 `pred_deep_use_prior_only=1`: `pred_depth_sep = num_frac_features_pred - 1` (浅层排除偏置)
- 否则: `pred_depth_sep = 0` (所有位置使用所有特征)

##### 深度分离的工作原理

| 树区域 | 条件特征 | 预测特征 |
|--------|----------|----------|
| 浅层 (`depth < depth_sep`) | `cond_feature_idx[0..cond_depth_sep]` (仅先验) | `pred_feature_idx[0..pred_depth_sep]` |
| 深层 (`depth >= depth_sep`) | `cond_feature_idx[0..end]` 或 `[cond_depth_sep..end]` | `pred_feature_idx[pred_depth_sep..end]` |

##### 可视化示例

```
depth_sep=2, cond_shallow_use_prior_only=1 的树:

深度 0:  [仅使用先验条件特征分裂]
           /                              \
深度 1:  [先验条件特征]                    [先验条件特征]
          /        \                      /        \
深度 2: [所有条件特征]  [所有]            [所有]     [所有条件特征]
```

#### 3. 特征共享参数 (Feature Sharing Parameter)

##### `cond_pred_share`

控制预测特征是独立采样还是从条件特征派生。

| 值 | 行为 |
|----|------|
| `0` | **独立采样**: 条件和预测特征根据 `cond_feature_fraction` 和 `pred_feature_fraction` 分别采样 |
| `1` | **共享特征**: 预测特征 = 条件特征(排除先验) + 偏置 |

**当 `cond_pred_share=0` (独立):**
```
cond_feature_idx = [0, 1, 3, 5]      (用 cond_feature_fraction 采样)
pred_feature_idx = [2, 4, 6, bias]   (用 pred_feature_fraction 采样)
```

**当 `cond_pred_share=1` (共享):**
```
cond_feature_idx = [prior_0, prior_1, 3, 5]
pred_feature_idx = [3, 5, bias]      (从条件特征派生，排除先验)
```

#### 参数总结表

| 参数 | 类型 | 描述 |
|------|------|------|
| `depth_sep` | 输入 | 切换特征集的树深度阈值 |
| `cond_depth_sep` | 计算 | 分隔先验与其他条件特征的索引 |
| `pred_depth_sep` | 计算 | 分隔浅层与深层预测特征的索引 |
| `cond_pred_share` | 输入 | `0`: 独立采样，`1`: 预测从条件派生 |
| `cond_shallow_use_prior_only` | 输入 | `0`: 使用所有，`1`: 浅层仅先验，`2`: 排他(浅层先验，深层其他) |
| `pred_deep_use_prior_only` | 输入 | `0`: 使用所有预测特征，`1`: 按深度限制特征 |

#### 4. 分数聚合参数 (Score Aggregation Parameter)

##### `score_agg`

控制**在评估潜在分裂时如何聚合预测特征和分箱的分数**。

对于每个潜在分裂，算法计算一个 3D 分数数组:

```
score_all[ci][pi][bin] = 分裂增益，对于:
  - 条件特征 ci
  - 预测特征 pi
  - 分箱阈值 bin
```

`score_agg` 参数决定如何找到最佳的 `(ci, pi, bin)` 组合:

##### 确定性方法 (优化分箱选择)

| 值 | 方法 | 描述 |
|----|------|------|
| `0` | **所有中的最大值** | 找到分数最大的单个最佳 `(ci, pi, bin)`。最贪心。 |
| `1` | **预测特征平均，分箱最大** | 对每个 `(ci, bin)`，对所有预测特征的分数取平均，然后找最佳分箱。减少对单个预测特征的过拟合。 |
| `2` | **预测特征中位数，分箱最大** | 对每个 `(ci, bin)`，对预测特征分数取中位数，然后找最佳分箱。对异常值更鲁棒。 |
| `3` | **随机预测特征，分箱最大** | 对每个 `(ci, bin)`，随机选择一个预测特征的分数，然后找最佳分箱。增加多样性。 |

##### 随机化方法 (随机分箱选择)

| 值 | 方法 | 描述 |
|----|------|------|
| `10` | **随机分箱** | 对每个 `(ci, pi)`，随机选择一个分箱。极端随机化。 |
| `11` | **预测特征平均，随机分箱** | 对预测特征取平均，随机选择分箱。 |
| `12` | **预测特征中位数，随机分箱** | 对预测特征取中位数，随机选择分箱。 |
| `13` | **随机预测特征，随机分箱** | 随机选择预测特征和分箱。最大随机化。 |

##### 可视化说明

```
单个条件特征 ci 的 score_all 数组:

           bin_0  bin_1  bin_2  ...  bin_k
pred_0  [  s00    s01    s02   ...   s0k  ]
pred_1  [  s10    s11    s12   ...   s1k  ]
pred_2  [  s20    s21    s22   ...   s2k  ]
  ...
pred_m  [  sm0    sm1    sm2   ...   smk  ]

score_agg=0:  max(所有值)              → 最佳 (pi*, bin*)
score_agg=1:  avg(列) → max(行)        → 最佳 bin*
score_agg=2:  median(列) → max(行)     → 最佳 bin*
score_agg=3:  随机行 → max(行)         → 最佳 bin*

score_agg=10-13: 相同逻辑但随机分箱而非最大值
```

##### 权衡

| 方法 | 优点 | 缺点 |
|------|------|------|
| **Max (0)** | 找到全局最优分裂 | 可能对特定预测特征过拟合 |
| **Average (1)** | 更鲁棒，考虑所有预测特征 | 可能错过强个体信号 |
| **Median (2)** | 对异常预测特征鲁棒 | 类似平均 |
| **Random (3, 10-13)** | 增加多样性，减少过拟合 | 可能错过最优分裂 |

##### 推荐

- **`score_agg=0`** (默认): 适用于大多数情况，找到最优分裂
- **`score_agg=1` 或 `2`**: 当你希望分裂在多个预测特征上泛化良好时使用
- **`score_agg=10-13`**: 用于额外的随机化，类似于"极端随机树"