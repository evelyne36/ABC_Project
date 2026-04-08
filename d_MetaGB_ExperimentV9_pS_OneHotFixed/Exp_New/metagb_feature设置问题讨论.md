## MetaGB 中 feature_fraction 和 pred_feature_fraction 的区别和作用分析

基于源码和原理摘要，我已分析清楚两个参数的区别：

### 📌 核心概念

MetaGB（Meta Gradient Boosting）将特征分为两种类型：
- **条件特征**：用于树的分裂决策（决定样本走哪个分支）
- **预测特征**：用于叶子节点的预测计算

---

### 📊 参数对比表

| 参数 | 另称 | 作用范围 | 对应C变量 | 代码位置 |
|------|------|----------|----------|----------|
| **`feature_fraction`** | `colsample_bytree`, `cond_feature_fraction` | 条件特征采样 | `cond_feature_fraction` | [`metagb.c:1751`](Exp_New/metagb/metagb.c:1751) |
| **`pred_feature_fraction`** | 无 | 预测特征采样 | `pred_feature_fraction` | [`metagb.c:1802`](Exp_New/metagb/metagb.c:1802) |

---

### 🔍 详细作用机制

#### 1. **`feature_fraction`（条件特征采样）**

**作用位置**：[`metagb.c:1744-1760`](Exp_New/metagb/metagb.c:1744-1760)

```c
double r = ((double)rand()/RAND_MAX);
if (r <= cond_feature_fraction) {
    cond_feature_idx[num_frac_features_cond] = fi;
    num_frac_features_cond++;
}
```

**作用**：
- 从所有特征中随机选择 `feature_fraction * 100%` 的特征作为**条件特征**
- 这些选中的特征用于**树的分裂决策**（选择最佳分裂点和分裂特征）
- 每棵树训练时独立采样

**影响**：
- 直接影响树的**结构构建**
- 减少每棵树考虑的特征数量，加快训练速度
- 强制树使用不同的特征子集，增加树之间的多样性

---

#### 2. **`pred_feature_fraction`（预测特征采样）**

**作用位置**：[`metagb.c:1797-1822`](Exp_New/metagb/metagb.c:1797-1822)

```c
if (!cond_pred_share) {
    for (int fi = 0; fi < num_features; ++fi) {
        if (fi != bias_feature_idx) {
            double r = ((double)rand()/RAND_MAX);
            if (r <= pred_feature_fraction) {
                pred_feature_idx[num_frac_features_pred] = fi;
                num_frac_features_pred++;
            }
        }
    }
}
```

**作用**：
- 从所有特征中随机选择 `pred_feature_fraction * 100%` 的特征作为**预测特征**
- 这些选中的特征用于**叶子节点的预测计算**
- 可通过 `cond_pred_share` 参数决定是否与条件特征共享

**影响**：
- 影响叶子节点的**预测权重计算**
- 限制叶子节点只能考虑部分特征进行预测
- 减少叶子节点的计算复杂度

---

### ⚙️ Python接口默认行为

在 [`metagb_utils.py:39-43`](Exp_New/metagb/metagb_utils.py:39-43)：

```python
def __init__(self, feature_fraction, pred_feature_fraction=None, ...):
    if pred_feature_fraction is None:
        self.pred_feature_fraction = feature_fraction  # 默认相同
    else:
        self.pred_feature_fraction = pred_feature_fraction
```

**默认情况下**，如果未显式指定 `pred_feature_fraction`，则它等于 `feature_fraction`。

---

### 🎯 训练与推理流程对比

#### 训练阶段：
1. 选择条件特征 → 构建树的分裂结构
2. 选择预测特征 → 计算叶子节点权重
3. 叶子节点预测：`y_pred = pred_feature_value * weight`

#### 推理阶段（[`metagb.c:1536-1568`](Exp_New/metagb/metagb.c:1536-1568)）：
```c
// 1. 使用条件特征进行分裂决策
while (isnan(tree_i[current_node*5+2])) {
    int feature_idx = (int)tree_i[current_node*5];
    // 根据条件特征的值决定走向左子树或右子树
}

// 2. 在叶子节点使用预测特征
y_pred[sample_idx] += learning_rate * 
                      x[sample_idx*num_features + tree_i[current_node*5]] *  // 预测特征的值
                      tree_i[current_node*5+2];  // 权重
```

---

### 💡 实践建议

| 场景 | 推荐设置 | 原因 |
|------|----------|------|
| 特征稀疏、重要特征集中 | `feature_fraction = 1.0`<br>`pred_feature_fraction = 0.5` | 所有特征可用于分裂，但预测简化 |
| 特征冗余严重 | `feature_fraction = 0.7`<br>`pred_feature_fraction = 0.7` | 双重采样增强多样性 |
| 高噪声数据 | `feature_fraction = 0.6`<br>`pred_feature_fraction = 0.4` | 更强的正则化 |

---

### 📝 总结

| 维度 | `feature_fraction` | `pred_feature_fraction` |
|------|-------------------|-------------------------|
| **作用阶段** | 树分裂 | 叶子预测 |
| **影响范围** | 树的结构 | 预测值的计算 |
| **传统GBDT对应** | `colsample_bytree` | 无（MetaGB独有） |
| **核心创新** | 与XGBoost相同 | MetaGB的双特征分离机制 |
| **调节建议** | 通常设为0.6-1.0 | 通常设为0.3-1.0 |


## MetaGB v3 版本是否可分别设置 feature_fraction 和 pred_feature_fraction？

**答案：是的，MetaGB v3 版本完全可以分别设置这两个参数！**

---

### ✅ 证据分析

#### 1. Python 接口支持（[`metagb_utils.py:10-43`](Exp_New/metagb/metagb_utils.py:10-43)）

```python
class MetaGB:
    def __init__(self,
                 learning_rate,
                 num_trees,
                 feature_fraction,      # 条件特征采样率
                 bagging_fraction,
                 min_data_in_leaf,
                 max_depth,
                 max_bin,
                 pred_feature_fraction=None,  # 可单独设置预测特征采样率
                 depth_sep=None,
                 use_random_depth_sep=0,
                 leaf_use_all_pred_feature=None,
                 ...
                 ):
        self.feature_fraction = feature_fraction
        # 默认行为：如果不指定pred_feature_fraction，则等于feature_fraction
        if pred_feature_fraction is None:
            self.pred_feature_fraction = feature_fraction
        else:
            self.pred_feature_fraction = pred_feature_fraction  # 可以单独设置
```

#### 2. v3 版本的 fit_simple 函数明确调用（[`metagb_utils.py:266-292`](Exp_New/metagb/metagb_utils.py:266-292)）

```python
lib.train_simple(x_ptr,
                x_origin_ptr,
                bin_ptr,
                y_ptr,
                tree_ptr,
                n, d, d-1,
                self.num_trees,
                self.max_depth,
                self.max_bin,
                self.min_data_in_leaf,
                self.bagging_fraction,
                self.feature_fraction,      # 第12个参数
                self.pred_feature_fraction,  # 第13个参数 - 独立传递
                self.learning_rate,
                self.leaf_use_all_pred_feature,
                self.leaf_only_use_bias,
                ...
                )
```

#### 3. C 语言函数签名确认（[`metagb.c:3290-3306`](Exp_New/metagb/metagb.c:3290-3306)）

```c
void train_simple(const unsigned char *restrict x,
    const double *restrict x_origin,
    const double *restrict bins,
    const double *restrict y,
    double *restrict tree,
    int num_samples,
    int num_features,
    int bias_feature_idx,
    int num_trees,
    int max_depth,
    int max_bin,
    int min_data_in_leaf,
    double bagging_fraction,
    double cond_feature_fraction,      // 第13个参数：条件特征采样率
    double pred_feature_fraction,      // 第14个参数：预测特征采样率（独立！）
    double learning_rate,
    int leaf_use_all_pred_feature,
    int leaf_only_use_bias,
    int cond_pred_share,
    ...
    )
```

**断言确认**（[`metagb.c:3311-3312`](Exp_New/metagb/metagb.c:3311-3312)）：
```c
assert(cond_feature_fraction > 0);      // 条件特征采样率必须 > 0
assert(pred_feature_fraction >= 0);    // 预测特征采样率可以 >= 0（包括0）
```

#### 4. 实际使用示例

在代码中有多处显式分别设置的示例：

**示例1**（[`compare_metagb_xgboost_housing_cv.py:247`](Exp_New/metagb/compare_metagb_xgboost_housing_cv.py:247)）：
```python
metagb_param_grid = {
    'feature_fraction': [0.8],
    'pred_feature_fraction': [0.0, 0.4, 0.8],  # 独立设置不同值
    'version': [3],
    ...
}
```

**示例2**（[`compare_metagb_xgboost_grid.py:214`](Exp_New/metagb/compare_metagb_xgboost_grid.py:214)）：
```python
metagb_param_grid = {
    'feature_fraction': [0.8],
    'pred_feature_fraction': [0.0],  # 0.0 = bias only（仅用bias，类似标准GBDT）
    'version': [3],
    ...
}
```

---

### 📋 使用建议

#### 场景1：使用相同采样率（默认行为）
```python
model = MetaGB(
    feature_fraction=0.8,
    # pred_feature_fraction 不设置，默认=0.8
    version=3
)
```

#### 场景2：分别设置不同采样率
```python
model = MetaGB(
    feature_fraction=1.0,           # 所有特征用于分裂
    pred_feature_fraction=0.5,     # 仅50%特征用于预测
    version=3
)
```

#### 场景3：仅用bias预测（类似传统GBDT）
```python
model = MetaGB(
    feature_fraction=0.8,
    pred_feature_fraction=0.0,      # =0表示只使用bias特征
    version=3
)
```

---

### 🎯 关键点总结

| 维度 | 说明 |
|------|------|
| **v3是否支持分别设置** | ✅ 完全支持 |
| **默认行为** | 如果不指定`pred_feature_fraction`，则等于`feature_fraction` |
| **独立传递** | Python层显式传递两个独立参数给C函数 |
| **取值范围** | `feature_fraction > 0`，`pred_feature_fraction >= 0` |
| **实际应用** | 代码中有多处分别搜索的示例 |


## 分析：代码中的 `pred_feature_fraction` 与 `colsample_bytree` 是否冲突？

**结论：这不冲突，代码设计是正确的！** 这是一个**兼容性设计**，允许在网格搜索中灵活控制参数组合。

---

### 📊 代码逻辑分析

#### 1. 第69行的默认值逻辑

```python
pred_feature_fraction=param_dict.get('pred_feature_fraction', param_dict['colsample_bytree']),
```

这行代码的含义是：
- 如果 `param_dict` 中有 `pred_feature_fraction` 键 → 使用它的值
- 如果没有 → 回退到使用 `colsample_bytree` 的值

#### 2. 第245-247行的参数网格定义

```python
metagb_param_grid = {
    'colsample_bytree': [0.8, 1.0],        # 2个候选值
    # MetaGB-specific parameters
    'pred_feature_fraction': [0.0, 0.4, 0.8],  # 3个候选值
    ...
}
```

---

### 🔍 实际参数组合推导

使用 `itertools.product` 展开网格搜索（第139行 `product(*param_values)`）：

| 组合编号 | colsample_bytree | pred_feature_fraction | 说明 |
|---------|----------------|---------------------|------|
| 1 | 0.8 | 0.0 | 所有情况都**显式指定**，不会使用默认值 |
| 2 | 0.8 | 0.4 | |
| 3 | 0.8 | 0.8 | |
| 4 | 1.0 | 0.0 | |
| 5 | 1.0 | 0.4 | |
| 6 | 1.0 | 0.8 | |

总组合数 = 2 × 3 = **6 个组合**

---

### 💡 设计意图说明

这个设计实现了**两种场景的兼容性**：

#### 场景A：显式指定 `pred_feature_fraction`（当前代码）

```python
metagb_param_grid = {
    'colsample_bytree': [0.8, 1.0],
    'pred_feature_fraction': [0.0, 0.4, 0.8],  # 显式指定
}
```

**结果**：6种组合，每个都使用独立的 `pred_feature_fraction` 值

#### 场景B：仅指定 `colsample_bytree`，希望复用默认行为

```python
metagb_param_grid = {
    'colsample_bytree': [0.8, 1.0],
    # 不指定 'pred_feature_fraction'
}
```

**结果**：2种组合，每个都使用 `colsample_bytree` 的值作为 `pred_feature_fraction`

```python
# 第69行的 get 充当智能默认值
pred_feature_fraction=param_dict.get('pred_feature_fraction', param_dict['colsample_bytree'])
```

---

### ✔️ 验证：不冲突的原因

1. **第69行的默认值只在 `pred_feature_fraction` 未在 param_dict 中时才会被使用**
   - 当前代码的 param_grid 中**明确指定了** `'pred_feature_fraction': [0.0, 0.4, 0.8]`
   - 因此所有6种组合都会**使用显式指定的值**，不会回退到 `colsample_bytree`

2. **参数独立性得到保持**
   ```python
   # 组合1
   feature_fraction=0.8          # 来自 colsample_bytree
   pred_feature_fraction=0.0     # 来自 pred_feature_fraction
   
   # 组合2  
   feature_fraction=0.8          # 来自 colsample_bytree
   pred_feature_fraction=0.4     # 来自 pred_feature_fraction
   ```

3. **设计灵活性**
   - 当需要**分别探索**两者时：提供两个独立的参数列表
   - 当需要**保持一致**时：只提供 `colsample_bytree`，让 `pred_feature_fraction` 使用默认值

---

### 📝 总结

| 问题 | 答案 |
|------|------|
| **是否冲突？** | ❌ 不冲突 |
| **为什么？** | `.get()` 的默认值只在参数**不存在**时生效，当前代码中参数**始终存在** |
| **实际效果** | 产生 2×3=6 种独立的参数组合 |
| **设计优势** | 灵活性：既可以独立控制，也可以统一设置 |

---

### 🎯 如果想避免混淆

可以显式说明设计意图：

```python
# MetaGB parameter grid
metagb_param_grid = {
    'colsample_bytree': [0.8, 1.0],
    'pred_feature_fraction': [0.0, 0.4, 0.8],  # 独立探索预测特征采样率
    ...
}

# 如果想复用 colsample_bytree 的值，只需移除上面的 'pred_feature_fraction' 行
# 代码第69行会自动使用 colsample_bytree 作为默认值
```