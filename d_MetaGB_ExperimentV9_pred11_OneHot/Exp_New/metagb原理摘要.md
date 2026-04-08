# MetaGB 训练和推理过程详解-**关于colsample_bytree的调研**

## 一、MetaGB 核心概念
MetaGB（Meta Gradient Boosting）是一种改进的梯度提升树算法，其核心创新在于将特征分为**条件特征（cond features）**和**预测特征（pred features）**，分别用于树的结构分裂和叶子节点的预测。

---

## 二、训练过程 (`train_custom` 函数：metagb.c:1571-1897)

### 1. 数据准备阶段
- **输入**：离散化特征 `x`、原始特征 `x_origin`、目标值 `y`
- **预分配**：初始化预测值 `yhat`、特征梯度 `x_grad`、特征海塞矩阵 `xx_hessian`
- **梯度计算**：`x_grad[i][j] = (yhat[i] - y[i]) * x_origin[i][j]`（metagb.c:1672）
- **海塞矩阵**：`xx_hessian[i][j] = x_origin[i][j] * x_origin[i][j]`（metagb.c:1650）

### 2. 主循环：逐树训练 (metagb.c:1666-1887)

每棵树的训练包含以下步骤：

#### 步骤 2.1：样本采样 - `subsample` 参数 (metagb.c:1682-1709)
```c
for (int di = 0; di < num_samples; ++di) {
    double r = ((double)rand()/RAND_MAX);
    if (r <= bagging_fraction) {  // bagging_fraction = subsample
        // 根据 split_sample 策略分配到 cond_active_idx 和 pred_active_idx
    }
}
```
- **作用**：从训练样本中随机选择 `bagging_fraction*100%` 的样本
- **目的**：防止过拟合，增加模型鲁棒性
- **实现**：通过 `cond_active_idx` 和 `pred_active_idx` 标记活跃样本

#### 步骤 2.2：条件特征采样 - `colsample_bytree` → `cond_feature_fraction` (metagb.c:1728-1793)

**2.2.1 优先条件特征优先选择** (metagb.c:1728-1736)
```c
for (int fi = 0; fi < num_prior_cond_features; ++fi) {
    double r = ((double)rand()/RAND_MAX);
    if (r < prior_cond_feature_fraction[fi]) {
        cond_feature_idx[num_frac_features_cond] = prior_cond_feature_idx[fi];
        num_frac_features_cond++;
    }
}
```

**2.2.2 普通特征采样** (metagb.c:1744-1760)
```c
for (int fi = 0; fi < num_features; ++fi) {
    if (fi != bias_feature_idx) {
        double r = ((double)rand()/RAND_MAX);
        if (r <= cond_feature_fraction) {  // cond_feature_fraction = colsample_bytree
            cond_feature_idx[num_frac_features_cond] = fi;
            num_frac_features_cond++;
        }
    }
}
```
- **作用**：从全部特征中随机选择 `cond_feature_fraction*100%` 用于树的分裂决策
- **目的**：减少每棵树的特征数量，增加树之间的多样性
- **结果**：`cond_feature_idx` 数组存储选中的条件特征索引

#### 步骤 2.3：预测特征采样 - `pred_feature_fraction` (metagb.c:1797-1822)

```c
int num_frac_features_pred = 0;
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
} else {
    // 共享策略：使用非优先条件特征
    for (int fi = num_prior_cond_features_sampled; fi < num_frac_features_cond; ++fi)
        pred_feature_idx[fi-num_prior_cond_features_sampled] = cond_feature_idx[fi];
}
```
- **作用**：从全部特征中随机选择 `pred_feature_fraction*100%` 用于叶子节点的预测
- **目的**：减少叶子节点的计算复杂度，提高预测效率
- **结果**：`pred_feature_idx` 数组存储选中的预测特征索引

#### 步骤 2.4：构建决策树 (metagb.c:1837-1867)
调用 [`build_tree_double_custom`](Exp_New/metagb/metagb.c:977-1530) 函数

**核心分裂逻辑**：
- 使用 `cond_feature_idx` 中的特征进行分裂决策（选择最佳分裂点和特征）
- 计算分裂分数时，考虑 `pred_feature_idx` 中每个特征的梯度和海塞矩阵贡献
- 通过 `score_agg` 策略聚合多个预测特征的分裂分数

**叶子节点预测**：
- 叶子节点存储一个预测特征索引和对应的权重
- 权重计算：`leaf_weight = -grad_sum[pi] / (hessian_sum[pi] + lambda_l2)` (metagb.c:1068)

#### 步骤 2.5：更新预测值 (metagb.c:1870-1886)
```c
predict_custom(x_origin, tree_i, yi, ...);
for (int di = 0; di < num_samples; ++di) {
    yhat[di] += yi[di];
}
```

---

## 三、推理过程 (`predict_custom` 函数：metagb.c:1536-1568)

```c
for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    for (int tree_idx = start_tree_idx; tree_idx < end_tree_idx; ++tree_idx) {
        const double *tree_i = tree + tree_idx * num_nodes_per_tree * 5;
        int current_node = 0;
        
        // 遍历树的内部节点
        while (isnan(tree_i[current_node*5+2])) {
            int feature_idx = (int)tree_i[current_node*5];
            // 使用条件特征进行分裂决策
            current_node = (x[sample_idx*num_features + feature_idx] <= tree_i[current_node*5+1]) 
                         ? (int)tree_i[current_node*5+3] 
                         : (int)tree_i[current_node*5+4];
        }
        
        // 叶子节点：使用预测特征乘以权重
        y_pred[sample_idx] += learning_rate * 
                              x[sample_idx*num_features + tree_i[current_node*5]] * 
                              tree_i[current_node*5+2];
    }
}
```

**推理总结**：
1. 对于每个样本，遍历所有树
2. 在每棵树中，沿着分裂路径到达叶子节点（使用**条件特征**）
3. 在叶子节点，使用**预测特征**的值乘以权重作为预测结果
4. 累加所有树的预测结果，乘以学习率得到最终预测

---

## 四、三个参数的详细作用机制

### 1. `subsample` (bagging_fraction) - 样本采样

**作用位置**：metagb.c:1682-1709
```c
double r = ((double)rand()/RAND_MAX);
if (r <= bagging_fraction) {
    // 根据策略分配到 cond_active_idx 和 pred_active_idx
}
```

**作用机制**：
- 从 `num_samples` 个训练样本中随机选择 `floor(bagging_fraction * num_samples)` 个样本
- 被选中的样本标记为 `active_idx=true`，其他标记为 `false`
- 每棵树重新采样，实现数据的随机性

**影响**：
- 减少每棵树的训练样本数，加快训练速度
- 增加树之间的差异性，防止过拟合
- 当参数较小时，可能导致欠拟合

### 2. `colsample_bytree` (cond_feature_fraction) - 条件特征采样

**作用位置**：metagb.c:1744-1760
```c
double r = ((double)rand()/RAND_MAX);
if (r <= cond_feature_fraction) {
    cond_feature_idx[num_frac_features_cond] = fi;
    num_frac_features_cond++;
}
```

**作用机制**：
- 从 `num_features` 个特征中随机选择 `floor(cond_feature_fraction * num_features)` 个特征
- 选中的特征存储在 `cond_feature_idx` 数组中
- 每棵树重新采样，实现特征的随机性

**影响范围**：
- 只影响**条件特征**的选择
- 条件特征用于树的**分裂决策**（选择最佳分裂点和分裂特征）
- 不影响叶子节点的预测特征

**影响**：
- 减少每棵树考虑的特征数量，加快训练速度
- 强制树使用不同的特征子集，增加多样性
- 当参数较小时，可能错过重要的分裂特征

### 3. `pred_feature_fraction` - 预测特征采样

**作用位置**：metagb.c:1797-1822
```c
double r = ((double)rand()/RAND_MAX);
if (r <= pred_feature_fraction) {
    pred_feature_idx[num_frac_features_pred] = fi;
    num_frac_features_pred++;
}
```

**作用机制**：
- 从 `num_features` 个特征中随机选择 `floor(pred_feature_fraction * num_features)` 个特征
- 选中的特征存储在 `pred_feature_idx` 数组中
- 可以与 `cond_feature_fraction` 共享部分特征（通过 `cond_pred_share` 参数）

**影响范围**：
- 只影响**预测特征**的选择
- 预测特征用于**叶子节点的预测权重计算**
- 不影响树的分裂决策

**影响**：
- 减少叶子节点的计算复杂度
- 限制叶子节点只能考虑部分预测特征
- 当参数较小时，可能错过最重要的预测特征

---

## 五、参数协同作用示例

假设：
- 总样本数 = 1000
- 总特征数 = 100
- `subsample = 0.8`
- `colsample_bytree = 0.7`
- `pred_feature_fraction = 0.6`

**单棵树的训练配置**：
1. **样本层**：随机选择 800 个样本参与训练
2. **条件特征层**：随机选择 70 个特征用于分裂决策
3. **预测特征层**：随机选择 60 个特征用于叶子预测

**训练效果**：
- 样本随机性 + 条件特征随机性 + 预测特征随机性
- 三重随机性最大化树之间的差异性
- 有效防止过拟合，提高泛化能力

---

## 六、与传统GBDT的区别

| 维度 | 传统GBDT (如XGBoost) | MetaGB |
|------|---------------------|--------|
| 特征采样 | `colsample_bytree` 统一作用于所有分裂 | 分为 `cond_feature_fraction` 和 `pred_feature_fraction` |
| 叶子节点预测 | 固定预测值（梯度的负梯度/海塞矩阵） | 使用预测特征值 × 权重（类似线性模型） |
| 样本采样 | `subsample` 统一作用于训练 | `subsample` + `split_sample` 策略可分开条件/预测样本 |
| 模型形式 | 每个叶子 = 一个标量值 | 每个叶子 = 一个特征索引 + 权重 |

这种设计使MetaGB能够捕捉更复杂的非线性关系，同时保持计算效率。