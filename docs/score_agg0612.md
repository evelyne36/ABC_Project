# MetaGB `score_agg` 理念、语义修正与实验结论

## 1. 背景与核心问题

MetaGB 将特征分为两种角色：

- **conditional feature**：用于划分样本区域；
- **predictive feature**：用于最终叶节点的局部预测函数
  `f(x)=beta*x_j`。

因此，讨论 `score_agg` 时必须区分两个独立问题：

1. 如何利用 predictive-feature gain 选择 conditional split；
2. conditional split 确定后，最终叶节点使用哪个 predictive feature。

本次探究最初发现，嵌入版 `current_metagb` 在 `score_agg=1/2/3`
的切分处理中出现了 `split_pred_features.front()`。这看起来意味着模型会固定使用候选列表中的第一个 predictive feature，与 MetaGB 理念不一致。

进一步追踪原版和嵌入版完整数据流后发现：

- 原版 MetaGB 的 `score_agg` 只负责选择 conditional split；
- 原版每个最终叶节点都会基于该叶样本重新选择最佳 predictive feature；
- 修改前嵌入版的正常构树路径通常也会通过后续 `BestLeaf` 隐式覆盖
  `front()`；
- 但修改前的实现依赖隐式覆盖，并在切分阶段保存了没有理论意义的临时
  predictive feature 和 leaf weight。

因此，本次修改的目标是将原版最终叶重选语义变成嵌入版中的显式强制步骤，
彻底解除 split aggregation 与最终 leaf model 的错误耦合。

---

## 2. 原版 MetaGB 的真实语义

### 2.1 `score_agg` 只控制 conditional split 选择

原版首先计算三维得分：

```text
score_all[conditional feature][predictive feature][split bin]
```

不同 `score_agg` 的行为如下：

| `score_agg` | conditional split 评分方式 | 定位 |
|---:|---|---|
| `0` | 在所有 `(cond, pred, bin)` 中选择最大 gain | 联合贪心切分 |
| `1` | 对同一 `(cond, bin)` 下所有 pred-feature gain 取均值 | 均值稳健切分 |
| `2` | 对同一 `(cond, bin)` 下所有 pred-feature gain 取中位数 | 中位数稳健切分 |
| `3` | 对每个 `(cond, bin)` 随机取一个 pred-feature gain | 随机化切分 |

代码证据：

- `metagb/metagb.c:3008-3019`：`score_agg=0`
- `metagb/metagb.c:3021-3033`：`score_agg=1`
- `metagb/metagb.c:3034-3057`：`score_agg=2`
- `metagb/metagb.c:3058-3069`：`score_agg=3`

对于 `score_agg=1/2/3`，原版只确定最佳 conditional feature 和 split bin，
并不会保存某个与聚合得分对应的 predictive feature。

### 2.2 最终叶节点独立重新选择 predictive feature

原版到达最终叶节点时，会重新遍历全部候选 predictive features，计算每个
feature 在当前叶内的 leaf score：

```text
score_j = grad_x_j^2 / (hess_xx_j + lambda)
```

然后选择 score 最大的 predictive feature，并保存对应权重。

代码证据：

- `metagb/metagb.c:2881-2904`：遍历并选择最佳 leaf predictive feature；
- `metagb/metagb.c:2910-2912`：保存 feature 和 leaf weight；
- `metagb/metagb.c:3192-3198`：提前停止形成叶节点时同样保存重新选择的 feature。

因此原版实际流程是：

```text
score_agg 选择 conditional split
    -> 划分出最终叶区域
    -> 每个最终叶根据叶内样本重新选择最佳 predictive feature
    -> 使用 leaf_weight * predictive_feature 预测
```

`score_agg=1/2` 不会选择所谓“均值 predictive feature”或“中位数 predictive
feature”。它们只是利用所有 predictive features 的信息稳健地选择 conditional
split。最终叶 feature 仍然独立选择。

---

## 3. 理念判断

### 3.1 MetaGB 的核心理念

本次讨论后，MetaGB 更准确的核心理念应表述为：

> conditional features 负责构造局部区域，predictive features 负责在最终局部区域
> 内构造叶预测函数。

原版 README 中“联合选择最佳 `(ci, pi, bin)`，叶继续使用 `pi`”的描述更接近
`score_agg=0` 的切分解释，但原版真实代码最终仍会在叶节点重新选择 predictive
feature。因此，最终叶重选才是原版实现应复刻的真实语义。

### 3.2 `score_agg=0`

`score_agg=0` 使用某个 predictive feature 的最大 gain 驱动 conditional split，
属于最贪心的 MetaGB 版本。

它符合 MetaGB 理念，但容易让单个局部强 predictive feature 主导切分，并可能在深树
中产生过拟合。最终叶仍会根据叶内样本重新选择最佳 predictive feature，不一定继续
使用切分评分时的 feature。

### 3.3 `score_agg=1/2`

当前版本的 `score_agg=1/2` 符合 MetaGB 整体理念。

- `1` 使用 predictive-feature gains 的均值选择 conditional split；
- `2` 使用 predictive-feature gains 的中位数选择 conditional split；
- 最终叶仍独立选择最佳 predictive feature 并用于预测。

它们没有改变 MetaGB 的叶预测机制，只是在 conditional split 选择阶段降低贪心程度：

```text
agg=0：选择对某一个 predictive feature 最有利的切分
agg=1：选择对候选 predictive features 平均有利的切分
agg=2：选择对典型 predictive feature 有利、且抗极端值的切分
```

因此，`score_agg=1/2` 可以称为 **robust-split MetaGB**。

### 3.4 `score_agg=3`

当前版本的 `score_agg=3` 仍保持 MetaGB 的核心角色分离：

- 随机 predictive-feature gain 仅用于选择 conditional split；
- 最终叶仍会重新选择最佳 predictive feature；
- 预测仍使用最终叶保存的最佳 feature。

但 `score_agg=3` 的理论一致性弱于 `1/2`：

- 不同候选 split bin 可能由不同随机 predictive feature 评价；
- 不同 split gain 的比较不再完全基于相同聚合目标；
- 随机评分 feature 与最终叶最佳 feature 没有直接对应关系。

所以 `score_agg=3` 应定位为 **Randomized MetaGB** 或随机正则化扩展，而不是标准
或稳健目标优化版本。它并未违背 MetaGB 的核心角色分离理念，但不建议与
`score_agg=0/1/2` 一起无区别地称为标准 MetaGB。

---

## 4. 本次嵌入版修改

修改文件：

- `beforereview/xgboost-master/src/tree/updater_metagb_hist.cc`

### 4.1 修改前的问题

修改前标量分支的 split metadata 保存：

- split-level predictive feature；
- 按该 feature 计算的左右叶临时权重。

对于 `score_agg=1/2/3`，代码临时使用 `split_pred_features.front()`。虽然后续
`FindBestSplit -> BestLeaf` 通常会覆盖最终叶，但这种表达：

- 没有理论意义；
- 容易误解为最终叶固定使用第一个 predictive feature；
- 依赖后续节点访问隐式修正；
- 在未来流程调整、提前返回或新增分支时存在留下错误叶模型的风险。

### 4.2 修改后语义

当前标量分支中：

1. `MetaGBSplitInfo` 只保存 `split_bin`；
2. split aggregation 只选择 conditional split；
3. 应用 split 时不再保存某个真实 predictive feature 和临时叶权重；
4. 整棵树构建完成后，显式调用 `RefreshLeaves`；
5. `RefreshLeaves` 遍历全部最终叶，并调用 `BestLeaf` 重新选择最佳 predictive
   feature 和对应权重。

关键代码：

- `updater_metagb_hist.cc:491-545`：`BestLeaf` 计算并选择最佳 leaf feature；
- `updater_metagb_hist.cc:1391-1407`：最终叶统一刷新；
- `updater_metagb_hist.cc:1639`：标量树构建完成后强制调用最终叶刷新；
- `cpu_predictor.cc:91-99`：预测读取最终叶 feature，并执行
  `leaf_weight * predictive_feature_value`。

最终闭环为：

```text
score_agg=0/1/2/3 选择 conditional split
    -> 最终叶独立重选最佳 predictive feature 和 weight
    -> CPU predictor 使用该 feature 进行预测
```

---

## 5. 正确性测试

新增测试：

- `beforereview/xgboost-master/tests/python/test_metagb.py:83-113`

测试构造：

- feature 0 是第一个候选 predictive feature，但为无效常量；
- feature 1 能解释目标；
- 对 `score_agg=0/1/2/3` 分别训练；
- 检查模型文件中所有最终叶保存的 predictive feature。

预期和实际结果：

```text
所有 score_agg 的最终叶都选择 feature 1，而不是第一个 feature 0。
```

MetaGB 专属测试结果：

```text
18 passed
```

这证明当前标量 `current_metagb` 的 `score_agg=0/1/2/3` 均能：

1. 根据各自聚合方式选择 conditional split；
2. 在最终叶重新选择候选 predictive features 中 leaf score 最大的 feature；
3. 在预测阶段真正使用该 feature。

---

## 6. 修改前后性能验证

### 6.1 测试设计

分别使用修改前和修改后独立编译的动态库，在完全相同条件下运行：

- 数据集：airfoil、concrete；
- `max_depth={2,4,6}`；
- `n_estimators={50,200,500}`；
- `score_agg={0,1,2,3}`；
- 固定 3 折；
- 每个版本共 216 次拟合。

对比脚本：

- `beforereview/exp12/scripts/compare_score_agg_before_after.py`

结果：

- `beforereview/exp12/results/score_agg_before_after/paired_comparison.csv`

### 6.2 预测性能

修改前后逐项比较：

```text
最大 RMSE 差异：0.0
最大 R2 差异：0.0
预测指标发生变化的配置：0 / 216
```

因此，本次修改没有改变当前正常路径的预测结果。

这说明修改前最终叶通常已经被后续 `BestLeaf` 隐式覆盖。本次修改的主要价值是修正
代码契约、消除错误临时语义，并保证未来路径也遵守原版最终叶重选语义。

### 6.3 训练时间

修改前后总体训练时间基本一致：

```text
修改前总体中位数：0.08632 秒
修改后总体中位数：0.08599 秒
总体变化：-0.38%
配对运行变化中位数：-1.70%
```

按 `score_agg` 的配对耗时变化中位数：

| `score_agg` | 修改后耗时变化 |
|---:|---:|
| `0` | -1.84% |
| `1` | -1.64% |
| `2` | -1.36% |
| `3` | -1.86% |

这些差异属于小幅计时波动。可以确认本次修改没有带来可测的性能回退。

---

## 7. 当前 `score_agg` 小规模实验观察

在 airfoil 和 concrete 的 `3x3` 深度/树数量网格中，各方法成为当前配置最优值的次数：

| 数据集 | `agg=0` | `agg=1` | `agg=2` | `agg=3` |
|---|---:|---:|---:|---:|
| airfoil | 1 | 3 | 2 | 3 |
| concrete | 0 | 2 | 5 | 2 |

当前最佳配置：

| 数据集 | 深度 | 树数量 | `score_agg` | RMSE |
|---|---:|---:|---:|---:|
| airfoil | 4 | 500 | 3 | 2.1451 |
| concrete | 4 | 500 | 2 | 4.5395 |

观察：

- 最贪心的 `agg=0` 在这两个数据集上并不稳定占优；
- `agg=1/2` 的稳健切分在 concrete 上表现尤其稳定；
- `agg=3` 的随机化可能缓解深树贪心过拟合，但应作为随机化扩展解释；
- 两个数据集的小测试不足以证明任何 `score_agg` 在全部任务上普遍最优。

---

## 8. 主实验建议

### 8.1 理念优先的主实验

若主实验目标是回答标准 MetaGB 机制是否优于 XGBoost，建议：

```text
XGBoost vs MetaGB(score_agg=0)
```

`score_agg=0` 最直接体现 predictive-feature gain 驱动 conditional split 的贪心
MetaGB 机制，也与默认配置一致。

### 8.2 允许 MetaGB 稳健切分调优的主实验

如果目标是比较经过合理调优的 MetaGB 与 XGBoost，可在嵌套 CV 内搜索：

```text
score_agg in {0, 1, 2}
```

其中：

- `0`：greedy MetaGB；
- `1`：mean-aggregated robust-split MetaGB；
- `2`：median-aggregated robust-split MetaGB。

应明确声明 `score_agg` 是 MetaGB 专属超参数，并保证 XGBoost 获得公平的搜索预算。

### 8.3 `score_agg=3` 的定位

建议将 `score_agg=3` 单独作为：

```text
Randomized MetaGB
```

放入消融或扩展实验，不建议与 `0/1/2` 无区别地纳入标准 MetaGB 主方法搜索。

---

## 9. 最终结论

1. 原版 MetaGB 的 `score_agg` 负责选择 conditional split，最终叶 predictive
   feature 始终根据叶内样本重新选择。
2. 当前嵌入版已显式复刻这一语义，删除了 `split_pred_features.front()` 对标量最终
   叶模型的错误临时表达。
3. 当前 `score_agg=0/1/2/3` 都能在最终叶选择并预测时使用最佳 predictive feature。
4. `score_agg=1/2` 符合 MetaGB 理念，是降低贪心程度的稳健切分版本。
5. `score_agg=3` 保持 MetaGB 的核心角色分离，但属于理论一致性较弱的随机化扩展。
6. 本次修改没有改变现有正常路径的预测性能，也没有产生训练时间回退。

