# MetaGB 分类任务扩展实践路线

**日期：** 2026-05-23

## 1. 目标

在当前已嵌入 `xgboost-master` 的 CPU MetaGB 基础上，扩展分类任务支持，同时保留 MetaGB 的关键创新：

```text
条件特征 c：用于 split 路径
预测特征 p：用于叶子线性预测
leaf = w * x_p
```

分类扩展必须继续使用 MetaGB 的线性叶统计：

```text
sum grad * x_p
sum hess * x_p^2
w = - sum(grad * x_p) / (sum(hess * x_p^2) + lambda)
```

不允许退化成普通 XGBoost 常数叶。

## 2. 已确认设计点

1. 二分类同时支持：

```text
binary:logistic
binary:logitraw
```

二者主要区别是最终预测是否经过 sigmoid；训练时均在 raw margin 上使用 logistic loss 的 `grad/hess`。

2. 多分类第一版采用 class-wise scalar tree：

```text
每个 class 一棵 MetaGB scalar tree
第 k 棵树只更新 margin_k
```

后续预留 vector leaf 扩展方向，但不在第一版实现。

3. 多分类下每个 class 独立采样条件特征和预测特征。

4. 分类任务第一版仍只支持：

```text
CPU
tree_method="hist"
```

5. 分类版本继续保持与原版 MetaGB v3 一致的分箱逻辑：

```text
np.quantile 风格 bins
np.digitize(..., right=True) 风格离散化
训练统计用 bin 代表值
预测用原始 x
```

## 3. XGBoost 实践基础

XGBoost 的 objective 负责从当前 raw margin 计算 `grad/hess`。

平方误差：

```text
grad = margin - y
hess = 1
```

二分类 logistic：

```text
p = sigmoid(margin)
grad = p - y
hess = p * (1 - p)
```

多分类 softmax：

```text
每个样本有 K 个 margin
每个 class 有自己的 grad/hess
```

MetaGB updater 不需要手写交叉熵，只需要消费 XGBoost objective 已经计算好的 `grad/hess`。

## 4. 二分类实现路线

目标入口：

```python
XGBClassifier(
    objective="binary:logistic",
    enable_metagb=True,
)

XGBClassifier(
    objective="binary:logitraw",
    enable_metagb=True,
)
```

实现要点：

1. 放宽当前 MetaGB updater 的“回归限定”语义，改为支持单输出 objective。
2. 保持单输出梯度输入：

```text
Grad()->Shape(1) == 1
```

3. 确认 prediction cache 更新 raw margin：

```text
margin += learning_rate * w * x_p
```

4. `binary:logistic` 的 `predict_proba` 由 XGBoost objective 做 sigmoid。
5. `binary:logitraw` 输出 raw margin。
6. sklearn wrapper 中 `enable_metagb=True` 时继续使用普通 `DMatrix`，避免 `QuantileDMatrix` 缺少 `SparsePage`。

测试要求：

```text
XGBClassifier + binary:logistic
XGBClassifier + binary:logitraw
predict
predict_proba
logloss / accuracy
保存加载一致性
普通 XGBoost 分类 baseline 对比
```

## 5. 多分类实现路线

目标入口：

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=K,
    enable_metagb=True,
)

XGBClassifier(
    objective="multi:softmax",
    num_class=K,
    enable_metagb=True,
)
```

第一版采用 class-wise scalar tree。

每轮 boosting：

```text
for class k in 0..K-1:
    使用第 k 类 grad/hess 建一棵 MetaGB tree
    只更新 margin_k
```

利用 XGBoost `GBTree::DoBoost` 现有多输出 scalar-tree 路径：

```text
OutputLength() > 1:
    CopyGradient(..., gid, ...)
    BoostNewTrees(...)
    UpdatePredictionCache(..., out_preds[:, gid])
```

实现要点：

1. 保持 MetaGB updater 内部单输出树逻辑。
2. 让 XGBoost 外层按 class/group 调用 updater。
3. `UpdatePredictionCache` 接收切片后的单列 `out_preds[:, gid]`，只更新对应 class margin。
4. 每棵 class tree 独立保存 `metagb_leaf_features_`。
5. 每个 class 独立采样条件特征和预测特征。

测试要求：

```text
multi:softprob
multi:softmax
num_class=3/5
predict_proba 每行和为 1
mlogloss / accuracy
保存加载一致性
普通 XGBoost 多分类 baseline 对比
```

## 6. 暂不实现 vector leaf

第一版不实现 vector leaf。

后续 vector leaf 可能形式：

```text
leaf = [w_1, w_2, ..., w_K] * x_p
```

或：

```text
leaf_k = w_k * x_{p_k}
```

需要预留但暂不实现的问题：

- 每个 class 是否共享 `pred_feature`。
- 每个叶子存一个 `pred_feature` 还是 K 个。
- `RegTree` 序列化格式扩展。
- CPU predictor vector MetaGB leaf。
- split gain 跨 class 聚合。

## 7. 实施顺序

1. 文档固化本路线。
2. 实现二分类 `binary:logistic` 和 `binary:logitraw`。
3. 补二分类测试。
4. 实现/验证多分类 class-wise scalar tree。
5. 补多分类测试。
6. 回归测试确保 `reg:squarederror` 行为不退化。
7. 更新启动指南和分类支持说明。

## 8. 完成度记录

- [x] 路线设计确认。
- [x] 二分类实现。
- [x] 二分类测试。
- [x] 多分类 class-wise scalar tree 实现/验证。
- [x] 多分类测试。
- [x] 文档回填。

## 9. 当前验证结果

已通过专项测试：

```bash
PYTHONPATH=python-package python3 -m pytest tests/python/test_metagb.py -q
```

结果：

```text
13 passed
```

覆盖范围：

```text
reg:squarederror
binary:logistic
binary:logitraw
multi:softprob
multi:softmax
模型 JSON 保存/加载一致性
predict
predict_proba
```

小规模分类 smoke 结果：

```text
binary:logistic  acc=0.9167  logloss=0.2085
binary:logitraw  acc=0.9306
multi:softprob   acc=0.8056  logloss=0.4850
multi:softmax    acc=0.8056
```

## 10. 后续预留

后续仍需单独规划：

- vector leaf 版本的 MetaGB 多分类。
- 大规模分类数据集 benchmark。
- 分类任务下 `score_agg=1/2/3` 的系统评估。
- 分类任务下速度剖析和 histogram 复用优化。
