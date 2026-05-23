# MetaGB 分类任务支持说明

**日期：** 2026-05-21

## 1. 结论

当前嵌入到 `xgboost-master` 的 MetaGB 已经作为 XGBoost CPU updater 接入，可以通过：

```python
enable_metagb=True
```

启用。当前版本已经完成并验证：

```text
reg:squarederror
binary:logistic
binary:logitraw
multi:softprob
multi:softmax
```

其中多分类第一版采用 class-wise scalar MetaGB tree；vector leaf 多分类仍是后续预留方向。

它不能像普通 XGBoost tree updater 一样，直接适配所有回归和分类任务。原因不是单纯“loss 没换成交叉熵”，而是 MetaGB 的树结构、叶子形式、split 统计量、prediction cache 和多输出处理方式都比普通 XGBoost 更特殊。

## 2. XGBoost 为什么能较通用地支持多种任务

普通 XGBoost 的树叶是常数叶：

```text
leaf = w
```

只要 objective 能提供每个样本的一阶梯度和二阶梯度：

```text
grad_i = dL / dŷ_i
hess_i = d²L / dŷ_i²
```

普通 XGBoost 的 split 和 leaf weight 就可以使用统一形式：

```text
G = sum grad_i
H = sum hess_i
w = -G / (H + lambda)
```

平方误差回归：

```text
grad = ŷ - y
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

因此，普通 XGBoost 的树 updater 不需要理解任务细节，只要消费 objective 提供的 `grad/hess`。

## 3. MetaGB 与普通 XGBoost 的核心差异

MetaGB 当前版本使用条件特征和预测特征分离的结构。

普通 XGBoost split：

```text
if x_c <= threshold:
    leaf = w_left
else:
    leaf = w_right
```

MetaGB split：

```text
if x_c <= threshold:
    leaf = w_left * x_p
else:
    leaf = w_right * x_p
```

其中：

- `x_c` 是条件特征，用于决定样本走左子树还是右子树。
- `x_p` 是预测特征，用于叶子里的线性预测。
- bias 特征由内部用 `pred_feature == num_features` 表示，相当于常数 1。

因此 MetaGB 的叶权重统计量不是普通 XGBoost 的：

```text
sum grad
sum hess
```

而是：

```text
sum grad * x_p
sum hess * x_p^2
```

叶权重为：

```text
w = - sum(grad * x_p) / (sum(hess * x_p^2) + lambda)
```

这意味着 MetaGB 不只是换了 split gain，而是改变了叶子预测函数。

## 4. 为什么不能直接适配所有任务

### 4.1 当前实现只完成单输出回归闭环

当前 C++ updater 主要围绕单输出标量树实现：

```text
一棵树输出一个标量 margin 增量
```

早期版本测试和 benchmark 主要围绕：

```python
objective="reg:squarederror"
XGBRegressor(...)
```

当前已补充 `XGBClassifier`、`binary:logistic`、`binary:logitraw`、`multi:softprob`、`multi:softmax` 的 smoke 测试、概率输出测试和保存加载一致性测试。

### 4.2 二分类虽然数学上可扩展，但工程语义还没有验证

二分类 logistic 本质上仍是单输出 margin：

```text
margin += tree(x)
prob = sigmoid(margin)
```

因此 MetaGB 理论上可以使用 logistic objective 提供的：

```text
grad = p - y
hess = p * (1 - p)
```

继续计算：

```text
sum grad * x_p
sum hess * x_p^2
```

当前版本已通过专项 smoke 测试覆盖以下问题：

- `XGBClassifier` sklearn 包装入口。
- prediction cache 更新 raw margin。
- `predict_proba` 输出概率。
- 模型保存/加载后 `predict_proba` 一致。
- 小规模 logloss、accuracy smoke。

所以当前可以声明第一版支持二分类，但仍需在更多真实分类数据集上做系统 benchmark。

### 4.3 多分类不是一个标量输出问题

多分类 softmax 中，每个样本有 `K` 个 class margin：

```text
margin_i = [m_i1, m_i2, ..., m_iK]
```

每个 class 都有自己的梯度和 Hessian：

```text
grad_{i,k}
hess_{i,k}
```

普通 XGBoost 通常每轮为每个 class 建一棵 scalar tree：

```text
tree_0 更新 class 0 的 margin
tree_1 更新 class 1 的 margin
...
tree_K-1 更新 class K-1 的 margin
```

当前第一版采用 XGBoost 现有 class-wise scalar-tree 路径处理：

- 一轮 boosting 中多棵 class-wise MetaGB tree 的组织。
- 每棵树读取对应 class 的 `grad/hess`。
- prediction cache 更新对应 class 的 `out_preds[:, k]`。
- 多分类 `predict_proba` 的 softmax 变换。
- 每个 class 独立采样条件特征和预测特征。
- 每棵 class tree 独立保存、加载和预测 MetaGB leaf feature。

因此当前可以声明第一版支持 class-wise scalar-tree 多分类；vector leaf 多分类尚未实现。

### 4.4 MetaGB 不能完全复用普通 XGBoost hist updater 的通用性

普通 XGBoost hist 的统计量是二维的：

```text
feature × bin
```

每个 bin 存：

```text
sum grad
sum hess
```

MetaGB 的统计量近似变成三维：

```text
condition_feature × bin × prediction_feature
```

每个 bin 需要存：

```text
sum grad * x_p
sum hess * x_p^2
```

所以 MetaGB 无法只改一个 objective 或只改一个 leaf weight 公式，就自动继承普通 XGBoost 对所有任务的支持。

## 5. 分类任务实现难点

MetaGB 做分类的难点主要不是“能不能计算交叉熵”，而是分类任务优化的是 raw margin 上的分类损失，而 MetaGB 又把普通常数叶改成了依赖预测特征的线性叶：

```text
leaf = w * x_p
```

这会带来以下实现难点。

### 5.1 必须更新 raw margin，而不是 probability

分类任务中，树模型累加的是 raw margin。

二分类：

```text
margin -> sigmoid(margin) -> probability
```

多分类：

```text
margins -> softmax(margins) -> class probabilities
```

MetaGB tree 必须输出 margin 增量：

```text
margin += learning_rate * w * x_p
```

不能直接更新 probability，否则下一轮 objective 计算出的 `grad/hess` 会错误。

### 5.2 二分类需要验证 logistic grad/hess 与线性叶兼容

二分类理论上可以直接使用 XGBoost logistic objective 提供的：

```text
grad = sigmoid(margin) - y
hess = p * (1 - p)
```

然后 MetaGB 继续使用：

```text
sum grad * x_p
sum hess * x_p^2
```

但工程上必须验证：

- `base_score` 是否正确转换成 raw margin。
- prediction cache 是否始终更新 raw margin。
- `predict_proba` 是否正确走 sigmoid。
- 模型保存/加载后概率预测是否一致。
- logloss、accuracy 等分类指标是否稳定。

### 5.3 多分类需要处理 K 个 margin

多分类不是一个标量输出，而是每个样本有 `K` 个 class margin：

```text
margin_i = [m_i1, m_i2, ..., m_iK]
```

每个 class 都有自己的：

```text
grad_{i,k}
hess_{i,k}
```

当前 MetaGB 是单输出 scalar tree。因此多分类扩展必须明确：

- 一棵树更新哪个 class。
- 每轮是否建 `K` 棵 MetaGB tree。
- 每棵树读取哪一列 `grad/hess`。
- prediction cache 应更新 `out_preds[:, k]` 的哪一列。

最稳妥的方案是 class-wise scalar tree：

```text
第 k 棵 MetaGB 树只使用第 k 类 grad/hess
第 k 棵 MetaGB 树只更新 margin_k
```

### 5.4 预测特征是否跨类别共享需要设计

多分类下还需要决定：

- 每个 class 是否独立采样条件特征和预测特征。
- 所有 class 是否共享同一组条件特征和预测特征。
- 每个叶子保存一个 `pred_feature`，还是每个 class 保存一个 `pred_feature`。

如果采用 class-wise scalar tree，这些问题相对简单，因为每棵树仍然只对应一个 class margin。若直接做 vector leaf，这些问题会显著复杂化。

### 5.5 模型存储和预测器需要按分类语义扩展

当前 MetaGB 通过 sidecar 保存每个叶子的：

```text
metagb_leaf_features_
```

回归时一棵树只输出一个标量。分类扩展时必须保证：

- 每棵 class tree 的 `pred_feature` 正确保存。
- 预测时输出加到正确 class margin。
- JSON 保存/加载后不丢失 MetaGB leaf feature。
- CPU predictor 与 training-time prediction cache 逻辑一致。

### 5.6 sklearn 接口需要适配 XGBClassifier

当前主要验证的是：

```python
XGBRegressor
objective="reg:squarederror"
```

分类任务需要补充：

```python
XGBClassifier
objective="binary:logistic"
objective="multi:softprob"
objective="multi:softmax"
predict
predict_proba
```

并增加分类指标测试：

```text
logloss
mlogloss
error
accuracy
```

### 5.7 小结

MetaGB 分类的核心难点是：

```text
分类任务优化 raw margin 上的交叉熵；
MetaGB 的叶子又是 w * x_p 的线性叶；
二分类要确保这套线性叶正确作用在 logistic margin 上；
多分类还要处理每个 class 的独立 margin、梯度列、树组织和 prediction cache。
```

因此，二分类是自然的下一步；多分类应优先采用 class-wise scalar tree，而不是直接实现 vector leaf。

## 6. 后续支持二分类的推荐路线

第一阶段已完成二分类支持：

```python
XGBClassifier(
    objective="binary:logistic",
    enable_metagb=True,
)
```

目标语义：

```text
MetaGB tree 输出 raw margin 增量
XGBoost objective 负责 sigmoid 概率转换
```

已完成：

1. 允许 `enable_metagb=True` 与 `objective="binary:logistic"` 共存。
2. 保持单输出梯度输入：

```text
Grad()->Shape(1) == 1
```

3. 确认 updater 使用 logistic objective 提供的 `grad/hess`。
4. 确认 prediction cache 更新 raw margin。
5. 检查 binary base score 到 margin 的转换。
6. 增加测试：

```text
fit
predict
predict_proba
logloss
accuracy
模型保存/加载一致性
```

推荐表述：

```text
MetaGB extends to binary classification by using first- and second-order derivatives of logistic loss on raw margins.
```

## 7. 后续支持多分类的推荐路线

第二阶段再支持多分类：

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=K,
    enable_metagb=True,
)
```

当前已采用 class-wise scalar tree，而不是一开始做 vector leaf。

目标机制：

```text
每轮 boosting 为每个 class 建一棵 MetaGB scalar tree
第 k 棵树使用第 k 类的 grad/hess
第 k 棵树只更新 margin_k
```

形式为：

```text
margin_k += learning_rate * w_k * x_p
```

已完成第一版：

1. 检查并适配 XGBoost `GBTree::DoBoost` 的多分类 scalar-tree 路径。
2. 支持按 class/group 调用 MetaGB updater。
3. prediction cache 按 class 更新对应输出列。
4. `XGBClassifier` 支持 `multi:softprob` 和 `multi:softmax`。
5. 增加测试：

```text
num_class=3/5
predict_proba 每行和为 1
multi:softprob logloss
multi:softmax predict
模型保存/加载一致性
与普通 XGBoost baseline 对比
```

## 8. 不建议优先做 vector leaf

另一种理论方案是 vector leaf：

```text
leaf = [w_1, w_2, ..., w_K] * x_p
```

或者：

```text
leaf_k = w_k * x_{p_k}
```

但这会引入更多设计问题：

- 每个 class 是否共享同一个 `pred_feature`。
- 每个叶子存一个预测特征，还是每个 class 存一个预测特征。
- `RegTree` 序列化格式需要扩展。
- CPU predictor 需要支持 MetaGB vector leaf。
- split gain 需要聚合多个 class 的 MetaGB score。

因此第一版多分类更建议使用 class-wise scalar tree。

## 9. 总结

MetaGB 不能直接嵌入 XGBoost 后自动适配所有回归和分类任务，原因是：

```text
普通 XGBoost 的叶子是常数叶，任务差异主要由 objective 的 grad/hess 吸收。
MetaGB 的叶子是 w * x_p，split 和 leaf 都依赖额外预测特征维度。
分类任务尤其是多分类还涉及 margin 维度、概率变换和 prediction cache 的输出列管理。
```

当前扩展状态是：

```text
已完成：平方误差回归
已完成：binary:logistic / binary:logitraw 二分类
已完成：multi:softprob / multi:softmax class-wise scalar MetaGB tree
后续：vector leaf 多分类
后续：更多真实分类数据集 benchmark 与速度优化
```
