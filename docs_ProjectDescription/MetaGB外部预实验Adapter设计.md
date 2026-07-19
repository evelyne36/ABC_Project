# MetaGB 外部预实验 Adapter 设计

更新时间：2026-06-27  
定位：MetaGB 机制适配度预分析 / preflight adapter  
当前阶段：设计与初步 synthetic 机制验证框架

## 1. 背景与动机

当前实验结果显示，`current_metagb` 并不是对 `xgb_cpu` 的全面替代，而是在特定数据结构上更可能获得收益。直接对每个领域、每个数据集都运行大规模 Random300/Random600 nested-CV 实验，计算成本过高，也不利于形成清晰的理论边界。

因此，需要一个外部预实验性质的 adapter：

```text
在完整 MetaGB 训练前，先判断数据本身是否具备 MetaGB 能利用的结构。
```

这里的 adapter 不是一个监督式模型选择器，不是根据历史结果学习“MetaGB 最后赢不赢”。它更接近一个机制诊断器：

```text
给定数据 D，估计 D 是否匹配 MetaGB 的归纳偏置。
```

换句话说，它回答的问题不是：

```text
MetaGB 是否一定比 XGBoost 精度更高？
```

而是：

```text
这个数据是否存在 MetaGB 的 feature-selected local linear leaf 能利用的局部结构？
```

如果 adapter 给出高分，含义是“该数据从机制上值得尝试 MetaGB”。如果高分但最终实验没有赢，则应进入 case analysis，分析是否由于噪声、分箱、正则、超参、特征尺度、predictive-feature sampling 等原因导致机制没有转化为最终精度收益。

## 2. MetaGB 的核心机制

标准 XGBoost 的树叶是常数响应：

```text
leaf output = w
```

MetaGB 将叶响应扩展为 feature-selected local linear leaf：

```text
leaf output = beta_l * z_j(x)
```

其中 `z_j(x)` 可以是真实输入特征，也可以是 bias feature。bias feature 恒为 1，因此 MetaGB 在函数类层面包含普通常数叶：

```text
beta_l * 1 = beta_l
```

MetaGB 的关键创新点可以概括为两个：

1. **局部一维线性叶**：每个叶子不只输出常数，而是可以选择一个 predictive feature 做局部线性响应。
2. **conditional/predictive feature 解耦**：用于划分样本空间的 conditional feature 和用于叶内预测的 predictive feature 可以不同。

这意味着 MetaGB 最适合的数据结构不是任意复杂非线性，而是：

```text
在某些局部区域内，目标或残差能被某个特征方向解释；
并且划分区域的特征和局部预测的特征可能不是同一个。
```

## 3. 二阶目标下的理论动机

在一个局部区域 `R` 内，标准 XGBoost 常数叶的二阶近似 score 为：

```text
w* = - sum_i g_i / (sum_i h_i + lambda)

score_const(R) = (sum_i g_i)^2 / (sum_i h_i + lambda)
```

其中 `g_i` 和 `h_i` 分别是一阶梯度和二阶 Hessian。

MetaGB 对某个候选 predictive feature `x_j` 的局部线性叶为：

```text
f_R(x) = beta_j * x_j
```

对应的解析最优系数和 score 为：

```text
beta_j* = - sum_i g_i x_ij / (sum_i h_i x_ij^2 + lambda_meta)

score_j(R) = (sum_i g_i x_ij)^2 / (sum_i h_i x_ij^2 + lambda_meta)
```

因此，MetaGB 相对常数叶的结构性机会来自：

```text
max_j score_j(R) - score_const(R)
```

直观解释是：

```text
如果局部区域 R 内的 gradient / residual 在某个特征方向上有强投影，
那么 beta * x_j 这种局部线性叶就有可能比常数叶更合适。
```

为了避免常数项本身主导判断，预实验 adapter 可以在局部区域内先去掉局部均值，得到 centered residual：

```text
r_i = y_i - mean(y_R)
```

再对标准化后的候选特征 `z_ij` 计算 projected-gradient alignment：

```text
alignment_j(R) = (sum_i r_i z_ij)^2 / (sum_i z_ij^2 + lambda_meta)
```

这个量就是 adapter 的核心理论信号。它不依赖最终 MetaGB 是否赢，只反映数据局部结构是否匹配 MetaGB 的叶模型形式。

## 4. Adapter 的总体结构

当前 adapter 可以理解为五个机制探针的加权和：

```text
S_meta(D) =
  w1 * local_linear_leaf_advantage
+ w2 * conditional_predictive_decoupling
+ w3 * local_response_heterogeneity
+ w4 * predictive_feature_concentration
+ w5 * local_signal_support
```

其中每个分量都是根据 MetaGB 机制构造的 scoring probe，而不是由最终 benchmark outcome 训练出来的判别器。

默认权重可以先设为：

```text
local_linear_leaf_advantage      0.35
conditional_predictive_decoupling 0.25
local_response_heterogeneity     0.20
predictive_feature_concentration 0.10
local_signal_support             0.10
```

最终分数可分为三档：

```text
STRONG_MECHANISM_MATCH
MODERATE_MECHANISM_MATCH
WEAK_MECHANISM_MATCH
```

这些档位表示数据与 MetaGB 机制的匹配程度，不直接等同于最终精度提升概率。

## 5. 五个机制探针

### 5.1 局部线性叶优势

该探针对应 MetaGB 的 `beta * x_j` 叶模型。

实现步骤：

1. 用一个浅层树构造局部区域 `R`。
2. 在每个区域内去掉目标均值或当前 residual 的局部均值。
3. 对每个候选特征计算：

```text
alignment_j(R) = (sum_i r_i z_ij)^2 / (sum_i z_ij^2 + lambda_meta)
```

4. 取最大值或平均局部最大值：

```text
max_j alignment_j(R)
```

如果该值高，说明常数叶解释完局部均值后，仍存在某个特征方向可以解释剩余变化。这是 MetaGB 局部线性叶最直接的适用信号。

### 5.2 conditional / predictive feature 解耦

该探针对应 MetaGB 的角色分离思想：

```text
conditional feature: 用于划分样本区域
predictive feature: 用于叶内局部预测
```

实现步骤：

1. 浅层树使用的 split features 视为 conditional features。
2. 每个叶内 projected alignment 最大的特征视为 best predictive feature。
3. 若 best predictive feature 不在 split features 集合中，则该叶记为 decoupled。

统计：

```text
decoupled_leaf_fraction =
  number of decoupled leaves / number of valid leaves
```

如果该比例高，说明数据中经常出现“一个特征决定在哪里生效，另一个特征决定如何预测”的结构，这正是 MetaGB 相对普通树的核心优势之一。

### 5.3 局部响应异质性

该探针衡量：

```text
全局关系弱，但局部分区后关系变强。
```

实现方式：

```text
global_alignment = 全局 projected-gradient alignment
local_alignment  = 浅层分区后叶内 projected-gradient alignment 平均值

local_heterogeneity = local_alignment - global_alignment
```

如果局部 alignment 显著强于全局 alignment，说明数据具有局部响应异质性。MetaGB 先用 conditional feature 划分局部区域，再用 predictive feature 做叶响应，因此天然适合这种结构。

### 5.4 predictive feature 集中度

MetaGB 当前每个叶选择一个 predictive feature，而不是完整多变量线性模型。因此它更适合局部预测信号集中在少数特征上的数据。

在每个叶内，计算所有特征的 alignment score，定义：

```text
concentration(R) =
  max_j alignment_j(R) / sum_j positive_alignment_j(R)
```

如果该比例高，说明局部预测信号由少数特征主导，适合 feature-selected leaf。

### 5.5 局部信号支撑

前四个分量可能受噪声、分母过小或低方差特征影响。因此需要一个支撑项，避免只有比例好看但绝对信号很弱的情况。

该探针衡量局部 projected alignment 本身是否足够强：

```text
signal_support = normalized_strength(local_alignment)
```

它的作用是要求：

```text
不仅存在某种机制比例，而且该机制有足够统计支撑。
```

## 6. 为什么它不是 AutoML landmarking

普通 AutoML landmarking 或 meta-learning model selection 通常会从历史 benchmark outcome 中学习：

```text
哪些数据集上某模型更容易赢。
```

当前 adapter 不这么做。它不使用最终 `current_metagb` vs `xgb_cpu` 的胜负结果来拟合分数，而是从 MetaGB 的叶目标函数和机制结构推导出 intrinsic score。

因此它更适合被表述为：

```text
mechanism-aware applicability diagnostic
```

而不是：

```text
supervised model selector
```

真实 benchmark 结果只用于 retrospective audit：

```text
高机制分是否更容易出现 MetaGB gain？
高机制分但没赢的数据是否揭示优化、正则、分箱或噪声问题？
```

## 7. 高分但 MetaGB 不赢如何解释

高机制分表示：

```text
数据中存在 MetaGB 设计上可以利用的结构。
```

但它不保证当前实现和当前超参搜索一定能把该结构转化成最终精度提升。若出现高分但 MetaGB 不赢，应作为 case analysis，而不是直接判定 adapter 失败。

可能原因包括：

- 噪声较强，局部投影信号泛化不稳定；
- 特征尺度影响 `sum(g*x_j)` 与 `sum(h*x_j^2)`；
- 分箱统计与 prediction-time raw value 存在不完全一致；
- `pred_feature_fraction` 或 `score_agg` 没有选到合适配置；
- 正则项或树深限制使局部线性机制没有充分发挥；
- 当前 benchmark 的 XGBoost baseline 已能通过更多树或更深结构近似该关系。

因此，高分非赢案例的价值是帮助分析：

```text
机制存在，但机制到最终收益之间的转换环节出了什么问题？
```

## 8. 后续 synthetic 自洽验证路线

为了把该 adapter 做成自洽的理论贡献，后续应优先在 synthetic 数据上验证机制恢复能力，而不是先在真实数据上追求全部胜负预测。

建议 synthetic 设计至少包含四类。

### 8.1 piecewise constant

生成机制：

```text
if x0 > threshold:
    y = c1 + noise
else:
    y = c2 + noise
```

预期：

- XGBoost 常数叶已经匹配该机制；
- MetaGB 局部线性叶不应获得明显结构优势；
- adapter score 应较低。

### 8.2 local linear response

生成机制：

```text
if x0 > threshold:
    y = beta1 * x3 + noise
else:
    y = beta2 * x3 + noise
```

预期：

- 条件区域由 `x0` 决定；
- 叶内响应由 `x3` 决定；
- adapter score 应较高；
- true predictive feature hit rate 应较高。

### 8.3 conditional-predictive decoupling

生成机制：

```text
if x0 > threshold:
    y = beta1 * x3 + noise
else:
    y = beta2 * x5 + noise
```

预期：

- conditional feature 是 `x0`；
- predictive features 是 `x3` 和 `x5`；
- adapter 的 decoupling 分量应较高；
- selected predictive feature hit rate 应较高。

### 8.4 high-dimensional sparse / noisy

生成机制：

```text
少数 true predictive features + 大量 irrelevant features
```

预期：

- 若信号足够稀疏且集中，adapter 仍应给较高分；
- 随着 noise 增大或 irrelevant features 增多，score 应逐渐下降；
- predictive feature concentration 和 feature-hit rate 可作为核心指标。

## 9. 需要报告的验证指标

Synthetic 验证建议报告：

```text
mechanism_alignment_score
score by scenario family
true predictive feature hit rate
conditional feature hit rate
score under increasing noise
score under increasing dimensionality
score under decreasing sparsity
ablation without each component
sensitivity under different component weights
```

其中最关键的是：

1. piecewise constant 低分；
2. local linear / decoupling 高分；
3. high-dimensional sparse 场景中 score 随信号稀疏性和噪声合理变化；
4. true predictive feature hit rate 较高；
5. 去掉 projected-gradient 或 decoupling 分量后机制恢复能力下降。

## 10. 当前初步结果口径

基于当前 `adapter_academic` 初步 synthetic recovery：

- `piecewise_constant` 平均分较低；
- `local_linear_response` 平均分约 0.76；
- `conditional_predictive_decoupling` 平均分约 0.75；
- `high_dim_sparse_noise` 在高维下仍保持较高分；
- true predictive feature hit rate 约 0.89 到 0.92。

这说明当前分数确实能在 synthetic 机制上恢复 MetaGB 的关键适用结构。

真实 regression/drop audit 中，高分不一定对应最终 MetaGB win。例如 `airfoil_self_noise` 和 `concrete_compressive_strength` 属于 high-score non-win case。这类结果不应简单视为 adapter 失败，而应进入机制到优化转换的 case analysis。

## 11. 论文贡献定位

该外部 adapter 可以作为 MetaGB 论文的第三个贡献：

1. **Method**：提出 feature-selected local linear leaf，并通过 bias feature 包含常数叶。
2. **System**：将 MetaGB 嵌入 XGBoost hist updater，支持 regression、binary、multiclass。
3. **Analysis**：提出 mechanism-aware preflight adapter，用于解释并预判 MetaGB 的适用边界。

推荐论文表述：

```text
We introduce a mechanism-aware preflight adapter for MetaGB. Instead of
learning a supervised model selector from benchmark outcomes, the adapter
derives an intrinsic mechanism-alignment score from MetaGB's second-order
leaf objective. The score estimates whether local residuals exhibit
projected-gradient alignment with predictive features inside conditional
regions, which is the structural condition under which feature-selected
local linear leaves can improve over constant leaves.
```

中文表述：

```text
我们进一步提出一个面向 MetaGB 的机制感知预实验 adapter。它不是根据历史 benchmark
胜负训练模型选择器，而是从 MetaGB 的二阶叶目标出发，构造数据与 feature-selected
local linear leaf 机制之间的内在匹配度分数。该分数用于判断数据中是否存在局部残差
沿 predictive feature 强投影、conditional/predictive feature 解耦、以及局部预测信号
稀疏集中的结构，从而在完整训练前给出 MetaGB 是否值得尝试的机制先验。
```

## 12. 后续工作清单

短期建议：

- 完善 synthetic generator，加入 graded noise、graded sparsity、graded dimensionality；
- 记录 true conditional / predictive features；
- 计算 feature-hit rate 和 conditional-hit rate；
- 做 component ablation 和 weight sensitivity；
- 在 synthetic 上验证 pred fraction 偏好是否与 score 分量一致。

中期建议：

- 将真实 benchmark audit 扩展到 regression/drop 之外，加入 onehot、binary、multiclass；
- 对 high-score non-win case 做专门 case analysis；
- 对 high-score 数据做 fixed-backbone pred sweep，检查是否存在合适 pred fraction 但 CV 未选出；
- 如果能从训练后的 MetaGB 模型中提取 leaf predictive features，进一步做真实数据上的 selected feature stability 分析。

长期建议：

- 将 adapter 从外部分析器扩展为实验调度器；
- 根据 score 自动选择 `USE_XGB / TRY_MGB_LIGHT / TRY_MGB_FULL`；
- 根据 predictive concentration 推荐 `pred_feature_fraction` 或 predictive-feature whitelist；
- 将该机制先验与实际小预算 pilot 结合，形成更稳健的 preflight protocol。
