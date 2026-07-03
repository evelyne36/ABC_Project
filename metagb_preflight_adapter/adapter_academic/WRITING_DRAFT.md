# Writing Draft For Mechanism-Aware MetaGB Preflight Adapter

## 1. Meeting Version

我现在在做的不是一个普通的“用历史结果训练模型选择器”，而是一个围绕 MetaGB 本质创新点设计的 **preflight adapter**。MetaGB 相比 XGBoost 的核心区别是：XGBoost 叶子是常数，而 MetaGB 的叶子可以选择一个 predictive feature，形成 `beta * x_j` 的局部线性响应；同时它把用于划分区域的 conditional feature 和用于叶预测的 predictive feature 解耦。因此，我设计的 adapter 不是直接根据“MetaGB 最后赢不赢”来学习，而是先问一个机制问题：这个数据集内部是否存在 MetaGB 能利用的结构，比如局部区域内 residual/gradient 是否沿某个特征方向有强投影，划分区域的特征和局部预测特征是否可能不同，以及预测信号是否集中在少数 feature 上。

具体实现上，我从 MetaGB 的二阶叶目标出发。XGBoost 的常数叶 score 依赖 `sum(g)` 和 `sum(h)`，而 MetaGB 的一维局部线性叶 score 依赖 `sum(g*x_j)` 和 `sum(h*x_j^2)`。所以 adapter 用浅层条件划分构造局部区域，然后估计五个机制分量：局部线性叶优势、conditional/predictive 解耦、局部异质性、predictive feature 集中度，以及局部信号支撑。最后得到一个不使用 outcome label 的机制匹配分数。这个分数不是说“MetaGB 一定赢”，而是说“这个数据从结构上是否值得尝试 MetaGB”。

当前初步效果是：在 synthetic mechanism recovery 上结果比较干净。piecewise constant 这种 XGBoost 常数叶友好的数据得分低；local linear response、conditional-predictive decoupling 和 high-dimensional sparse/noisy 机制数据得分高，而且 true predictive feature hit rate 大约在 0.89 到 0.92 之间，说明这个分数确实在捕捉 MetaGB 的机制。在真实 regression/drop benchmark 上，结果更复杂：有些高分数据 MetaGB 最后没有赢，比如 airfoil 和 concrete。这不一定说明 adapter 错，而说明这些数据存在 MetaGB-compatible structure，但当前实现、分箱、正则、超参或噪声可能没有把机制转化成最终精度收益。后续我会把这些 high-score non-win case 作为 case analysis，用来解释 MetaGB 的适用边界。

所以这个贡献可以作为 MetaGB 论文里的第三部分：第一是方法本身，也就是 feature-selected local linear leaf；第二是系统实现，也就是嵌入 XGBoost hist；第三就是这个 mechanism-aware preflight adapter，用于解释和预测 MetaGB 的适用边界。它的价值不是替代完整实验，而是在大规模实验之前给出一个有理论依据的机制先验，帮助判断哪些数据值得进一步跑 MetaGB，哪些数据从结构上就不太匹配。

## 2. Academic Writing Version

We introduce a mechanism-aware preflight adapter for MetaGB. The goal is not to train a supervised model selector from historical benchmark outcomes, but to estimate whether a dataset exhibits the structural conditions under which MetaGB's inductive bias is applicable. MetaGB extends a constant-leaf boosted tree by replacing each leaf response with a feature-selected local linear response, `beta_l x_j`, and by decoupling conditional features used for sample routing from predictive features used inside leaves. This suggests an intrinsic applicability question: before running a full MetaGB experiment, can we determine whether the data contains local residual structure that a feature-selected leaf can exploit?

The adapter is derived from the same second-order objective that motivates MetaGB. In a local region `R`, a standard XGBoost leaf has score

```text
score_const(R) = (sum_i g_i)^2 / (sum_i h_i + lambda),
```

whereas a MetaGB one-dimensional local linear leaf has score

```text
score_j(R) = (sum_i g_i x_ij)^2 / (sum_i h_i x_ij^2 + lambda_meta).
```

Thus, MetaGB's additional capacity is structurally useful only when the local gradient residual has strong projected alignment with some predictive feature. Based on this observation, our preflight adapter constructs cheap conditional regions using shallow partitions and computes a mechanism-alignment score from five components: local linear leaf advantage, conditional/predictive decoupling, local response heterogeneity, predictive-feature concentration, and local signal support. Importantly, this score is computed without using final MetaGB-vs-XGBoost performance labels; benchmark outcomes are used only for retrospective audit.

This design differs from standard AutoML landmarking or meta-learning model selection. Rather than learning that MetaGB tends to perform well on certain historical datasets, the adapter evaluates whether a new dataset matches the structural premise of MetaGB's leaf objective. A high score therefore should not be interpreted as a guaranteed benchmark win. Instead, it indicates that the dataset contains mechanisms MetaGB is designed to exploit. If a high-scoring dataset does not yield a final accuracy gain, this points to a mechanism-to-optimization gap, such as binning effects, regularization, predictive-feature sampling, scale sensitivity, noise, or hyperparameter selection.

We evaluate the adapter through three complementary protocols. First, on synthetic mechanism-recovery tasks, piecewise-constant data receives low scores, while local-linear, conditional/predictive decoupling, and high-dimensional sparse/noisy mechanisms receive high scores. The score also recovers true predictive features with high hit rates in mechanism-aligned synthetic settings. Second, we perform a real benchmark retrospective audit: outcomes are not used to fit the score, but completed experiments are used to inspect whether high mechanism scores correspond to MetaGB gains or reveal high-score non-win cases requiring case analysis. Third, we conduct component ablations and sensitivity checks to assess whether the score is robust to plausible theory-driven weighting choices.

This preflight adapter forms the analysis contribution of the MetaGB system. The broader paper contributes: (i) a feature-selected local linear leaf formulation that strictly extends constant leaves through a bias feature; (ii) an integration of this formulation into the XGBoost histogram training pipeline; and (iii) a mechanism-aware applicability diagnostic that characterizes when the additional local linear capacity is structurally justified. This third contribution strengthens the empirical story from "MetaGB sometimes outperforms XGBoost" to "MetaGB has a computable mechanism prior that explains when its inductive bias is applicable and when ordinary constant-leaf boosting is likely sufficient."

## 3. Short Abstract-Style Version

We further propose a mechanism-aware preflight adapter for MetaGB. Instead of learning a model selector from historical benchmark wins, the adapter derives an intrinsic mechanism-alignment score from MetaGB's second-order leaf objective. The score estimates whether local residuals exhibit projected-gradient alignment with predictive features inside conditional regions, which is the structural condition under which feature-selected local linear leaves can improve over constant leaves. Synthetic mechanism-recovery experiments show that the score is low on piecewise-constant data and high on local-linear, conditional/predictive decoupling, and sparse high-dimensional mechanisms, with high true predictive-feature hit rates. Real benchmark audits then use completed experiments only as retrospective evidence, identifying both aligned gains and high-score non-wins for case analysis. This adapter provides a theory-guided applicability prior for MetaGB and helps turn heterogeneous benchmark behavior into an interpretable mechanism boundary.
