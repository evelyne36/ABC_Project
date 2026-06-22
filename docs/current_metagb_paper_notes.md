# current_metagb Implementation and Paper Notes

## 1. High-Level Idea

`current_metagb` extends the standard XGBoost tree booster by replacing constant leaf responses with feature-selected local linear responses. In standard XGBoost, each leaf stores a scalar value `w`, and a sample contributes `w` once it reaches that leaf. In current_metagb, each leaf stores both a coefficient and a selected predictive feature:

```text
leaf output = beta_l * z_j(x)
```

Here `z_j(x)` is the predictive feature selected for leaf `l`. It can be a real input feature or a bias feature. The bias feature is always equal to 1, so the model can degenerate to the ordinary XGBoost constant-leaf form:

```text
beta_l * 1 = beta_l
```

This gives current_metagb a useful compatibility property: it is a strict extension of ordinary scalar-leaf gradient boosting rather than a completely separate model family.

The core modeling idea is to decouple two roles that are tied together in ordinary tree learning:

- **Conditional features** are used to split the sample space.
- **Predictive features** are used inside leaves to form local predictions.

In other words, a node split can use one feature to define a region, while the leaf in that region can use another feature as a local linear predictor.

## 2. Comparison With Standard XGBoost

Standard XGBoost uses a second-order Taylor approximation of the objective. For a leaf containing samples `i`, the optimal constant leaf weight is:

```text
w* = - sum_i g_i / (sum_i h_i + lambda)
```

The corresponding score is:

```text
score = (sum_i g_i)^2 / (sum_i h_i + lambda)
```

Here `g_i` and `h_i` are the first-order and second-order gradients of the loss with respect to the current raw prediction.

current_metagb changes the leaf function from a constant `w` to a one-dimensional linear response `beta * x_j`. For a candidate predictive feature `x_j`, the optimal coefficient is:

```text
beta_j* = - sum_i (g_i * x_ij) / (sum_i (h_i * x_ij^2) + lambda_meta)
```

The corresponding score becomes:

```text
score_j = (sum_i (g_i * x_ij))^2 / (sum_i (h_i * x_ij^2) + lambda_meta)
```

Therefore, the core statistics used by current_metagb are:

```text
grad_x  = sum_i (g_i * x_ij)
hess_xx = sum_i (h_i * x_ij^2)
hess    = sum_i h_i
```

By contrast, standard XGBoost mainly needs:

```text
grad = sum_i g_i
hess = sum_i h_i
```

This is the main algorithmic distinction. current_metagb evaluates splits according to how well the resulting child leaves can fit local feature-scaled responses, not merely constant residual corrections.

## 3. Split Search in current_metagb

At each node, current_metagb evaluates candidate conditional splits. For each split, it considers how the samples are divided into left and right children. For each child, it also evaluates candidate predictive features.

For a candidate split and predictive feature, the gain is:

```text
gain = score(left) + score(right) - score(parent)
```

where each score is computed with the MetaGB leaf score:

```text
score = grad_x^2 / (hess_xx + lambda_meta)
```

The selected split is the split that maximizes this MetaGB gain. The selected predictive feature is stored as leaf metadata. During prediction, if a sample reaches a MetaGB leaf, the tree output is:

```text
leaf_value * selected_feature_value
```

If the selected feature is the bias feature, the output is simply the leaf value.

## 4. Regression Execution Process

For regression, the objective is usually squared error. XGBoost produces gradients such as:

```text
g_i = y_hat_i - y_i
h_i = 1
```

The current_metagb regression procedure is:

1. Compute current raw predictions from the existing ensemble.
2. Compute first-order and second-order gradients from the regression objective.
3. Sample or choose conditional features for split search.
4. Sample or choose predictive features for leaf response search.
5. Add the bias feature to the predictive feature set.
6. For each active node, compute the best current leaf response `beta * z_j(x)`.
7. Enumerate conditional feature split candidates.
8. For each split, compute MetaGB statistics for left and right children.
9. Select the split with the largest MetaGB gain.
10. Store the split feature and split threshold in the tree.
11. Store the selected predictive feature and coefficient in each leaf.
12. During prediction, traverse the tree as usual, then output `beta_l * z_j(x)`.

In regression, current_metagb acts like a boosted tree ensemble whose leaves are lightweight local linear models. It can capture patterns where the local response is not well represented by a constant but is well approximated by one selected feature.

## 5. Binary Classification Execution Process

For binary classification, current_metagb uses the same tree-building mechanism, but the gradients come from the logistic objective. With raw margin `m_i`, probability:

```text
p_i = sigmoid(m_i)
```

The gradients are:

```text
g_i = p_i - y_i
h_i = p_i * (1 - p_i)
```

The current_metagb binary classification procedure is:

1. Compute raw margins from the current ensemble.
2. Convert margins to probabilities with sigmoid.
3. Compute logistic gradients and Hessians.
4. Build a MetaGB tree using the same conditional-predictive split procedure.
5. Add the tree output to the raw margin.
6. Apply sigmoid at prediction time to obtain probabilities.

Compared with ordinary XGBoost binary classification, current_metagb replaces each leaf's constant logit correction with a feature-scaled local correction:

```text
delta margin = beta_l * z_j(x)
```

This is useful when the effect of a feature on the positive-class logit differs across regions of the feature space.

## 6. Multiclass Classification: Scalar Path

The main and most stable multiclass path in current_metagb is the `k-class scalar tree` path. This follows the standard XGBoost `one_output_per_tree` strategy.

For `K` classes, each boosting round trains `K` scalar trees:

```text
round t:
  tree for class 0
  tree for class 1
  ...
  tree for class K-1
```

The objective `multi:softprob` produces a gradient matrix:

```text
shape = [n_samples, K]
```

The booster then trains one scalar MetaGB tree for each class by passing one gradient column at a time into the MetaGB updater.

The scalar multiclass execution process is:

1. Compute current raw margins for all classes.
2. Apply softmax to get class probabilities.
3. Compute gradient and Hessian for each sample-class pair.
4. For each class `k`, extract the class-specific gradient column.
5. Train one scalar MetaGB tree for class `k`.
6. Each class-specific tree has its own conditional splits.
7. Each class-specific tree has its own selected leaf predictive features.
8. Add all class tree outputs to their corresponding class margins.
9. Apply softmax at prediction time.

This scalar path is currently the recommended path for accuracy. Its main advantage is that each class gets its own tree structure. Class `0` can split the space differently from class `1`, which is important when different classes are separated by different conditional rules.

## 7. Multiclass Classification: Vector Branch

The current vector branch is an experimental path enabled by:

```text
multi_strategy = "multi_output_tree"
enable_metagb = true
```

Instead of training `K` scalar trees per boosting round, it trains one shared tree whose leaves output `K` values. The vector branch has been extended so that each class can select its own predictive feature in each leaf:

```text
class k leaf output = beta_lk * z_jk(x)
```

The model stores these class-specific predictive features in:

```text
metagb_leaf_features_vector[node, class]
```

During prediction, for a sample reaching leaf `l`, each class receives:

```text
margin_k += beta_lk * z_jk(x)
```

The vector branch therefore supports class-specific predictive features, but it still shares a single conditional tree structure across all classes.

## 8. Why the Vector Branch Is Weaker Than Scalar

The vector branch remains weaker than the scalar path mainly because of shared tree structure.

In the scalar path:

```text
class 0 has its own conditional splits
class 1 has its own conditional splits
...
class K-1 has its own conditional splits
```

In the vector path:

```text
all classes share the same conditional splits
```

Even though the vector branch now allows each class to choose its own predictive feature inside each leaf, the sample routing path is still shared. If different classes require different partitions of the feature space, the vector tree must compromise.

This creates a structural trade-off:

```text
scalar path:
  higher expressiveness
  more trees
  class-specific structure

vector path:
  fewer trees per round
  shared structure
  class-specific leaf responses
  lower structural flexibility
```

Empirically, the original vector branch used sum aggregation across class gains:

```text
split_score = sum_k gain_k
```

This can be dominated by one class. A split that strongly helps class `0` but hurts or barely helps other classes can still be selected. This made the vector branch unstable and sometimes divergent.

The current version changes the default vector split aggregation to:

```text
split_score = min_k gain_k
```

This forces a shared split to be useful for all classes, improving stability.

Example smoke result on `maternal_health_risk.csv`:

```text
scalar 20 rounds:
  logloss = 0.7600
  trees   = 60

vector default 20 rounds:
  logloss = 0.8672
  trees   = 20

vector default 60 rounds:
  logloss = 0.7979
  trees   = 60

vector sum aggregation 60 rounds:
  logloss = 1.7527
```

This shows that the min aggregation substantially improves the vector branch, but the scalar path remains stronger because it preserves class-specific tree structures.

## 9. Current Vector Stabilization Parameters

Two vector-specific parameters are available:

```text
metagb_vector_score_agg
```

Controls how per-class split gains are aggregated:

```text
0 = sum across classes
1 = mean across classes
2 = min across classes
```

The default is currently:

```text
metagb_vector_score_agg = 2
```

This is the most stable choice in the current smoke tests.

The second parameter is:

```text
metagb_vector_leaf_scale
```

It applies an additional scale to vector leaf weights only. This is useful because vector trees update all class margins at once, which can be more aggressive than scalar per-class updates.

The default is:

```text
metagb_vector_leaf_scale = 1.0
```

Lower values can improve stability in some settings, but the current best smoke result used min aggregation with scale 1.0.

## 10. Innovation Points for Paper Writing

### 10.1 Conditional-Predictive Decoupling

current_metagb separates the features used for sample partitioning from the features used for leaf prediction. This differs from ordinary GBDT, where leaves are constants and split features only define regions.

A possible paper phrase:

```text
We decouple conditional partitioning from local predictive response by allowing each leaf to select a predictive feature independent of the split feature used to route samples.
```

### 10.2 Feature-Selected Local Linear Leaves

Instead of fitting a full linear model in each leaf, current_metagb selects a single predictive feature and learns one coefficient. This provides a lightweight middle ground between constant leaves and full linear leaves.

A possible paper phrase:

```text
The proposed leaf model is a sparse one-dimensional local linear model, selected from candidate predictive features using second-order gradient statistics.
```

### 10.3 Strict Extension of XGBoost Leaves

Because the bias feature is always included, current_metagb can exactly recover the constant-leaf behavior of XGBoost.

```text
By including a constant bias feature in the predictive feature set, the proposed method strictly generalizes standard constant-leaf gradient boosting.
```

### 10.4 Unified Second-Order Optimization

The method works with any objective that provides first-order and second-order gradients. This makes it compatible with regression, binary classification, and multiclass classification.

```text
The method preserves the second-order functional gradient framework of XGBoost while replacing the leaf scoring rule with a feature-conditioned local response score.
```

### 10.5 XGBoost-Compatible System Integration

current_metagb is implemented as a CPU hist updater inside XGBoost. It reuses XGBoost's objective functions, data matrix, training loop, model serialization, and Python/sklearn interface.

```text
Rather than introducing a standalone learner, we implement the method as an XGBoost-compatible tree updater, allowing direct reuse of mature objectives, data structures, and prediction infrastructure.
```

The retained implementation is the CPU histogram updater with an automatic histogram-build strategy:

```text
METAGB_HIST_BUILD_MODE=auto|row_serial|cf_parallel|row_parallel
```

The default `auto` path selects `cf_parallel` for wider or higher-work nodes and keeps lighter serial-style behavior for small tables where parallel overhead can dominate. This keeps the paper implementation practical for the random-search experiments without changing the mathematical scoring rule.

Histogram subtraction was rechecked as a system optimization but is disabled by default. On the representative tested paths it slowed training because the current implementation must preserve or copy a full 3D node histogram before it knows whether subtraction can be reused. The retained default is therefore:

```text
metagb_hist_subtract = false
```

This is an implementation choice, not a modeling change. The option remains available for ablation, but the main experiments should report the default no-subtraction path.

For experiment scale, the current repository contains two random-search suites:

```text
random600: main full-scale suite, 600 candidates per model
random300: lighter suite, 300 candidates per model
```

For a fixed seed, `random300` uses the first 300 paired shared-backbone candidates
from `random600`. XGBoost and current_metagb use identical common backbone
parameters at each candidate index. Current MetaGB jointly balances 10
`pred_feature_fraction` values and `score_agg` in `{0, 1, 2}`: each of the 30
combinations appears 10 times in Random300 and 20 times in Random600.

When reporting runtime, it is important to separate three different questions:

```text
1. Modeling overhead versus standard XGBoost
2. Engineering speedup versus the original standalone MetaGB
3. End-to-end experiment throughput under nested CV
```

`current_metagb` should not be positioned as a speed replacement for ordinary `xgb_cpu`. Standard XGBoost only accumulates ordinary gradient/hessian histograms, while current_metagb must evaluate conditional-feature by predictive-feature statistics for local feature-selected leaf responses. The method is computationally heavier by design; the system goal is to make this richer leaf model practical inside XGBoost, not to match the raw training speed of constant-leaf hist trees.

For a fair speed comparison against the standalone MetaGB baseline, use `n_jobs=1`. The standalone CFFI MetaGB implementation does not use XGBoost's `n_jobs`; using `n_jobs=4` or `n_jobs=16` for current_metagb measures additional OpenMP/XGBoost integration benefits and should be reported separately.

The current timing interpretation is:

```text
small tables:
  current_metagb may be slower than standalone MetaGB because fixed updater overhead dominates

ordinary medium/large tables:
  current_metagb is roughly comparable to slightly faster under n_jobs=1

wide tables:
  current_metagb can be clearly faster even under n_jobs=1, and benefits further from cf_parallel
```

Wide datasets such as `QSAR-TID-11.csv` should be reported separately rather than hidden inside a single average. They are the clearest stress test for the conditional-feature by predictive-feature histogram build and for the `cf_parallel` optimization.

For full Random300/Random600 paper experiments on a 16-core server, prefer task-level parallelism:

```text
--task-workers 16 --n-jobs 1
```

This maximizes nested-CV throughput while keeping each model fit on a consistent single-thread setting. `--task-workers 4 --n-jobs 4` is better treated as a targeted wide-table latency experiment, not as the default paper sweep configuration.

### 10.6 Multiclass Structure Study

The scalar and vector multiclass paths provide an interesting ablation:

- scalar path tests class-specific tree structures;
- vector path tests shared tree structure with class-specific leaf responses.

The current results suggest that class-specific tree structure remains important for multiclass tabular prediction.

```text
Our vector-leaf variant shows that class-specific predictive leaves alone are insufficient to fully match class-specific tree structures, highlighting the importance of conditional partition flexibility in multiclass boosting.
```

## 11. Recommended Paper Positioning

For the main method, use the scalar current_metagb path:

```text
one_output_per_tree for multiclass
scalar MetaGB tree for regression and binary classification
```

This path is more expressive and currently more accurate.

The vector branch should be positioned as an exploratory compression or structural-sharing variant:

```text
The vector variant reduces the number of trees per boosting round by sharing conditional structure across classes, while preserving class-specific predictive leaf features. However, this sharing can reduce accuracy when class-specific partitions are important.
```

## 12. Concise English Summary

```text
current_metagb extends gradient boosted decision trees by replacing constant leaf responses with feature-selected local linear responses. It decouples conditional features used for tree partitioning from predictive features used for leaf response, while preserving the second-order optimization framework of XGBoost. The method strictly generalizes constant-leaf boosting through a bias predictive feature and can be applied uniformly to regression, binary classification, and multiclass classification. In multiclass tasks, the scalar one-output-per-tree variant retains class-specific tree structures and provides stronger accuracy, while the vector variant shares conditional structure across classes and supports class-specific predictive leaf features as a compact but less flexible alternative.
```

## 13. Concise Chinese Summary

```text
current_metagb 将传统梯度提升树的常数叶子扩展为特征选择的局部线性叶子。该方法将用于样本空间划分的 conditional feature 与用于叶子预测的 predictive feature 解耦，同时保留 XGBoost 的二阶梯度优化框架。通过引入恒为 1 的 bias predictive feature，current_metagb 可以自然退化为普通常数叶子模型，因此是对 XGBoost 叶子模型的严格扩展。该方法可统一应用于回归、二分类和多分类任务。在多分类中，scalar one-output-per-tree 路径保留了每个类别独立的树结构，因此表达力更强；vector 路径共享类别间的条件树结构，并在叶子中支持每个类别独立选择 predictive feature，是一种更紧凑但结构灵活性较弱的变体。
```
