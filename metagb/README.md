# Generalized Gradient Boosting Library

## Review of XGBoost

The Taylor expansion of the loss function at step $t$ is:

$$\sum_{i=1}^n \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t)$$

In most cases, we assume $f_t(x_i)$ is constant for both left and right subtrees. Let $w_j = f_t(x_i)$ for samples in leaf $j$, yielding:

$$
\begin{align}
L &= \sum_{i=1}^n \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t) \\
&= \sum_{j=1}^T \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i\right) w_j^2\right] + \Omega(f_t)
\end{align}
$$

where $I_j$ denotes the set of samples in leaf $j$.

For a fixed tree structure, we can compute the optimal weight $w_j^*$ of leaf $j$ by:

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

In regression with unweighted samples, $h_i = 1$ and $g_i = y_i - \hat{y}_i^{(t-1)}$, which is simply the residual. Thus, the optimal weight for a given leaf is the mean of the residuals.

## Conditional Variables vs. Predictive Variables

We can conceptualize tree node splitting as using two types of variables:

- **Conditional variables**: Used to decide on the split
- **Predictive variables**: Used to evaluate and compute predictions at the leaves

In standard gradient boosting, the predictive variable is simply a bias term (constant) $w_j$ for each leaf $j$.

### Motivation: When Bias Terms Are Insufficient

In some cases, conditional variables have no prediction power individually, making bias-only predictive variables too weak.

Consider this classic XOR-like classification example:

```
y-axis
  ^
  | x o
  | o x
  +-------> x-axis
```

We want to classify `x` from `o`. Neither axis (x or y) alone is informative. The loss will not reduce significantly if we use either x or y as decision variables individually. Standard gradient boosting would produce a two-level tree:

```
      x < 1
      /   \
   y<1     y<1
   / \     / \
  o   x   x   o
```

However, if we could use variables as predictive functions, we would only need a one-level tree:

```
    x < 1
    /   \
   y    1-y
```

This makes the decision boundary much more efficient and expressive!

## Generalized Gradient Boosting

In XGBoost, $f_t(x_i)$ is a constant bias, which provides significant speedup in the tree learning algorithm. We propose generalizing this so that:

$$f_t(x_i) = \beta \cdot x_{ij}$$

where $x_{ij}$ is feature $j$ of sample $x_i$. Note that $f_t(x_i)$ can be any function, but we maintain a speed-versus-expressiveness trade-off—more complex functions decrease speed and increase overfitting risk.

### Modified Objective

With this generalization, the objective becomes:

$$
\begin{align}
L &= \sum_{i=1}^n \left[g_i \beta x_{ij} + \frac{1}{2}h_i \beta^2 x_{ij}^2\right] + \Omega(f_t) \\
&= \sum_{j=1}^T \left[\left(\sum_{i \in I_j} g_i x_{ij}\right) \beta_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i x_{ij}^2\right) \beta_j^2\right] + \Omega(f_t)
\end{align}
$$

For a fixed tree structure, we can compute the optimal coefficient $\beta^*$ for leaf $j$ as:

$$
\beta_j^* = -\frac{\sum_{i \in I_j} g_i x_{ij}}{\sum_{i \in I_j} h_i x_{ij}^2 + \lambda}
$$

### Backward Compatibility

For every dataset, we can add a bias feature (constant 1) and use it as a predictive variable candidate at every split. When the bias is selected, the algorithm reduces to the original XGBoost formulation, ensuring backward compatibility.

## Key Advantages

- **Increased expressiveness**: Linear predictive functions can capture patterns that constant predictions cannot
- **Efficiency**: Maintains computational efficiency comparable to standard gradient boosting
- **Flexibility**: Naturally handles cases where individual features have weak marginal effects but strong interaction effects
- **Backward compatible**: Subsumes standard gradient boosting as a special case


## MetaGB API

### Version Selection

- **version=0**: Standard gradient boosting (equivalent to XGBoost). Produces similar results when using the same parameters.
- **version=1**: Deprecated, can be ignored.
- **version=2**: Core generalized gradient boosting algorithm. **This is the recommended version for experimentation.**

### Core Algorithm

The algorithm separates features into two distinct roles:

- **Conditional variables** ($c_1, c_2, \ldots, c_n$): Used to determine tree splits
- **Predictive variables** ($p_1, p_2, \ldots, p_m$): Used to compute leaf predictions

For each tree split, the algorithm:

1. Enumerates all combinations of conditional variable $c_i$ and predictive variable $p_j$
2. For each split point $s_k$ of the conditional variable $c_i$, computes the split gain:
   $$\text{Gain} = L_{\text{left}} + L_{\text{right}} - L_{\text{overall}}$$
   where the leaf prediction is $\beta \cdot p_j$ instead of a constant bias (see [Modified Objective](#modified-objective) above)
3. Selects the optimal combination $(c_i^*, p_j^*, s_k^*)$ that minimizes the gain
4. Splits the tree using $c_i^*$ at threshold $s_k^*$, with leaves predicting using $p_j^*$

### Backward Compatibility

When the predictive variable set contains only the **bias term** (a constant feature of 1), the algorithm reduces to standard gradient boosting where each leaf predicts a constant value $w_j$. This is automatically handled by appending a bias feature to the input data.


### Parameters

#### 1. Prior Conditioning Features

Use these when you have prior knowledge that certain features should be used as conditional variables. For example, if you know that "weekday" is not directly predictive of stock returns but is a powerful conditioning feature (i.e., the relationship between other features and returns differs by weekday), you can set it as a prior conditioning feature.

| Parameter | Description |
|-----------|-------------|
| `prior_cond_feature_idx` | Array of feature indices, e.g., `[0, 1, 2]` |
| `prior_cond_feature_fraction` | Array of sampling probabilities (same length), e.g., `[0.5, 0.6, 0.9]`. Specifies the probability that each feature will be sampled as a conditioning feature per tree. |
| `num_prior_cond_features` | Length of the feature index array |

#### 2. Depth Separation Parameters

These parameters control which features are used at different tree depths, enabling a **hierarchical feature selection** strategy.

##### `depth_sep` (Input Parameter)

A user-specified depth threshold that divides the tree into **shallow** (`depth < depth_sep`) and **deep** (`depth >= depth_sep`) regions.

##### `cond_depth_sep` (Computed Internally)

An index into the `cond_feature_idx[]` array that separates **prior conditioning features** from **other conditioning features**.

```
cond_feature_idx = [prior_0, prior_1, ..., other_0, other_1, ...]
                   |<--- 0 to cond_depth_sep --->|<--- cond_depth_sep to end --->|
                         (prior features)               (other features)
```

- If `cond_shallow_use_prior_only=1`: `cond_depth_sep = num_prior_cond_features_sampled`
- Otherwise: `cond_depth_sep = num_frac_features_cond` (all features)

##### `pred_depth_sep` (Computed Internally)

An index into the `pred_feature_idx[]` array that controls which predictive features are used at different depths.

```
pred_feature_idx = [feat_0, feat_1, ..., feat_n, bias]
                   |<--- 0 to pred_depth_sep --->|<--- pred_depth_sep to end --->|
                        (deep regions only)            (shallow regions)
```

- If `pred_deep_use_prior_only=1`: `pred_depth_sep = num_frac_features_pred - 1` (excludes bias at shallow)
- Otherwise: `pred_depth_sep = 0` (all features everywhere)

##### How Depth Separation Works

| Tree Region | Conditioning Features | Predictive Features |
|-------------|----------------------|---------------------|
| Shallow (`depth < depth_sep`) | `cond_feature_idx[0..cond_depth_sep]` (prior only) | `pred_feature_idx[0..pred_depth_sep]` |
| Deep (`depth >= depth_sep`) | `cond_feature_idx[0..end]` or `[cond_depth_sep..end]` | `pred_feature_idx[pred_depth_sep..end]` |

##### Visual Example

```
Tree with depth_sep=2, cond_shallow_use_prior_only=1:

Depth 0:  [Split using PRIOR cond features only]
           /                              \
Depth 1:  [PRIOR cond features]           [PRIOR cond features]
          /        \                      /        \
Depth 2: [ALL cond features]  [ALL]     [ALL]     [ALL cond features]
```

#### 3. Feature Sharing Parameter

##### `cond_pred_share`

Controls whether predictive features are sampled independently or derived from conditioning features.

| Value | Behavior |
|-------|----------|
| `0` | **Independent sampling**: Conditioning and predictive features are sampled separately based on `cond_feature_fraction` and `pred_feature_fraction` |
| `1` | **Shared features**: Predictive features = conditioning features (excluding priors) + bias |

**When `cond_pred_share=0` (Independent):**
```
cond_feature_idx = [0, 1, 3, 5]      (sampled with cond_feature_fraction)
pred_feature_idx = [2, 4, 6, bias]   (sampled with pred_feature_fraction)
```

**When `cond_pred_share=1` (Shared):**
```
cond_feature_idx = [prior_0, prior_1, 3, 5]
pred_feature_idx = [3, 5, bias]      (derived from cond features, excluding priors)
```

#### Summary Table

| Parameter | Type | Description |
|-----------|------|-------------|
| `depth_sep` | Input | Tree depth threshold for switching feature sets |
| `cond_depth_sep` | Computed | Index separating prior vs other conditioning features |
| `pred_depth_sep` | Computed | Index separating shallow vs deep predictive features |
| `cond_pred_share` | Input | `0`: independent sampling, `1`: pred derived from cond |
| `cond_shallow_use_prior_only` | Input | `0`: use all, `1`: only prior at shallow, `2`: exclusive (prior at shallow, others at deep) |
| `pred_deep_use_prior_only` | Input | `0`: use all pred features, `1`: restrict features by depth |

#### 4. Score Aggregation Parameter

##### `score_agg`

Controls **how to aggregate scores across predictive features and bins** when evaluating potential splits.

For each potential split, the algorithm computes a 3D score array:

```
score_all[ci][pi][bin] = split gain for:
  - conditioning feature ci
  - predictive feature pi
  - split threshold at bin
```

The `score_agg` parameter determines how to find the best `(ci, pi, bin)` combination:

##### Deterministic Methods (Optimized Bin Selection)

| Value | Method | Description |
|-------|--------|-------------|
| `0` | **Max across all** | Find the single best `(ci, pi, bin)` with maximum score. Most greedy. |
| `1` | **Average across pred, max across bin** | For each `(ci, bin)`, average scores across all pred features, then find best bin. Reduces overfitting to a single pred feature. |
| `2` | **Median across pred, max across bin** | For each `(ci, bin)`, take median score across pred features, then find best bin. More robust to outliers. |
| `3` | **Random pred, max across bin** | For each `(ci, bin)`, randomly select one pred feature's score, then find best bin. Adds diversity. |

##### Randomized Methods (Random Bin Selection)

| Value | Method | Description |
|-------|--------|-------------|
| `10` | **Random bin** | For each `(ci, pi)`, randomly select a bin. Extreme randomization. |
| `11` | **Average across pred, random bin** | Average across pred features, randomly select bin. |
| `12` | **Median across pred, random bin** | Median across pred features, randomly select bin. |
| `13` | **Random pred, random bin** | Randomly select both pred feature and bin. Maximum randomization. |

##### Visual Illustration

```
score_all array for a single conditioning feature ci:

           bin_0  bin_1  bin_2  ...  bin_k
pred_0  [  s00    s01    s02   ...   s0k  ]
pred_1  [  s10    s11    s12   ...   s1k  ]
pred_2  [  s20    s21    s22   ...   s2k  ]
  ...
pred_m  [  sm0    sm1    sm2   ...   smk  ]

score_agg=0:  max(all values)           → best (pi*, bin*)
score_agg=1:  avg(columns) → max(row)   → best bin*
score_agg=2:  median(columns) → max(row)→ best bin*
score_agg=3:  random row → max(row)     → best bin*

score_agg=10-13: Same logic but random bin instead of max
```

##### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Max (0)** | Finds globally optimal split | May overfit to specific pred feature |
| **Average (1)** | More robust, considers all pred features | May miss strong individual signals |
| **Median (2)** | Robust to outlier pred features | Similar to average |
| **Random (3, 10-13)** | Adds diversity, reduces overfitting | May miss optimal splits |

##### Recommendations

- **`score_agg=0`** (default): Best for most cases, finds optimal split
- **`score_agg=1` or `2`**: Use when you want splits that generalize well across multiple predictive features
- **`score_agg=10-13`**: Use for extra randomization, similar to "extremely randomized trees"
