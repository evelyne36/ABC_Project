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
