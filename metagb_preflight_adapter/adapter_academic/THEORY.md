# Theory: Mechanism-Aware Preflight For MetaGB

## Setup

For a local region `R`, standard XGBoost uses a constant leaf response:

```text
w* = - sum_{i in R} g_i / (sum_{i in R} h_i + lambda)
score_const(R) = (sum_i g_i)^2 / (sum_i h_i + lambda)
```

MetaGB replaces the constant response with a feature-selected one-dimensional
local linear response:

```text
f_R(x) = beta_j x_j
beta_j* = - sum_i g_i x_ij / (sum_i h_i x_ij^2 + lambda_meta)
score_j(R) = (sum_i g_i x_ij)^2 / (sum_i h_i x_ij^2 + lambda_meta)
```

## Proposition

If, inside a local region `R`, the gradient residual has non-trivial projected
alignment with some predictive feature `x_j`, then a feature-selected local
linear leaf has strictly larger second-order score than a constant-only update
after the local constant component has been removed.

Equivalently, if:

```text
max_j (sum_i r_i z_ij)^2 / (sum_i z_ij^2 + lambda_meta) > 0
```

where:

```text
r_i = residual/gradient after removing the local constant component
z_ij = standardized predictive feature value inside R
```

then MetaGB's local linear basis contains a direction not available to a pure
constant leaf in the same region.

## Proof Sketch

After the local constant component is removed, a constant leaf has zero remaining
projection on the centered residual. A MetaGB leaf can still choose a basis
function `z_j`. The second-order quadratic objective restricted to this basis has
closed-form optimum:

```text
beta_j* = - <g, z_j> / (<h z_j, z_j> + lambda_meta)
```

and objective decrease proportional to:

```text
<g, z_j>^2 / (<h z_j, z_j> + lambda_meta)
```

Therefore any non-zero projected-gradient alignment gives positive additional
local score. This is exactly the structural signal the preflight score estimates
without using final benchmark outcomes.

## Why This Is Not AutoML Landmarking

AutoML landmarking typically learns from historical model performance. This
adapter instead computes an intrinsic mechanism score from MetaGB's own leaf
objective. Real benchmark outcomes are used only for audit, not for fitting the
score.

## Interpretation Of High-Score Failures

If a dataset scores high but MetaGB does not win, the diagnostic is not
automatically invalidated. The high score means the data contains a mechanism
MetaGB is designed to exploit. Failure to convert the mechanism into benchmark
gain may indicate optimization, binning, regularization, noise, hyperparameter,
or implementation limitations.
