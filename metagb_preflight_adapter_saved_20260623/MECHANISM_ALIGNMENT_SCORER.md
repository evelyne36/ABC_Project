# Mechanism-Alignment Preflight Scorer

## Purpose

This scorer does **not** learn from final MetaGB-vs-XGBoost wins and losses.

It asks a more structural question:

```text
How well does this dataset match the modeling mechanism that makes MetaGB different?
```

Full experiment labels are allowed only for retrospective audit. They must not
enter the score.

## MetaGB Mechanisms Being Scored

### 1. One-feature Local Linear Leaf Advantage

For a local region `R`, XGBoost uses a constant leaf update:

```text
score_const(R) = (sum_i g_i)^2 / (sum_i h_i + lambda)
```

MetaGB can use a feature-scaled leaf:

```text
score_j(R) = (sum_i g_i x_ij)^2 / (sum_i h_i x_ij^2 + lambda_meta)
```

The scorer estimates whether local projected-gradient alignment exists:

```text
max_j score_j(R) / score_const(R)
```

This is a direct proxy for whether `beta * x_j` is structurally useful.

### 2. Conditional/Predictive Decoupling

MetaGB's key design separates:

```text
conditional feature: routes samples into regions
predictive feature: forms the local leaf response
```

The scorer uses shallow conditional partitions and checks whether the strongest
local predictive feature is frequently outside the split-feature set.

### 3. Local Heterogeneity

If global feature-target association is weak but local association becomes
strong after a cheap partition, the dataset better matches MetaGB's local
response idea.

### 4. Sparse Predictive Concentration

MetaGB selects one predictive feature per leaf. This is better matched when the
local predictive signal is concentrated in a small number of features rather
than diffusely spread across many weak features.

### 5. Signal Support

A dataset with no meaningful local one-feature association should not receive a
high mechanism score even if other ratios are unstable.

## Score Definition

The default intrinsic score is a weighted sum:

```text
score =
  w_linear      * local_linear_leaf_advantage
+ w_decoupling  * conditional_predictive_decoupling
+ w_hetero      * local_heterogeneity
+ w_sparse      * predictive_sparse_concentration
+ w_support     * local_signal_support
```

Default weights are theory choices, not supervised fitted parameters.

## Tunable But Not Outcome-Trained

The module exposes weights and thresholds as hyperparameters:

```text
--weight-linear
--weight-decoupling
--weight-heterogeneity
--weight-concentration
--weight-support
--strong-threshold
--moderate-threshold
```

These should be tuned by mechanism goals or synthetic mechanism recovery, not by
optimizing final benchmark wins on the same datasets.

## Recommended Academic Claim

Use this scorer as:

```text
an intrinsic applicability score for MetaGB's conditional-predictive local
linear leaf mechanism
```

not as:

```text
a supervised predictor of benchmark wins
```

Benchmark results can audit whether the mechanism score aligns with observed
gains, but the score itself remains grounded in the method's structural premise.
