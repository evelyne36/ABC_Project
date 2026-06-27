# Theory Support and Mechanism-Aligned Preflight Design

## 0. Current Design Position

The primary adapter is **not** trained from final "MetaGB wins/loses" labels.

The main score is an intrinsic mechanism-alignment score:

```text
phi(D) -> mechanism_match_score(D)
```

where `phi(D)` contains dataset-level proxies for MetaGB's own modeling
mechanism:

- one-feature local linear leaf advantage;
- conditional/predictive feature decoupling;
- local response heterogeneity;
- sparse predictive-feature concentration;
- sufficient local signal support.

Completed benchmark outcomes are used only for retrospective audit. They should
not define or fit the score in the first version.

## 1. Why A Preflight Signal Should Exist

For a leaf or local region `R`, ordinary XGBoost chooses a constant response:

```text
w* = - sum_i g_i / (sum_i h_i + lambda)
score_const(R) = (sum_i g_i)^2 / (sum_i h_i + lambda)
```

MetaGB instead allows a one-feature local linear response:

```text
f_R(x) = beta_j * x_j
beta_j* = - sum_i g_i x_ij / (sum_i h_i x_ij^2 + lambda_meta)
score_j(R) = (sum_i g_i x_ij)^2 / (sum_i h_i x_ij^2 + lambda_meta)
```

So the local advantage of MetaGB over a constant leaf is tied to whether any feature has strong projected gradient alignment inside the local region:

```text
max_j score_j(R) - score_const(R)
```

This gives a theoretical basis for preflight analysis:

- If residual/gradient mass inside local regions is mostly explained by a constant offset, XGBoost should be sufficient.
- If residual/gradient mass has strong one-feature alignment inside local regions, MetaGB has a plausible benefit.
- If the feature that defines a region differs from the feature that predicts residuals in that region, MetaGB's conditional/predictive decoupling is especially relevant.

The current shallow-tree local-linearity probe is a cheap proxy for this quantity. It does not exactly compute MetaGB's second-order objective, but it approximates the same question:

```text
After a cheap conditional partition, is there stronger local one-feature predictive signal than global one-feature signal?
```

## 2. From Heuristic Score To Tunable Mechanism Policy

The preflight module should first become a tunable mechanism policy:

```text
phi(D, B_pilot) -> mechanism_match_score -> decision
```

where:

- `D` is the dataset after a candidate preprocessing mode;
- `B_pilot` is optional cheap pilot evidence from shallow trees, tiny XGB runs, or tiny MetaGB probes;
- `decision` is one of `WEAK_MECHANISM_MATCH`, `MODERATE_MECHANISM_MATCH`,
  `STRONG_MECHANISM_MATCH`;
- optionally, the module also predicts a recommended `pred_feature_fraction` range.

### Tunable Mechanism Score

The first tunable version is deliberately transparent:

```text
score = w^T phi
```

Candidate feature groups:

1. Static dataset meta-features:
   - rows, features, feature/row ratio;
   - missingness;
   - categorical fraction;
   - one-hot expansion ratio;
   - sparsity;
   - target/class imbalance.

2. Theoretical mechanism proxies:
   - local projected-gradient or residual-feature alignment;
   - local/global alignment lift;
   - conditional/predictive decoupling fraction;
   - top-k predictive concentration;
   - feature scale dispersion.

3. Cheap pilot evidence:
   - shallow conditional partition;
   - residual structure after shallow XGB;
   - projected-gradient alignment by feature;
   - top-k predictive-feature concentration.

Tunable outputs:

- recommended `pred_feature_fraction`;
- recommended `score_agg`;
- optional predictive-feature whitelist size.

## 3. Hyperparameters To Tune

The preflight module itself should expose tunable hyperparameters:

```text
pilot_tree_depth
pilot_min_samples_leaf
pilot_sample_cap
top_k_predictive_features
try_light_candidate_budget
try_full_candidate_budget
pred_fraction_grid
weight_linear_leaf_advantage
weight_cond_pred_decoupling
weight_local_heterogeneity
weight_predictive_concentration
weight_signal_support
strong_match_threshold
moderate_match_threshold
```

These hyperparameters should be selected from theory and mechanism-recovery
experiments, especially synthetic datasets with known conditional and predictive
features. They should not be optimized on final benchmark wins in the same
datasets used for claims.

## 4. Generalization Protocol

To avoid overfitting the preflight scorer to final outcomes:

1. Split by dataset, not by fold or candidate.
2. Fit/tune thresholds using mechanism-labeled synthetic tasks or held-out
   historical datasets only.
3. For a held-out dataset, allow only cheap preflight probes on its training split.
4. Evaluate whether high mechanism scores correspond to interpretable MetaGB
   behavior, such as selected predictive-feature hits or gains in mechanism
   families.

Report:

- retained-positive rate;
- rejected-loss rate;
- selected predictive-feature hit rate on synthetic tasks;
- score distribution by mechanism family;
- retrospective gain audit, clearly marked as audit.

## 5. Research Framing

This turns the adapter into a mechanism-grounded contribution rather than a
hand-written benchmark predictor:

```text
MetaGB Preflight computes an intrinsic mechanism-alignment score for when a
dataset appears compatible with feature-selected local linear leaves and
conditional/predictive feature decoupling. The score is grounded in MetaGB's
second-order leaf score and estimates whether local projected-gradient alignment
is present before allocating a full MetaGB experiment.
```

In paper terms, this supports a stronger claim:

- not "MetaGB always beats XGBoost";
- but "MetaGB includes a mechanism-aware applicability adapter that identifies
  datasets whose structure matches its extra local linear leaf capacity."

## 6. Implementation Roadmap

1. Extend the current table builder with exact residual-gradient alignment
   proxies from a tiny XGB pilot.
2. Keep `score_mechanism_alignment.py` as the primary preflight scorer.
3. Add synthetic mechanism calibration with known true conditional/predictive
   features and feature-hit metrics.
4. Add `pred_budget_policy.py` to recommend `pred_feature_fraction` or
   predictive-feature whitelist size from the mechanism components.
5. Keep supervised `train_meta_policy.py` as an optional retrospective audit
   until enough completed datasets exist.
