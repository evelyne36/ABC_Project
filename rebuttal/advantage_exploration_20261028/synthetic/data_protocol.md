# CGB advantage exploration synthetic data v1 (2026-10-28)

This is a new, outcome-independent exploratory source, `cgb-advantage-synthetic-v1-20261028`. Eight settings and two fresh development data seeds (110001, 110002) are declared before generation. No models, scores, downloads, or target-driven case selection are used here. The old S1 setting (n=240,d=32) is not rerun as a new case. All eight settings, including controls and unsuccessful results, must remain visible in a downstream report; two DEV draws establish exploration, not confirmatory evidence of superiority.

| Case | n | d | Signal f(X) | Information sought |
|---|---:|---:|---|---|
| A01 | 120 | 32 | 2 sign(x0) x1 | Half-sized S1; sparse local slopes with less data |
| A02 | 240 | 8 | 2 sign(x0) x1 | Fewer irrelevant coordinates |
| A03 | 240 | 64 | 2 sign(x0) x1 | More irrelevant coordinates |
| A04 | 240 | 32 | 2 (I(x0>=Phi^-1(.8))-.2)/sqrt(.2*.8) x1 | Unequal region sample sizes; centered, unequal slopes |
| A05 | 240 | 32 | -2 x1 when x0<0; +2 x2 otherwise | One gate switches between independent predictors |
| A06 | 240 | 32 | sqrt(2)(sign(x0)x2+sign(x1)x3) | Two additive sparse gate modules |
| A07 | 120 | 32 | 2 sign(x0) sum(x1..x31)/sqrt(31) | Dense local slope control, paired with A01 |
| A08 | 120 | 32 | 2 sign(x0) | Region-intercept control, paired with A01 |

All covariates are independent N(0,1); epsilon is independent N(0,.3²); y=f(X)+epsilon. Every signal has population mean zero and E[f²]=4, hence population L2 norm 2. Scaling is analytic and fixed before drawing data; there is no empirical target rescaling. Continuous input ranges have the same distribution for every method and case. Across sparse, dense, and intercept controls, equal signal norm controls aggregate amplitude rather than maximum slope or pointwise magnitude. Dense sums are divided by sqrt(d-1); two independent modules are divided by sqrt(2) relative to single-module slope 2.

For A04 the gate's right-region population probability is p=.2, with threshold Phi^-1(.8)=0.8416212335729143. Its standardized gate takes values -.5 and 2, giving local slopes -1 and +4. Its population mean is zero and second moment is one. Thresholds and slopes never use sampled targets, empirical region counts, or quantile fitting. In slope settings E[f|gate]=0 because predictive coordinates are independent, centered, and absent from routing coordinates. The gate alone has no population mean effect. A05 deliberately differs: the prediction coordinate changes with region, so it is not simply another name for S1. A06 has two gates; each module uses one independent prediction coordinate. A08 intentionally supplies a mean effect as a control.

For each seed, independently seeded streams produce a 480x64 master covariate array and a 480-vector noise array. A setting selects the first n rows and d columns; this yields common covariates/noise for equal shapes and nested inputs for dimension/sample-size comparisons. A01/A07/A08 exactly share X, epsilon, and split indices. A04/A05/A06 also share X, epsilon, and splits; their signals differ. Across data seeds the entire base draw is independent. Dimensions are full observed covariates for every baseline; no oracle is appended or substituted. Test membership is fixed by the independent seeded 60/20/20 row permutation reused from `fastlane_20261027/prepare_data.py`; it is not stratified or adjusted using outcomes.

The generator imports and reuses that source's `save`, `random_split`, and SHA-256 array/NPZ conventions. A new protocol version is recorded, together with generator and imported source hashes. NPZ includes X (float32), y (float64), int64 train/val/test, task_type, y_train/y_val/y_test, and separate diagnostic oracle_signal/oracle_region/noise. X is rounded to float32 only on save, following the existing protocol; signals are calculated from the underlying float64 draw. Oracle arrays and JSON support descriptors must not be passed as model inputs. Search workers may read only training and validation labels; test labels and diagnostics must not influence HPO. All baselines receive identical complete X and identical split membership; no additional normalization is pre-applied.

Confirmation seeds **111001–111005 are registered but not generated**. The generator exposes no confirmation mode. Generate and evaluate these only after freezing HPO, the exact selected settings, model configurations, and comparison/reporting rule. Preserve exploratory selection provenance. Both DEV and future confirmation seeds were scanned against historical `rebuttal/` and `local_sync/` text integer tokens and filenames before generation; decimal score substrings are excluded. The current exploration directory is excluded from that historical audit to permit its own registration.

Reproduce in a clean output directory with the existing NumPy+pandas runtime:

```sh
python rebuttal/advantage_exploration_20261028/synthetic/prepare_synthetic.py --out /tmp/cgb-advantage-synthetic-v1
python rebuttal/advantage_exploration_20261028/synthetic/verify_synthetic.py
```

Do not overwrite generated data. Verification checks hashes, deterministic arrays, signal formulas, split coverage, explicit pairing, full input shapes, and absence of confirmation files. It does not fit models or summarize held-out labels.
