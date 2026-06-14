# MetaGB Synthetic Feature Validation

## 1. Design Understanding

MetaGB replaces each constant XGBoost leaf response with a selected
one-feature linear response:

```text
ordinary XGBoost leaf:  w
MetaGB leaf:            beta * selected_feature
```

The bias feature is always available, so MetaGB can still choose a constant
leaf. Its distinctive modeling assumption is that features may have two
different roles:

- conditional features partition the sample space;
- predictive features explain the response inside a partition.

The most plausible advantage is therefore **representation efficiency** on
functions that are piecewise one-feature-linear. Ordinary trees must
approximate the within-region slope with many constant leaves, while MetaGB
can use a small number of conditional splits and retain a slope in each leaf.

## 2. Advantage Scenarios

### Directly Motivated by the Original Design

- `xor_classification`: the original README example. One feature defines the
  region and the other becomes a signed leaf-level predictor.
- `regime_slope`: the effect of a predictive feature changes sign across a
  conditional split.
- `global_linear`: the simplest case where a feature-scaled leaf is a more
  direct representation than a staircase of constant leaves.

### Additional Scenarios Derived From the Principle

- `feature_switch`: a gate selects which of two features predicts the target.
- `multiplicative`: splitting one factor turns the other factor into a useful
  local linear predictor.
- `regime_logit`: the binary-class logit effect of a predictive feature changes
  by regime.

### Boundary / Control Scenarios

- `pure_step_control`: piecewise constant truth already matches XGBoost leaves.
- `curved_control`: a smooth quadratic is not directly represented by one
  feature-scaled leaf.

## 3. Experimental Protocol

The benchmark follows the important `random300` conventions:

- paired XGB/MetaGB hyperparameter backbones;
- nested cross-validation;
- inner-fold candidate selection and outer-fold test reporting;
- regression primary metric = RMSE, binary primary metric = AUC;
- `current_metagb_pred0` ablation to isolate the effect of predictive features;
- CSV and JSON artifacts for candidate-, fold-, aggregate-, and comparison-level
  results.

The full run used 3,000 samples per scenario, 5 outer folds, 3 inner folds, and
5 paired capacity candidates ranging from depth 1 / 30 trees to depth 5 / 240
trees.

## 4. Full Results

Positive advantage means full MetaGB is better.

| Scenario | Metric | XGB | MetaGB pred0 | MetaGB | Advantage vs XGB | Advantage vs pred0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| regime_slope | RMSE | 0.4267 | 0.4212 | **0.3682** | **+0.0585** | **+0.0529** |
| multiplicative | RMSE | 0.2769 | 0.2698 | **0.2013** | **+0.0756** | **+0.0685** |
| regime_logit | AUC | 0.9045 | 0.9055 | **0.9159** | **+0.0114** | **+0.0104** |
| feature_switch | RMSE | 0.2481 | 0.2555 | **0.2435** | **+0.0046** | **+0.0120** |
| global_linear | RMSE | 0.1602 | 0.1596 | **0.1541** | **+0.0061** | **+0.0055** |
| xor_classification | AUC | 0.9977 | 0.9970 | **0.9981** | **+0.0003** | **+0.0010** |
| pure_step_control | RMSE | 0.2344 | **0.2278** | 0.2288 | +0.0056 | -0.0010 |
| curved_control | RMSE | **0.1587** | 0.1601 | 0.1623 | -0.0036 | -0.0022 |

Source: `results/full_all_20260614/comparison.csv`.

## 5. Interpretation

The strongest evidence supports the central MetaGB hypothesis:

1. **Regime-dependent slopes are the clearest advantage.** `regime_slope`,
   `multiplicative`, and `regime_logit` remain better than both XGB and pred0
   under the full protocol.
2. **The predictive-feature mechanism matters.** The strongest scenarios also
   beat `current_metagb_pred0`, so their gains cannot be explained only by
   differences between the MetaGB and XGB split updaters.
3. **XOR is primarily a capacity-efficiency example.** In the smoke run,
   MetaGB AUC was 0.9930 versus XGB 0.7309. In the full run, deeper and larger
   XGB models caught up to 0.9977, leaving only a small MetaGB advantage.
4. **MetaGB is not uniformly better.** XGB wins the curved control, while the
   step control is effectively tied. This is consistent with ordinary
   constant-leaf trees already matching those structures well.
5. **The advantage shrinks with model capacity.** Quick runs show large gaps;
   full runs show smaller gaps because sufficiently deep boosted trees can
   approximate piecewise-linear behavior with many constant regions.

## 6. Reproduction

```bash
python3.13 synthetic_metagb/run_benchmark.py --mode smoke
python3.13 synthetic_metagb/run_benchmark.py --mode quick
python3.13 synthetic_metagb/run_benchmark.py --mode full
python3.13 -m pytest synthetic_metagb/test_scenarios.py -q
```

Primary artifacts:

- `results/full_all_20260614/comparison.csv`
- `results/full_all_20260614/aggregate.csv`
- `results/full_all_20260614/folds.csv`
- `results/full_all_20260614/candidates.csv`
- `results/full_all_20260614/metadata.json`
