# MetaGB Synthetic Validation

This benchmark tests the structural hypothesis behind MetaGB:

```text
conditional features define regions
predictive features provide one-feature linear responses inside leaves
```

It follows the important parts of `random300`: paired hyperparameter backbones,
nested cross-validation, validation-based candidate selection, outer-fold test
metrics, and CSV/JSON artifacts.

Models:

- `xgb_cpu`: ordinary constant-leaf XGBoost
- `current_metagb_pred0`: MetaGB updater with only the bias predictive feature
- `current_metagb`: MetaGB with all features available as predictive features

Scenarios:

- Expected MetaGB advantages: `global_linear`, `regime_slope`,
  `feature_switch`, `multiplicative`, `xor_classification`, `regime_logit`
- Negative controls: `pure_step_control`, `curved_control`

Run:

```bash
python3.13 synthetic_metagb/run_benchmark.py --mode quick
```

The key output is `comparison.csv`. A positive
`metagb_signed_advantage_vs_xgb` always means MetaGB beat XGBoost on the
scenario's primary metric. `metagb_signed_advantage_vs_pred0` isolates the
benefit of non-bias predictive features from other MetaGB updater differences.
