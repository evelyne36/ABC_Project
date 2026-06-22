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

Rigorous protocols:

```bash
python3.13 synthetic_metagb/run_benchmark.py --protocol smoke
python3.13 synthetic_metagb/run_benchmark.py --protocol random300_confirmatory
python3.13 synthetic_metagb/run_benchmark.py --protocol random600_confirmatory
```

The rigorous runner uses the actual Random300/Random600 candidate generator,
resumable task shards, audited shared nested-CV splits, and independent data
seeds as the statistical pairing unit.

See `SMOKE_REPORT.md` for the completed framework validation and
`LINUX_RUN_GUIDE.md` for formal server commands and output checks.
`SYNTHETIC_DATA_AND_RUNTIME_ZH.md` provides the Chinese synthetic-data design,
exact computation counts, and 15-task parallel runtime estimate.
