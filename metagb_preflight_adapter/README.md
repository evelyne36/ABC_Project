# MetaGB Preflight Adapter

This folder is an isolated workspace for the MetaGB preflight / adapter idea.

It reads existing experiment artifacts and datasets, then writes outputs only under:

```text
metagb_preflight_adapter/outputs/
```

## First-pass Command

```bash
python3 metagb_preflight_adapter/scripts/build_preflight_table.py \
  --result-dir datacollection0621/regression_drop_0621/random600_regression_drop_xgb_vs_current_metagb_regression_20260613_025516/regression_drop_0621results \
  --categorical-handling drop
```

## Trainable Meta-policy

The primary preflight output is an intrinsic mechanism-alignment score. It does
not use final XGB/MetaGB outcome labels:

```bash
python3 metagb_preflight_adapter/scripts/score_mechanism_alignment.py
```

The supervised meta-policy below is optional and should be treated only as an
audit/protocol scaffold until many more completed datasets are available:

```bash
python3 metagb_preflight_adapter/scripts/train_meta_policy.py
```

## Outputs

- `outputs/preflight_dataset_table.csv`: dataset meta-features plus observed XGB/MGB labels.
- `outputs/preflight_recommendations.csv`: compact recommendation view.
- `outputs/preflight_report.md`: short human-readable validation summary.
- `outputs/mechanism_alignment_scores.csv`: intrinsic MetaGB mechanism-match scores.
- `outputs/mechanism_alignment_report.md`: mechanism-alignment report.
- `outputs/meta_policy_cv_predictions.csv`: leave-one-dataset-out predictions.
- `outputs/meta_policy_final_coefficients.csv`: final regularized logistic coefficients.
- `outputs/meta_policy_threshold_sweep.csv`: decision-threshold sensitivity table.
- `outputs/meta_policy_report.md`: trainable policy evaluation report.

## Notes

The first version is deliberately interpretable. It uses static data profiling and a shallow-tree local-linearity probe before trying any new MetaGB run.
