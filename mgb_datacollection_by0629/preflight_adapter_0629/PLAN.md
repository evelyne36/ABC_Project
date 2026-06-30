# 0629 Preflight Adapter Plan

## Goal

Check whether a cheap, interpretable preflight score can separate the datasets
where MGB already wins from datasets where XGB remains stronger.

## Workflow

1. Build a dataset-level table from 0629 paired outcomes.
2. Resolve each dataset to either TabArena CSV, tabular-extra CSV, or synthetic
   generator output.
3. Compute static and shallow-tree mechanism features before using outcome
   labels.
4. Compute both the raw mechanism score and a family-calibrated adapter score.
5. Audit the calibrated score against existing wins/losses by data group and
   write a Chinese report with advantage shares.

## Outputs

- `outputs/preflight_0629_dataset_table.csv`
- `outputs/preflight_0629_scores.csv`
- `outputs/preflight_0629_audit.csv`
- `outputs/preflight_0629_group_band_audit.csv`
- `outputs/preflight_0629_group_threshold_audit.csv`
- `outputs/preflight_0629_report_zh.md`
- `outputs/positive_negative_case_audit.md`
