# TabArena30 Fixed-Backbone MetaGB

This isolated experiment runner targets expensive TabArena binary/multiclass
tails. It avoids rerunning a full Random300 MetaGB search when the XGBoost
backbone has already been selected.

## Protocol

For each `(task_type, preprocessing, dataset, outer_fold)`:

1. Load the selected `xgb_cpu` Random300 candidate from previous summaries or
   task logs.
2. If the XGB selected backbone is missing, run the corresponding `xgb_cpu`
   Random300 task first.
3. Freeze the shared XGB backbone.
4. Search only MetaGB-specific parameters:

```text
pred_feature_fraction = 0.1, 0.2, ..., 1.0
metagb_score_agg = 0, 1, 2
```

This yields exactly 30 `current_metagb` candidates per outer fold.

## Preprocessing

`drop` is strict in this runner:

1. remove rows with missing target;
2. remove rows with any missing feature;
3. drop non-numeric feature columns.

This aligns drop row counts with one-hot row counts for datasets such as
`HR_Analytics_Job_Change_of_Data_Scientists.csv`.

`in_vehicle_coupon_recommendation.csv` is excluded from the default queue because
strict missing-row removal leaves only 108 rows.

## Audit

```bash
python3 tabarena_30/scripts/audit_tabarena_status.py
```

Outputs:

- `tabarena_30/audits/tabarena_missing_drop_rates.csv`
- `tabarena_30/audits/queue_priority.csv`
- `tabarena_30/audits/audit_report.md`

## Dry Run

Existing XGB backbone available:

```bash
python3 tabarena_30/scripts/exp1_tabarena30_all.py \
  --mode quick \
  --task-types multiclass \
  --models current_metagb \
  --categorical-handling onehot \
  --multiclass-datasets anneal.csv \
  --max-candidates 300 \
  --outer-folds 5 --inner-folds 4 \
  --outer-fold-index 0 \
  --task-workers 1 --n-jobs 1 \
  --dry-run-tasks \
  --run-id dryrun_anneal_onehot_metagb30
```

Missing XGB backbone:

```bash
python3 tabarena_30/scripts/exp1_tabarena30_all.py \
  --mode quick \
  --task-types binary \
  --models current_metagb \
  --categorical-handling drop \
  --binary-datasets bank-marketing.csv \
  --max-candidates 300 \
  --outer-folds 5 --inner-folds 4 \
  --outer-fold-index 0 \
  --task-workers 1 --n-jobs 1 \
  --dry-run-tasks \
  --run-id dryrun_bank_marketing_drop_bootstrap_xgb
```

The first command plans one `current_metagb` task with 30 candidates. The second
plans only the missing `xgb_cpu` task first; the MetaGB task is created after the
XGB selected backbone is available.

## Smoke

```bash
python3 tabarena_30/scripts/exp1_tabarena30_all.py \
  --mode quick \
  --task-types binary \
  --models current_metagb \
  --categorical-handling drop \
  --binary-datasets blood-transfusion-service-center.csv \
  --max-candidates 300 \
  --outer-folds 2 --inner-folds 2 \
  --outer-fold-index 0 \
  --task-workers 1 --n-jobs 1 \
  --progress-every 10 \
  --run-id smoke_blood_drop_metagb30
```

Expected smoke evidence:

- `summary.csv`: 1 row with `status=ok`
- `candidates.csv`: 30 rows
- `xgb_backbone_source.csv`: 1 row
- `pred_fraction_analysis.csv`: 30 rows
