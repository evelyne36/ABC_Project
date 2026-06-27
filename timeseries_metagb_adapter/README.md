# Time-Series MetaGB Adapter

This directory adapts the existing `xgb_cpu` vs `current_metagb` benchmark to
time-series forecasting. It follows the LSTF convention used by Informer,
Autoformer, and PatchTST: preserve chronological order, split into future
validation/test blocks, fit scaling from training data only, and build supervised
examples from look-back windows.

## Datasets

The registry intentionally keeps a union of useful future datasets:

- `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2` from ETDataset.
- `exchange_rate`, `electricity`, `traffic`, `solar` from the LSTNet
  multivariate time-series collection.

Default smoke uses `ETTh1` and `exchange_rate`.

## Commands

Download smoke datasets:

```bash
python3 timeseries_metagb_adapter/scripts/download_datasets.py \
  --datasets ETTh1 exchange_rate
```

Run smoke:

```bash
python3 timeseries_metagb_adapter/scripts/run_timeseries_benchmark.py \
  --mode smoke \
  --datasets ETTh1 exchange_rate \
  --models xgb_cpu current_metagb \
  --outer-folds 2 \
  --inner-folds 2 \
  --run-all-outer-folds \
  --max-candidates 30 \
  --seq-len 24 \
  --pred-len 1 \
  --task-workers 1 \
  --n-jobs 1
```

Fast interactive smoke can use the tiny tree search space and a chronological
head slice:

```bash
python3 timeseries_metagb_adapter/scripts/run_timeseries_benchmark.py \
  --mode smoke \
  --datasets ETTh1 exchange_rate \
  --models xgb_cpu current_metagb \
  --outer-folds 2 \
  --inner-folds 2 \
  --run-all-outer-folds \
  --max-candidates 30 \
  --seq-len 24 \
  --pred-len 1 \
  --max-raw-rows 300 \
  --search-space-json timeseries_metagb_adapter/configs/smoke_search_space.json
```

Run the 5x4 protocol:

```bash
python3 timeseries_metagb_adapter/scripts/run_timeseries_benchmark.py \
  --mode full \
  --datasets ETTh1 exchange_rate \
  --models xgb_cpu current_metagb \
  --outer-folds 5 \
  --inner-folds 4 \
  --max-candidates 300 \
  --seq-len 96 \
  --pred-len 1
```

## Outputs

Each run writes to `timeseries_metagb_adapter/results/<run_id>/`:

- `summary.csv`: selected candidate and outer-test metrics.
- `candidates.csv`: candidate validation metrics and selected flag.
- `aggregate.csv`: mean/std by dataset and model.
- `split_manifest.json`: chronological split and scaler audit.
- `parameter_audit.json`: candidate search audit.
- `smoke_report.md`: compact human-readable report.
