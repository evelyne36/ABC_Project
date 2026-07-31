# Time-series extra Random300 benchmark

## Purpose

This suite compares `crossgb`, CPU XGBoost, LightGBM, and CatBoost after
leakage-safe tabularization of multivariate time series.

## Paper correspondence

This directory reproduces the time-series evaluation in Appendix Section F and
Table 18. Appendix Section B.4 and Table 4 define feature construction,
chronological selection/refit, the ten dataset--horizon tasks, and the recorded
300-candidate search space.

## Environment

```bash
python3 -m pip install -r requirements.txt
bash CrossGB/scripts/build_crossgb.sh
bash CrossGB/scripts/verify_crossgb.sh
```

`requirements-linux.txt` is retained only as a smaller experiment-specific
environment for constrained CPU servers.

## Data and protocol

- Dataset metadata: `configs/dataset_registry.json`; raw files live under
  `data/raw`.
- Splits are chronological (`fixed`) or expanding-window nested CV
  (`nested`); scalers fit only on training history.
- Window, summary, origin-state, and TabPFN-TS feature modes are available.
- Full tree searches use 300 candidates. LightGBM/CatBoost reuse the sampled
  shared backbones while applying model-specific L2 parameter names.

Run all commands from the repository root.

```bash
python3 scripts/download_datasets.py download --group timeseries
python3 scripts/download_datasets.py verify --group timeseries --allow-missing
```

The downloader places files in
`exp/timeseries_extra_random300/data/raw/`. Exact source URLs, target filenames,
and checksum availability are recorded in
`data_registry/timeseries_registry.json`. Entries without a pinned checksum are
listed but intentionally not downloaded automatically.

## Smoke versus paper execution

The smoke launcher uses a reduced dataset/candidate configuration and verifies
chronological splitting and leakage guards. It does not reproduce Table 18.
The paper panel uses sequence length 24, a fixed chronological 60/20/20 split,
the ten registered dataset--horizon tasks, and 300 candidates per model.

### Smoke run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
bash exp/timeseries_extra_random300/smoke_linux.sh
```

### Paper fixed-split runs

Table 18 uses horizon 1 for the four ETT datasets and horizons 1, 6, and 12 for
Exchange rate and ILI. Run the three groups separately so that no unreported
dataset--horizon cross-products are created:

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full --split-protocol fixed --feature-mode window_summary \
  --target-mode sequence --output-scope target \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 \
  --models xgb_cpu lgbm_cpu catboost_cpu crossgb \
  --max-candidates 300 --seeds 42 --seq-lens 24 --pred-lens 1 \
  --task-workers 4 --n-jobs 1 --run-id timeseries_ett_table18
```

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full --split-protocol fixed --feature-mode window_summary \
  --target-mode sequence --output-scope target --datasets exchange_rate \
  --models xgb_cpu lgbm_cpu catboost_cpu crossgb \
  --max-candidates 300 --seeds 42 --seq-lens 24 --pred-lens 1 6 12 \
  --task-workers 4 --n-jobs 1 --run-id timeseries_exchange_table18
```

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full --split-protocol fixed --feature-mode window_summary \
  --target-mode sequence --output-scope target --datasets ILI \
  --models xgb_cpu lgbm_cpu catboost_cpu crossgb \
  --max-candidates 300 --seeds 42 --seq-lens 24 --pred-lens 1 6 12 \
  --task-workers 4 --n-jobs 1 --run-id timeseries_ili_table18
```

### Extended tiered run

```bash
MODELS="crossgb xgb_cpu lgbm_cpu catboost_cpu" \
bash exp/timeseries_extra_random300/run_linux_tiered_tree.sh
```

## Outputs and restart behavior

Each run writes `exp/timeseries_extra_random300/results/<run-id>/`, including
`summary.csv`, `candidates.csv`, `split_manifest.json`, `metadata.json`,
candidate configs, parameter audits, and derived summaries. Task status/logs
are below `exp/timeseries_extra_random300/logs/<run-id>/` or
`exp/timeseries_extra_random300/nohup_logs/`. The benchmark checkpoints
aggregate CSVs but does not safely skip completed tasks; restart interrupted
runs with a new run ID. Table 18 is generated from the fixed-split test rows;
the tiered run is an extended diagnostic and must not replace that panel.

## Limitations

Tabular windows are not sequence models, horizon/feature choices materially
change the task, and optional LightGBM/CatBoost packages must be installed.
Large channel counts can make raw-window searches infeasible; report feature
caps and split protocol with every result.
