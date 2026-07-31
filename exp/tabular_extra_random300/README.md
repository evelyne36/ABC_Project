# Extra-tabular Random300 CrossGB benchmark

## Purpose

This suite evaluates CrossGB and CPU XGBoost on the staged extra tabular
datasets in `exp/tabular_extra_random300/data`.

## Paper correspondence

This directory supplies the additional real-tabular CrossGB--XGBoost panels.
Appendix Sections B.1--B.2 define the paired nested-CV and shared
300-candidate protocol. Appendix Table 14 contains the additional regression
panels, and Appendix Table 15 Panels A--D contain the additional binary and
multiclass panels.

## Environment

```bash
python3 -m pip install -r requirements.txt
bash CrossGB/scripts/build_crossgb.sh
bash CrossGB/scripts/verify_crossgb.sh
```

## Data and protocol

- Datasets are downloaded from fixed OpenML data IDs by the repository
  downloader:

```bash
python3 scripts/download_datasets.py download --group openml
python3 scripts/download_datasets.py verify --group openml
```

- Files are placed in `exp/tabular_extra_random300/data`; fixed OpenML IDs,
  target columns, and filenames are recorded in
  `data_registry/openml_source_manifest.json`.
- Full protocol uses 5 outer folds, 4 inner folds, and 300 candidates.
- CrossGB and XGBoost share the same sampled backbones; regression, binary, and
  multiclass tasks use the same preprocessing condition per run.
- The retained launch scripts default to `onehot`, matching the submitted
  preprocessing condition. Set `HANDLING` explicitly for a different panel.

Run all commands from the repository root.

## Smoke versus full execution

The smoke launcher uses a small row/candidate/fold budget to validate the
pipeline. It does not reproduce a paper cell. The full launchers use 5x4
nested CV and 300 paired candidates; select exactly one preprocessing condition
per run.

### Smoke run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
bash exp/tabular_extra_random300/smoke_linux.sh
```

### Full run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
bash exp/tabular_extra_random300/run_linux_full.sh
```

`run_linux_full.sh` is the retained representative 3-regression + 3-binary +
3-multiclass one-hot launcher. Use `run_linux_comprehensive.sh` for the larger
staged collection; it also defaults to `onehot`, while explicit
`HANDLING=drop` or `HANDLING=ordinal` selects a different condition.

## Outputs and restart behavior

Run artifacts are written to
`exp/tabular_extra_random300/results/<run-id>/`, with task logs in
`exp/tabular_extra_random300/logs/<run-id>/` or
`exp/tabular_extra_random300/nohup_logs/`. Inspect
`summary.csv`, `candidates.csv`, `metadata.json`, `parameter_audit.json`, and
`candidate_configs.json`. Partial training is not shard-resumable; use a new
run ID after interruption. Appendix Tables 14--15 use paired five-fold cells
whose preprocessing, candidates, and provenance audits agree.

## Limitations

The benchmark runner does not download data itself; use the root downloader,
which creates the staged directory. OpenML availability can change, row caps
are smoke-only, and categorical-handling choices must remain fixed when
comparing models. The repository registry pins numeric OpenML data IDs instead
of resolving mutable dataset names.
