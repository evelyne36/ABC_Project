# MetaGB AAAI-27 Reproducible Project

This directory is the migrated engineering workspace for the AAAI-27 paper. Paper-writing notes and manuscript materials live separately in `../docs_paperwriting/`.

## Frozen scope

- Venue: AAAI-27 Main Technical Track.
- Primary tabular protocol: drop; one-hot is retained as a secondary protocol.
- Synthetic: seed42, 5 outer folds, 4 inner folds, Random300, dim10/dim20 only.
- Time series: paper-facing results use `pred_len=1` only.
- Baselines: Linear, ExtraTrees, RandomForest, LightGBM, CatBoost, TorchMLP.
- RealMLP and TabM are removed from the migrated runner and artifacts.
- CatBoost is not rerun; paper tables must use common complete cells and report N.
- Component ablation is reserved but has no result in this snapshot.

## Layout

```text
vendor/xgboost-metagb/        migrated XGBoost/MetaGB source tree
experiments/                  experiment packages using their original import names
datasets/current_snapshot/    current local TabArena/TabPFN data snapshot
artifacts/raw/                whitelisted original result records
artifacts/canonical/          future parser-generated tables; currently empty
artifacts/provenance/         hashes and migration validation
analysis/scripts/             provenance and future canonical-data scripts
tests/                        migrated protocol/runner tests
environment/                  source environment notes
submission/                   future reproducibility/submission checks
```

`experiments/beforereview/xgboost-master` and `experiments/dataset` are relative symlinks that preserve the original runner path contract while keeping vendor code and datasets in dedicated directories.

## Data rule

Final tables and figures must be generated from `artifacts/raw/` through deterministic scripts into `artifacts/canonical/`. Narrative Markdown is a cross-check, not a numeric input.

Run the migration checks from this directory:

```bash
python3 analysis/scripts/build_source_manifest.py artifacts/raw artifacts/provenance/source_manifest.csv
python3 analysis/scripts/validate_migration.py
```

## Known missing provenance

- Exact upstream XGBoost base commit/tag and replayable MetaGB patch.
- Final compiler/CMake/OpenMP/build flags and loaded `libxgboost` hash.
- Frozen environment lock and hardware/thread record.
- Dataset license/source-version audit.
- Canonical 0702 parser and final table/figure generators.

