# Raw artifact contract

`datacollection0702/` is the only authoritative result snapshot in this project.

- Main tabular directories are copied intact except `.DS_Store` and marker files.
- Synthetic contains only dim10/dim20 seed42 tasks plus the five protocol/audit files.
- Baseline logs and migrated metadata contain only the six retained models; CatBoost has 257 successful folds and is not rerun. Original eight-model metadata hashes are recorded under `../provenance/excluded_source_hashes.csv`.
- Time-series raw files remain complete, but paper-facing parsers must filter `pred_len == 1`.
- The baseline `task_status.json` is excluded because it contains abandoned-model states and is not an authoritative metric source. Success coverage is rebuilt from raw finish records.
