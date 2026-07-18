# Time-series `pred_len=1` scope

Authoritative files:

- `project_paperwriting/artifacts/raw/datacollection0702/datacollection_ts_tree_seed42__non_pems/summary.csv`
- `candidates.csv`
- `candidate_configs.json`
- `split_manifest.json`
- `parameter_audit.json`

Paper filter: `status == ok` and `pred_len == 1`.

Verified coverage: 25 result rows over seven datasets. LightGBM, XGBoost and CatBoost each cover seven datasets; MetaGB covers ETTh1, ETTh2, ILI and exchange_rate. ETTm1, ETTm2 and Weather remain NA for MetaGB.

This is seed42, fixed-split, one-step targeted evidence. It must not be described as a complete long-horizon forecasting benchmark.

