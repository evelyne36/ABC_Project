# 0629 MGB vs XGB Analysis Plan

## Scope

Use `mgb_datacollection_by0629` as the dedicated 0629 analysis and report
directory. Read prior TabArena evidence from `datacollection0624`, and write new
derived tables and conclusions only under `mgb_datacollection_by0629`.

## Data Families

1. `datacollection0624`: TabArena main evidence, split by regression/binary/
   multiclass and drop/onehot.
2. `mgb_datacollection_by0629/tabarena30_*`: expensive TabArena tails using
   XGB-best backbone plus 10 pred fractions x 3 score aggregations.
3. `mgb_datacollection_by0629/tabular_otherdata_*`: additional moderate-size
   tabular datasets selected for time economy and MetaGB plausibility.
4. `mgb_datacollection_by0629/synthetic_main102050_random300_seed42`:
   synthetic mechanism evidence, 3000 samples and currently available 10/20
   feature results.

## Metrics

Compare only paired folds where both `xgb_cpu` and `current_metagb` finished
with `status=ok`.

- Regression: RMSE, lower is better.
- Binary: AUC, higher is better.
- Multiclass: logloss where present, lower is better.
- Tie rule: AUC absolute difference <= 1e-4; RMSE/logloss relative difference
  <= 0.1%.

## Outputs

- `derived/fold_results.csv`
- `derived/paired_fold_results.csv`
- `derived/dataset_level_mgb_vs_xgb.csv`
- `derived/win_tie_loss_by_group.csv`
- `derived/coverage_status.csv`
- `derived/remaining_time_estimates.csv`
- `tables/table_*.md`
- `mgb_vs_xgb_comprehensive_analysis.md`

## Interpretation Principles

- Keep full Random300/600 and TabArena30 fixed-backbone results separate.
- Treat synthetic as mechanism evidence, not as a proxy for real-data average
  performance.
- Report dataset size and `rows * features^2` pressure because MetaGB cost is
  feature-sensitive.
- Frame runtime honestly as a quality-cost tradeoff, not a speed claim.
