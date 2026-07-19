# MetaGB Embedded XGBoost Project

This repository contains an XGBoost-integrated implementation of MetaGB, named
`current_metagb`, together with the original standalone CFFI MetaGB baseline,
benchmark scripts, and Random300/Random600 experiment suites.

The main experimental comparison is:

```text
xgb_cpu vs current_metagb
```

`xgb_cpu` is the standard CPU hist updater from the same local XGBoost checkout.
`current_metagb` is a CPU hist updater embedded into XGBoost that replaces
constant tree leaves with feature-selected local linear leaves.

## Project Layout

```text
check_codereview0604cc/
  beforereview/
    xgboost-master/                 XGBoost source with current_metagb changes
    exp12/                          historical experiment scripts and results
    scripts/check_metagb_runtime.py runtime verification helper
  metagb/                           original standalone MetaGB, Python + CFFI + C
  dataset/
    TabArena_regression/
    TabArena_binary/
    TabArena_multiclass/
  random300/                        lighter 300-candidate experiment suite
  random600/                        600-candidate main experiment suite
  bench_timing.py                   quick original/current/xgb timing benchmark
  启动指南.md                         detailed build and run guide
  current_metagb_paper_notes.md     algorithm and paper-writing notes
  修改记录.md                         implementation and benchmark changelog
```

## Modeling Idea

Standard XGBoost stores one scalar value in each leaf:

```text
leaf output = w
```

`current_metagb` stores a leaf coefficient and a selected predictive feature:

```text
leaf output = beta_l * z_j(x)
```

The selected feature `z_j(x)` can be an input feature or a bias feature. The
bias feature is always equal to `1`, so `current_metagb` can fall back to the
ordinary constant-leaf form:

```text
beta_l * 1 = beta_l
```

This makes `current_metagb` a strict extension of scalar-leaf gradient boosted
trees. Its key modeling distinction is the decoupling of two feature roles:

```text
conditional features: used to split and route samples
predictive features: used inside leaves for local linear responses
```

Compared with XGBoost, the split statistics are heavier. XGBoost mainly needs
`sum(g)` and `sum(h)` per histogram bin. `current_metagb` evaluates predictive
leaf responses and therefore accumulates statistics such as:

```text
grad_x  = sum_i g_i * x_ij
hess_xx = sum_i h_i * x_ij^2
hess    = sum_i h_i
```

This extra conditional-feature by predictive-feature histogram is the main
source of both the method's expressiveness and its runtime cost.

## Implementation Summary

`current_metagb` is integrated as a CPU hist updater in:

```text
beforereview/xgboost-master/src/tree/updater_metagb_hist.cc
```

The Python/sklearn entry point is the local XGBoost package under:

```text
beforereview/xgboost-master/python-package
```

Use `enable_metagb=True` to activate the embedded updater. Without that
parameter, the same checkout runs ordinary CPU XGBoost.

Important implementation files:

```text
beforereview/xgboost-master/src/tree/updater_metagb_hist.cc
beforereview/xgboost-master/src/tree/param.h
beforereview/xgboost-master/src/tree/tree_model.cc
beforereview/xgboost-master/src/predictor/cpu_predictor.cc
beforereview/xgboost-master/python-package/xgboost/sklearn.py
```

Multiclass experiments should use the default scalar path:

```text
multi_strategy = one_output_per_tree
```

The `multi_output_tree` vector branch is experimental and should be treated as
an ablation or compression study, not as the main result path.

## Current Optimization State

The retained histogram-build optimization is controlled by:

```text
METAGB_HIST_BUILD_MODE=auto|row_serial|cf_parallel|row_parallel
```

The default `auto` mode selects `cf_parallel` for wider or higher-work nodes and
keeps lighter behavior for small tables where parallel overhead can dominate.

The current recommended default is:

```text
cf_parallel-capable auto path + metagb_hist_subtract=false
```

Histogram subtraction remains available as an explicit ablation, but it is not
the default. In the current implementation it was slower in representative
tests because caching or copying full 3D node histograms outweighed the saved
subtraction work.

Rejected or rolled-back experiments:

```text
row_parallel full thread-local 3D histograms: too much clear/reduce overhead
cf_immediate blocked immediate split evaluation: no speed win
InitOriginalBins upper_bound digitization: faster but changed RMSE
InitOriginalBins parallel-only variant: correct but slower
```

See `METAGB_HIST_BUILD_OPTIMIZATION_PLAN.md` and `修改记录.md` for the detailed
benchmark trail.

## Build And Verify

On Linux, rebuild the local XGBoost checkout on the target machine. Do not reuse
a macOS build directory.

```bash
cd ./beforereview/xgboost-master
rm -rf build
mkdir build
cd build
cmake .. -DUSE_OPENMP=ON -GNinja
cmake --build . -j"$(nproc)"
```

Verify that the local package loads and `enable_metagb` is available:

```bash
cd ./beforereview
PYTHONPATH=xgboost-master/python-package \
  python3 scripts/check_metagb_runtime.py
```

Expected status:

```text
enable_metagb: 1
status: OK
```

## Quick Timing Benchmark

Use `bench_timing.py` for a fast one-split comparison of:

```text
original_metagb
current_metagb
xgb_cpu
```

Fair single-core timing:

```bash
/opt/anaconda3/bin/python bench_timing.py \
  --dataset-dir dataset/TabArena_regression \
  --datasets airfoil_self_noise.csv concrete_compressive_strength.csv wine_quality.csv physiochemical_protein.csv \
  --n-estimators 80 \
  --max-depth 4 \
  --n-jobs 1 \
  --output-csv beforereview/exp12/results/bench_three_models_regular_njobs1.csv
```

Representative single-core observation from the current local run:

```text
small tables: current_metagb can be slower than original_metagb
ordinary medium/large tables: current_metagb is roughly comparable to slightly faster
wide tables: current_metagb can still be clearly faster
```

For example, `QSAR-TID-11.csv` with 1024 features showed about `2.4x` speedup
for `current_metagb` over the original standalone MetaGB under `n_jobs=1`.
Standard `xgb_cpu` remains much faster because it does not build the heavier
MetaGB conditional-predictive 3D histograms.

## Random300 Experiments

`random300/` is the lighter experiment suite. It keeps the same nested-CV and
logging structure as Random600, but uses 300 candidates per model.

Candidate policy:

```text
xgb_cpu/current_metagb: paired shared backbones generated from the same seed
current_metagb: balanced 10 pred_feature_fraction x 3 score_agg combinations
Random300 is the first 300 candidates of Random600
```

Smoke test:

```bash
python3 random300/scripts/run_random300_server.py \
  --mode quick \
  --task-types regression binary multiclass \
  --regression-datasets airfoil_self_noise.csv \
  --binary-datasets diabetes.csv \
  --multiclass-datasets anneal.csv \
  --models xgb_cpu current_metagb \
  --max-candidates 30 \
  --outer-folds 2 --inner-folds 2 \
  --single-outer-fold --outer-fold-index 0 \
  --task-workers 1 --n-jobs 1 \
  --run-id smoke_random300
```

Full 16-core run:

```bash
python3 random300/scripts/run_random300_server.py \
  --mode full \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb \
  --max-candidates 300 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  --experiment-name random300_main_xgb_vs_current_metagb
```

## Random600 Experiments

`random600/` is the larger main experiment suite.

Candidate policy:

```text
xgb_cpu/current_metagb: 600 paired shared backbones
current_metagb: each pred_feature_fraction x score_agg combination appears 20 times
```

Full 16-core run:

```bash
python3 random600/scripts/run_random600_server.py \
  --mode full \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb \
  --max-candidates 600 \
  --outer-folds 5 --inner-folds 4 \
  --task-workers 16 --n-jobs 1 \
  --experiment-name random600_main_xgb_vs_current_metagb
```

The shorthand is:

```bash
python3 random600/scripts/run_random600_full.py
```

## Recommended Parallelism

For the main `xgb_cpu vs current_metagb` experiments on a 16-core machine, use:

```text
--task-workers 16 --n-jobs 1
```

The experiment has many independent tasks:

```text
task_type x dataset x outer_fold x model
```

Task-level parallelism has lower coordination cost and avoids oversubscribing
the CPU. Use `4 task-workers x n_jobs=4` only for targeted wide-table timing
experiments where single-task latency matters more than total throughput. Avoid
`1 task-worker x n_jobs=16` for the main nested-CV sweep unless there are too
few independent tasks to fill the machine.

## Outputs

Random300 and Random600 write the same result structure under their own roots:

```text
random300/results/<run_id>/
random300/logs/<run_id>/

random600/results/<run_id>/
random600/logs/<run_id>/
```

Important result files:

```text
candidate_configs.json
candidates.csv
summary.csv
aggregate.csv
pred_fraction_analysis.csv
parameter_audit.json
run_metadata.json
```

`parameter_audit.json` records important defaults and warnings, including
whether `metagb_hist_subtract` was enabled.

## Documentation Map

```text
启动指南.md
  Practical build, smoke-test, Random300, and Random600 run instructions.

current_metagb_paper_notes.md
  Algorithm explanation, XGBoost comparison, multiclass scalar/vector notes,
  and paper-positioning language.

METAGB_HIST_BUILD_OPTIMIZATION_PLAN.md
  Histogram-build optimization plan, benchmark results, retained and rejected
  implementation variants.

修改记录.md
  Chronological engineering record, precision bug notes, rollback decisions,
  random300 creation, and benchmark validation.

random600_server_run_commands.md
  Server-oriented command cookbook for Random600.

METAGB_SPEEDUP_ANALYSIS.md
  Earlier performance-bottleneck analysis and candidate optimization ideas.
```

## Current Practical Takeaways

- `current_metagb` is a modeling extension of XGBoost, not a replacement for the
  highly optimized ordinary XGBoost hist updater.
- The method is computationally heavier because split search uses
  conditional-feature by predictive-feature statistics.
- The embedded implementation is most valuable when XGBoost compatibility,
  sklearn integration, multiclass support, and faster-than-standalone MetaGB
  execution matter.
- For fair speed comparisons with the standalone MetaGB baseline, use
  `n_jobs=1`.
- For full paper experiments on a 16-core server, prefer
  `--task-workers 16 --n-jobs 1`.
- Keep `metagb_hist_subtract=false` for main results unless explicitly running a
  subtraction ablation.
