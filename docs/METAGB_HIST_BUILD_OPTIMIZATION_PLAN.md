# MetaGB Histogram Build Optimization Demo Plan

## Goal

Speed up `current_metagb` training by optimizing the histogram build stage in
`beforereview/xgboost-master/src/tree/updater_metagb_hist.cc`.

Current priority is runtime speed. Memory overhead is acceptable for demo
experiments, but implementations should still keep an explicit rollback path.

## Baseline Observation

The scalar MetaGB updater already has:

- row-major predictive value cache
- parent/sibling histogram subtraction cache
- parallel split evaluation over conditional features

The remaining hot path is histogram construction. For each active node it builds
a 3D histogram over:

```text
conditional feature x predictive feature x bin
```

The current optimized path fills this histogram with a row-first serial loop.
It also creates a large `node_hist` vector per node.

## Demo Candidates

1. Reuse node histogram workspace
   - Replace per-node local allocation with the existing `hist_3d_workspace_`.
   - Keep cache semantics unchanged by copying into `node_hist_cache_` only when
     histogram subtraction needs a persistent node histogram.
   - Expected gain: lower allocator overhead and less memory churn.

2. Conditional-feature parallel histogram build
   - Parallelize build by `cf_idx`.
   - Each thread writes a disjoint `cf_slice`, so no atomics are required.
   - Expected gain: better CPU utilization on multi-core runs.
   - Risk: scans rows once per conditional feature, so it can lose cache locality
     against the row-first serial loop.

3. Thread-local row-parallel histogram build
   - Use thread-local 3D histograms, then reduce.
   - Expected gain: preserves row-first access and parallelizes the true hot
     loop.
   - Risk: memory scales by `threads x n_cf x n_pred x max_bin`; use only if a
     guarded prototype shows clear benefit.

4. Blocked/tiled histogram build
   - Build/evaluate subsets of conditional or predictive features at a time.
   - Expected gain: better cache locality and bounded workspace.
   - Risk: more intrusive, especially with histogram subtraction cache.

## Runtime Switches

Demo implementations should be controlled by environment variables so each
variant can be benchmarked without recompiling:

```text
METAGB_HIST_BUILD_MODE=auto|row_serial|cf_parallel|row_parallel
```

Demo result so far:

- `row_serial`: original row-first build, now using reusable workspace capacity.
- `cf_parallel`: best current demo for wide tables; disjoint writes by
  conditional-feature slice.
- `row_parallel`: not adopted so far; thread-local full 3D histograms make
  clearing and reduction too expensive.
- `auto`: default candidate. Uses `cf_parallel` only when the node-level work is
  large enough; otherwise keeps `row_serial`.

## Evaluation

Use smoke benchmarks first, then larger/wider datasets.

Metrics:

- training time
- prediction time
- RMSE/logloss/AUC parity
- crash/OOM behavior

Runtime interpretation should keep single-fit fairness and full-experiment
throughput separate:

- Use `n_jobs=1` when comparing `current_metagb` against the standalone CFFI
  `metagb/` baseline, since the standalone baseline does not consume XGBoost's
  `n_jobs`.
- Use `--task-workers 16 --n-jobs 1` for the Random300/Random600 16-core
  nested-CV sweeps. This favors task-level throughput and avoids CPU
  oversubscription.
- Treat `--task-workers 4 --n-jobs 4` as a targeted wide-table latency
  experiment, not as the default main-experiment setting.
- Avoid hiding wide-table behavior in a single global average. Datasets like
  `QSAR-TID-11.csv` should be reported separately because they stress the
  conditional-feature by predictive-feature 3D histogram most directly.

Rollback condition:

- compile failure
- correctness regression
- consistent slowdown on representative datasets
- unstable memory behavior

## Demo Results

Environment:

- local macOS build
- `/opt/anaconda3/bin/python`
- `n_jobs=4`
- `max_bin=255`

Sanity:

- `scripts/check_metagb_runtime.py`: OK after rebuild.
- `bench_metagb_hist_build_demo.py`: `py_compile` OK.

Representative timings:

| Dataset | Config | Mode | Mean train time | Note |
|---|---|---:|---:|---|
| QSAR-TID-11.csv | 10 trees, depth 3, colsample 0.2, pred 0.2 | row_serial | 11.52s | baseline |
| QSAR-TID-11.csv | same | cf_parallel | 4.11s | ~2.8x faster |
| QSAR-TID-11.csv | same | row_parallel | 13.99s | rejected |
| QSAR-TID-11.csv | same | auto | 4.34s | default candidate |
| wine_quality.csv | 80 trees, depth 4, colsample 0.8, pred 0.8 | row_serial | 0.235s | small/medium table |
| wine_quality.csv | same | cf_parallel | 0.174s | faster |
| airfoil_self_noise.csv | 80 trees, depth 4, colsample 0.8, pred 0.8 | row_serial | 0.049s | tiny table |
| airfoil_self_noise.csv | same | cf_parallel | 0.053s | slight slowdown |

Correctness:

- RMSE matched across modes in the representative benchmark rows, except one
  tiny-table first-run fluctuation that did not reproduce in repeated runs.

Current decision:

- Keep `auto` as the default demo behavior.
- Use `cf_parallel` for wide/high-work nodes.
- Keep `row_parallel` only as an explicit experiment mode; do not select it in
  auto.

## Follow-up Speed Round

Additional changes tested after the first `cf_parallel` win:

- Parallelized parent-sibling histogram subtraction.
- Delayed saving `node_hist_cache_` until a node actually has a valid split.
- Tested `metagb_hist_subtract=true/false`; disabling it was faster for the
  representative wide-table `cf_parallel` path.
- Changed `metagb_hist_subtract` default to `false`.
- Added a guarded parallel `BestLeaf` path for large predictive-feature work.
- Made `auto` more aggressive for `cf_parallel` selection.
- Reduced inner-loop overhead by reusing row pointers into
  `pred_train_value_cache_`.

Final representative timings:

| Dataset | Config | Mode | hist_subtract | Mean train time | Note |
|---|---|---:|---:|---:|---|
| QSAR-TID-11.csv | 10 trees, depth 3, colsample 0.2, pred 0.2 | auto | false | 2.01s | final default candidate |
| QSAR-TID-11.csv | same | cf_parallel | false | 2.06s | explicit best path |
| QSAR-TID-11.csv | same | row_serial | false | 10.96s | baseline after no-subtract |
| QSAR-TID-11.csv | same | cf_parallel | true | 3.89s | slower due to cache overhead |
| wine_quality.csv | 80 trees, depth 4, colsample 0.8, pred 0.8 | auto | false | 0.152s | faster than row_serial |
| wine_quality.csv | same | row_serial | false | 0.223s | small/medium baseline |
| airfoil_self_noise.csv | 80 trees, depth 4, colsample 0.8, pred 0.8 | auto | false | 0.047s | close to row_serial |
| airfoil_self_noise.csv | same | row_serial | false | 0.044s | tiny-table baseline |

Current final decision:

- Keep the second-round changes.
- Do not roll back: smoke checks passed and representative RMSE stayed matched.
- `row_parallel` remains rejected.
- `metagb_hist_subtract=true` remains available as an explicit parameter but is
  no longer the default.

## Histogram Subtraction Recheck

Rechecked on the final code, not the intermediate version.

| Dataset | Config | Mode | hist_subtract=false | hist_subtract=true | Ratio true/false |
|---|---|---:|---:|---:|---:|
| QSAR-TID-11.csv | 10 trees, depth 3, colsample 0.2, pred 0.2 | auto | 2.00s | 3.85s | 1.92x slower |
| QSAR-TID-11.csv | same | cf_parallel | 2.19s | 3.74s | 1.71x slower |
| wine_quality.csv | 80 trees, depth 4, colsample 0.8, pred 0.8 | auto | 0.224s | 0.281s | 1.25x slower |
| wine_quality.csv | same | row_serial | 0.338s | 0.366s | 1.08x slower |
| airfoil_self_noise.csv | 80 trees, depth 4, colsample 0.8, pred 0.8 | auto | 0.061s | 0.072s | 1.19x slower |

Current conclusion:

- The current histogram subtraction implementation has no measured speed
  benefit on the tested representative paths.
- It is slower because saving `node_hist_cache_` copies a full 3D histogram for
  every valid split node, and the implementation still clears/allocates the
  destination workspace before discovering whether subtraction can be used.
- Keep the default as `metagb_hist_subtract=false`.
- A future subtraction redesign should only be attempted if it avoids full hist
  copies, for example by moving reusable cache buffers or caching only selected
  child histograms.

## Rejected Follow-up Experiments

The following experiments were attempted after the final `cf_parallel` baseline.
They were not kept because they did not improve speed without risk.

| Experiment | Dataset/config | Result | Decision |
|---|---|---:|---|
| `cf_immediate`: per-cf temporary hist + immediate split evaluation | QSAR-TID-11.csv, 10 trees, depth 3, colsample 0.2, pred 0.2 | `cf_immediate` 2.19s vs `cf_parallel` 2.18s | rolled back |
| `InitOriginalBins`: parallel feature loop + `upper_bound` digitization | same | faster at 1.92s, but RMSE changed from 3.98 to ~5.70 | rolled back immediately |
| `InitOriginalBins`: parallel feature loop only, original while digitization | same | RMSE restored, but 2.09s vs previous ~2.01s | rolled back |

Current retained state after rollbacks:

- `auto`: 2.12s on the representative QSAR config.
- `cf_parallel`: 2.07s on the same config.
- RMSE stable at 3.983936.

Conclusion:

- Keep the previously validated `cf_parallel` + `metagb_hist_subtract=false`
  path.
- Do not use `cf_immediate`.
- Do not parallelize `InitOriginalBins` with the attempted implementations.
