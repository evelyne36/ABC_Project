# MetaGB Embedded Efficiency Guide

This guide tracks optimization work for the embedded XGBoost MetaGB updater.
The priority is removing unnecessary computation while preserving original
MetaGB semantics.

## Non-negotiable Semantics

- Keep original MetaGB row and feature sampling order:
  row subsample, condition feature sampling, prediction feature sampling.
- Keep the original C `rand()` stream seeded with `srand(1)` for MetaGB sampling.
- Keep training split/leaf statistics on original MetaGB bin representative values.
- Keep prediction/cache update on raw feature values.
- Keep `prediction feature fraction` and `condition feature fraction` as distinct concepts.
- Do not replace original MetaGB bins with XGBoost `GHistIndexMatrix` cuts unless a parity test explicitly proves equivalence.

## Validation Protocol

Run every optimization in two phases.

1. Smoke parity and timing:

```bash
cd ./beforereview
PYTHONPATH=xgboost-master/python-package \
ORIGINAL_METAGB_DIR=../metagb \
python compare_original_current_metagb_xgb_regression.py \
  --mode quick \
  --datasets airfoil_self_noise.csv concrete_compressive_strength.csv diamonds.csv Food_Delivery_Time.csv \
  --models original_metagb current_metagb xgb_cpu \
  --outer-folds 2 --inner-folds 2 --outer-fold-index 0 \
  --max-candidates 5 --n-jobs 1 \
  --output-dir exp12/results/<run_id>
```

2. Broader regression pass:

```bash
cd ./beforereview
PYTHONPATH=xgboost-master/python-package \
ORIGINAL_METAGB_DIR=../metagb \
python compare_original_current_metagb_xgb_regression.py \
  --mode quick \
  --datasets all \
  --models original_metagb current_metagb xgb_cpu \
  --outer-folds 2 --inner-folds 2 --outer-fold-index 0 \
  --max-candidates 5 --n-jobs 1 --max-features 200 \
  --output-dir exp12/results/<run_id>
```

Accept an optimization only when current MetaGB RMSE stays within the normal
run-to-run numerical tolerance versus the previous embedded version and original
MetaGB. Speed should improve on total train time, not just on one small dataset.

## Low-risk Optimizations

These should preserve exact algorithmic semantics.

1. Loop reordering inside MetaGB histogram accumulation.
   Current expensive shape is condition feature -> prediction feature -> row.
   Prefer condition feature -> row -> prediction feature so each row's gradient,
   hessian, and condition bin are loaded once per condition feature.

2. Reuse workspaces.
   Avoid repeated allocation through `assign` inside inner loops. Resize when
   needed, then clear with `std::fill`.

3. Node-level predictive statistics.
   Compute leaf candidate stats once per node. When `split_sample=0`, condition
   rows and prediction rows match, so split totals can share the same stats.

4. Current-tree prediction value cache.
   For sampled prediction features, cache `TrainValue(row, pred_feature)` once
   per tree. This removes repeated `OriginalBin + BinOffset` lookups. Enable
   conditionally when the memory cost is reasonable.

## Medium-risk Optimizations

Discuss and validate carefully before implementing.

### Approved plan: current-tree prediction value cache

This is the next optimization to implement on the main path. It removes repeated
calls to `TrainValue(row, pred_feature)` while preserving the same values and row
accumulation order.

Implementation rules:

- Build the cache once per tree after `pred_features` is sampled.
- Cache only sampled prediction features, not all raw features.
- Use `double` values to match existing `TrainValue` arithmetic.
- Bias feature entries must be exactly `1.0`.
- `BestLeaf` and split histogram accumulation may read from the cache, but they
  must keep the same row and prediction-feature iteration order.
- The cache key is prediction-feature index position, not feature id alone,
  because `pred_features` order is part of tie-breaking and score aggregation.

Expected result: RMSE delta should be zero. If not, revert immediately and check
for an index/order mismatch.

Validation target:

- 4-dataset smoke first.
- Full 11-dataset quick pass after smoke.
- Compare against the low-risk loop-reordered/vector-row baseline.

1. Thread-local MetaGB histogram accumulation.
   Use XGBoost-style `ParallelFor2d`, per-thread histograms, then reduce.
   This keeps the mathematical objective but changes floating-point addition
   order, so tiny numeric drift is expected.

Thread-local implementation guardrails:

- Start with condition-feature-level parallelism only: each worker evaluates one
  or more condition features using a private MetaGB histogram buffer. Each
  condition feature should still scan rows in the original order and loop over
  prediction features in the sampled order.
- Reduce best split candidates in original `cond_features` order.
- Keep `metagb_score_agg=3` on the serial path first, because it consumes the
  original `rand()` stream while evaluating bins.
- If performance worsens or RMSE drift cannot be explained and bounded, revert
  this change and keep the pred-value cache only.
- Validate with both `--n-jobs 1` for semantic parity and `--n-jobs 4` for
  speed direction; the default regression comparison script often uses
  `--n-jobs 1`, where this optimization is intentionally inactive.

Attempt status on 2026-06-03:

- A first implementation that refactored each condition feature into a local
  best-split result and then reduced those results was reverted.
- Even with `--n-jobs 1`, before any parallel execution, the refactor changed
  RMSE on the 4-dataset smoke set. This means the risk was not merely
  floating-point parallel reduction; the original in-loop best update path is
  semantically sensitive.
- The retained main path is the verified prediction-value cache only. A future
  thread-local attempt should first build a debug-only equivalence harness that
  evaluates one condition feature both ways and compares every selected split
  field before changing the main `FindBestSplit` control flow.

2. XGBoost-style row partition buffers.
   Replace per-node `std::vector` allocation with contiguous row buffers and
   node ranges. This should preserve semantics but affects split traversal
   infrastructure.

### Approved medium-risk plan: stable contiguous row arena

Original MetaGB stores node membership as `cond_active_idx` and
`pred_active_idx` boolean masks, then scans `di = 0..num_samples-1`. Therefore
the semantic detail to preserve is not the boolean-mask representation itself,
but the stable global row-id order inside each node.

The embedded updater may replace per-node row vectors with contiguous row
arenas if it follows these rules:

- Maintain separate condition-row and prediction-row arenas.
- Each node stores spans into those arenas, not ownership of separate vectors.
- Root rows are appended in increasing row id order.
- Child rows are created by stable two-pass partition:
  first count left/right rows, then append left rows in parent order and right
  rows in parent order.
- Never use `std::partition`, XGBoost `CommonRowPartitioner`, or another
  partition primitive that can reorder rows unless a parity test proves the
  order is unchanged.
- All existing queries must keep their meaning:
  `NodeRows(nidx).cond` returns exactly the condition-active rows for that
  node, and `NodeRows(nidx).pred` returns exactly the prediction-active rows.

Validation for this change must compare against the pre-arena embedded version.
The expected RMSE delta is zero because row order and floating-point addition
order are intended to be unchanged.

Status after 2026-06-03 experiment: this plan preserved RMSE exactly on smoke
datasets, but it did not improve wall-clock time. The stable two-pass child
partition added an extra scan and offset the allocation savings:

- `diamonds.csv`: current MetaGB elapsed was about 51.8s with row arena versus
  about 50.6s in the low-risk workspace/loop-reordered version.
- `Food_Delivery_Time.csv`: about 43.9s with row arena versus about 43.3s in
  the low-risk version.

Therefore the row arena implementation should not be kept on the main path
unless it is paired with a larger redesign, such as level-local reusable
buffers or a parent/child histogram cache that benefits from span-based row
storage. The current main implementation intentionally remains at the
low-risk loop-reordered/vector-row version.

## High-risk Optimizations

Do not implement without a dedicated parity plan.

1. Histogram subtraction trick.
   For each condition feature and sampled prediction feature, build one child
   histogram and derive the sibling histogram as parent minus child. This can be
   very fast, but requires carefully caching parent MetaGB histograms and keeping
   condition-row and prediction-row semantics separate.

Subtraction-trick constraints:

- Discuss again before implementation.
- Initially allow only `metagb_split_sample=0`.
- Do not reuse condition-row histograms for prediction-row leaf fitting.
- Keep a fallback to full histogram construction for unsupported modes.

Approved experiment on 2026-06-03:

- Apply histogram subtraction only to split histograms over `rows.cond`.
- Never use this cache for `BestLeaf`, which must keep using `rows.pred`.
- Cache each node's complete condition-feature histogram in the same sampled
  `cond_features` and `pred_features` order used by `FindBestSplit`.
- After splitting a parent node, scan only the child with fewer condition rows;
  derive the sibling histogram as `parent_hist - scanned_child_hist`.
- Use the derived child histogram only as an input to that child's later
  `FindBestSplit`; root and any missing-cache fallback must use the existing
  full scan path.
- Expected accuracy risk: tiny floating-point drift is possible because
  subtraction changes the sibling accumulation path. Larger RMSE changes,
  selected-candidate changes on smoke, or slower total training time mean this
  experiment should be reverted.

Attempt status on 2026-06-03:

- The complete-node histogram cache/subtraction prototype was reverted.
- On the 4-dataset smoke run it was already much slower on the first two
  current MetaGB datasets: airfoil increased from about 0.63s to 6.1s and
  concrete from about 1.21s to 14.0s.
- RMSE also showed non-zero drift, so the experiment failed both speed and
  parity criteria.
- Root cause: caching the full `condition feature x prediction feature x bin`
  tensor for every node adds large memory traffic and copy cost before the
  sibling subtraction can pay off. A future attempt would need a much narrower
  per-sibling/per-feature cache with a dedicated equivalence harness, not the
  full-node tensor prototype.

2. Reusing XGBoost bin matrices directly.
   This is attractive for speed but risky because original MetaGB uses its own
   bin representative semantics. Treat it as a separate research branch.

3. Parallel random sampling or XGBoost samplers.
   These can break the original `rand()` trajectory and change feature/row
   selection, so they are not appropriate for parity-preserving optimization.
