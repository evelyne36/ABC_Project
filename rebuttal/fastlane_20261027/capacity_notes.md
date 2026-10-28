# N5 fixed-route capacity probe

`capacity.py` is a standalone regression probe. It does not change the controlled native source, tune hyperparameters, or launch experiment batches. It produces seven rows: `stored_anchor`, `slope_k1`, `slope_k2`, `slope_k3`, `affine_k1`, `affine_k2`, `affine_k3`.

## Invocation and input contract

```sh
python capacity.py \
  --root /path/to/controlled/CrossGB/xgboost-master \
  --data /path/to/data.npz \
  --model-dir /path/to/saved_anchor \
  --output /path/to/capacity_result
```

`--root` contains `python-package/xgboost`; the direct `python-package` directory is also accepted. Its import must resolve inside the requested package. The controlled native library must already be built. The script forces BLAS/OpenMP thread environment limits to one before importing NumPy or XGBoost.

The NPZ contains a finite dense `X` of shape `(n,p)`, finite regression `y` of shape `(n,)`, and nonempty, disjoint integer row arrays `train`, `val`, `test`. Splits need not cover every row. Feature preprocessing and target scale must match the saved model. No imputation or scaling is performed. Input features are rounded to float32 to match native prediction, then used in float64 numerical solves.

The model directory contains:

- `model.ubj` (or worker-produced `model.json`): saved scalar `reg:squarederror` native `gbtree` model with the MetaGB `metagb_leaf_features` JSON field.
- `result.json`: actual original fit values in `effective_params`, `estimator_params`, or legacy `params`, e.g. `{"effective_params": {"metagb_lambda_l2": 0.5, "learning_rate": 0.1}}`. For each field, those sources are checked in that order, skipping absent/null values, and the chosen source is recorded in `parameter_sources`. `model.ubj` takes precedence over `model.json` if both exist; the chosen `model_file` is recorded. `eta` is accepted if `learning_rate` is absent. The saved ridge must be finite and strictly positive. There are no silent defaults or tuning overrides. Optional `task_id` is propagated.

JSON traversal freezes the original anchor's train/validation/test routes using native CrossGB scalar prediction semantics: float32 feature value **<=** float32 split threshold goes left, while missing values follow `default_left`. Generic XGBoost `pred_leaf` can use `<` and disagree at threshold equality, so it is not used for CrossGB routes. Categorical splits are rejected; CLI inputs remain finite dense data. All saved trees are used. Before fitting, the stored coefficient replay must agree with native predictions on **all three splits** within absolute tolerance `1e-5`; otherwise the CLI fails. This checks original prediction semantics, not raw-refit coefficient identity.

## Fit semantics

For each of the six refitted variants, each tree sees its **own** current training residual `r = y_train - prediction_train`. Each predictive leaf fits `theta = solve(Z.T @ Z + Lambda, Z.T @ r)` and adds `eta * Z @ theta`; all leaves in a tree use the pre-tree residual. The variant's training prediction is then updated before the next tree. It never follows the original anchor's residual trajectory.

`k` counts nonbias input variables. Every affine predictive leaf adds one intercept; that intercept is unpenalized, while each slope receives the saved positive ridge. Slope variants have no intercept. Feature selection starts with the stored feature and greedily minimizes training penalized SSE. It fills to `min(k,p)` and breaks exact ties by feature index. Each variant selects using its own training residual. Validation/test labels and validation/test leaf occupancy never affect fitting, feature selection, stopping, or fallbacks. A leaf with one training row is still refitted, and a leaf with training rows but zero validation/test rows is still refitted.

Constant leaves remain their stored constants across every variant. A predictive leaf without **training** rows retains its stored one-variable slope, and affine variants attach a frozen zero intercept; this fallback is explicit in the per-leaf policy and counts. Thus every affine predictive leaf includes an intercept even if it has no training rows. Neither held-out occupancy nor labels decide the fallback.

## Outputs

`capacity.json` is schema version 1. It contains provenance paths, task ID, model dimensions, split sizes, fixed parameters, exact policy descriptions and limitations, native replay differences and tolerance, and:

- `metrics[variant].rmse.{train,val,test}` and corresponding `delta_rmse_vs_stored_anchor`.
- `metrics[variant].fit_timing.{wall_seconds,cpu_seconds}`: subset selection, coefficient fits, and training prediction updates; excludes shared routing. Original anchor training was preexisting, so its fit timing is **null**, not zero.
- `metrics[variant].prediction_timing[split]` and `prediction_timing_all_splits`: wall/CPU seconds for frozen-route prediction replay, separately from fitting. Shared JSON route extraction plus native anchor prediction is reported in `route_and_native_anchor_timing`. Replay timing excludes shared routing and is not end-to-end inference latency.
- `metrics[variant].counts`: total/predictive/constant leaf counts, refitted/fallback counts, nonbias slope count, affine intercept count, total scalar coefficients including constants and base score, actual-k mean and histogram. These count leaf coefficients, not routing thresholds or structure.
- `leaf_models[variant][tree][entry]`: leaf ID, ordered selected features, shrunken coefficients, affine flag, kind, actual k, training occupancy and policy where relevant. In affine entries, the last coefficient is the intercept. Stored anchor occupancy is not enumerated and `n_train=null`; occupancy is recorded for refit variants.

`predictions.npz` stores `VARIANT__SPLIT` predictions for every variant and split, `rows__SPLIT` original row indices, and `native_anchor__SPLIT`. Predictions retain the NPZ split row order.

## Interpretation and limits

This is a **fixed-route stagewise capacity probe**, not a full retraining of a k>1 CrossGB algorithm. Routes and initial feature choices come from the k1 model and may favor k1. Affine leaves are a separate approximation and are not the original algorithm. Original binned training statistics need not match raw-input ridge coefficients; the stored anchor must replay, but no coefficient-equality claim is made for raw-refit k1. Original structural training time is not included in probe fitting costs. Timings are small-sample measurements, not a dedicated latency benchmark.

The planned empirical scope of at least cpu_activity and protein plus sparse, dense, and offset synthetic cases is the parent controller's responsibility. This implementation task did not launch those fits. Every invoked probe emits train, validation and test metrics for all seven rows; no HPO is performed.

## Verification performed

```sh
python -m unittest discover -s rebuttal/fastlane_20261027 -p test_capacity.py -v
python rebuttal/fastlane_20261027/capacity.py --help
```

Nine deterministic tests passed, including exact-threshold CrossGB routing/default-child semantics, categorical rejection, worker-format model/parameter metadata and legacy fallback. They verify variant-owned stagewise residuals, held-out-label invariance, fitting leaves absent from held-out splits, held-out-occupancy invariance, unpenalized affine intercepts, actual-k/parameter counts, one-row training leaves, train-empty fallbacks, stored constants/replay, and rejection of invalid splits/ridge/routes. No compiled `libxgboost` was found in the checked local controlled source trees; a native smoke test was not run. The deployed CLI itself requires native anchor replay before any refits.
