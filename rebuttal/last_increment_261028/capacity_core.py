#!/usr/bin/env python3
"""Regression-only, fixed-route stagewise capacity probe; no search or retuning.

--root is the controlled CrossGB checkout (containing python-package/xgboost).
NPZ must contain X, y, train, val, test; splits are disjoint integer row indices.
The saved model directory contains model.ubj (or model.json) and result.json.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

SPLITS = ("train", "val", "test")
VARIANTS = ("stored_anchor", "slope_k1", "slope_k2", "slope_k3", "affine_k1", "affine_k2", "affine_k3")


def scalar(value):
    values = np.asarray(json.loads(value) if isinstance(value, str) and value.startswith("[") else value).ravel()
    if len(values) != 1:
        raise ValueError("Only scalar regression base scores are supported")
    return float(values[0])


def clock_pair():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    return {"wall_seconds": time.perf_counter() - start[0], "cpu_seconds": time.process_time() - start[1]}


def design(X, features, affine):
    Z = X[:, features]
    return np.column_stack((Z, np.ones(len(X)))) if affine else Z


def solve_leaf(X, residual, features, ridge, affine):
    """Fit a positive ridge on slopes; affine intercept is unpenalized."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Probe requires saved metagb_lambda_l2 > 0")
    Z = design(X, features, affine)
    penalty = np.full(Z.shape[1], ridge)
    if affine:
        penalty[-1] = 0.0
    theta = np.linalg.solve(Z.T @ Z + np.diag(penalty), Z.T @ residual)
    error = residual - Z @ theta
    objective = float(error @ error + np.sum(penalty * theta * theta))
    return theta, objective


def select_features(X, residual, start, want, ridge, affine):
    """Training-only forward ridge objective, anchored on stored feature.

    Fill up to min(requested k, number of input features), even for zero gain;
    deterministic ties use ascending feature index. No held-out occupancy input.
    """
    selected = [int(start)]
    target = min(want, X.shape[1])
    while len(selected) < target:
        best = None
        for candidate in range(X.shape[1]):
            if candidate in selected:
                continue
            _, objective = solve_leaf(X, residual, selected + [candidate], ridge, affine)
            if best is None or objective < best[0]:
                best = (objective, candidate)
        selected.append(best[1])
    theta, _ = solve_leaf(X, residual, selected, ridge, affine)
    return selected, theta


def validate_arrays(X, y, splits):
    # Native input precision is float32. Refits use those same values in float64.
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0 or y.ndim != 1 or len(y) != len(X):
        raise ValueError("Expected X shape (n,p), p>0, and y shape (n,)")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Finite dense regression data required; no imputation is performed")
    clean = {}
    for name in SPLITS:
        rows = np.asarray(splits[name])
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) == 0:
            raise ValueError(f"{name} must be nonempty 1D integer row indices")
        if len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= len(X):
            raise ValueError(f"Invalid or duplicate {name} row indices")
        clean[name] = rows.astype(np.int64)
    combined = np.concatenate(list(clean.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("train, val and test must be disjoint")
    return X, y, clean


def crossgb_leaf_routes(X, trees):
    """Match native CrossGB scalar predictor: missing->default, otherwise <=.

    Generic XGBoost pred_leaf uses different equality semantics. Round both
    values and stored thresholds to native float32 before comparison.
    """
    X = np.asarray(X, dtype=np.float32)
    routes = np.empty((len(X), len(trees)), dtype=np.int64)
    for index, tree in enumerate(trees):
        left = np.asarray(tree["left_children"], dtype=np.int64)
        right = np.asarray(tree["right_children"], dtype=np.int64)
        thresholds = np.asarray(tree["split_conditions"], dtype=np.float32)
        split_features = np.asarray(tree["split_indices"], dtype=np.int64)
        default_left = np.asarray(tree["default_left"], dtype=bool)
        split_types = np.asarray(tree.get("split_type", np.zeros(len(left))), dtype=np.int64)
        if np.any(split_types != 0) or tree.get("categories", []):
            raise ValueError("Categorical splits are unsupported by this numeric CrossGB router")
        if any(len(array) != len(left) for array in (right, thresholds, split_features, default_left)):
            raise ValueError("Inconsistent JSON routing arrays")
        stack = [(0, np.arange(len(X)))]
        visited = 0
        while stack:
            node, rows = stack.pop()
            if not len(rows):
                continue
            visited += 1
            if visited > len(left) or node < 0 or node >= len(left):
                raise ValueError("Invalid or cyclic JSON tree")
            if left[node] == -1:
                routes[rows, index] = node
                continue
            feature = split_features[node]
            if feature < 0 or feature >= X.shape[1] or not np.isfinite(thresholds[node]):
                raise ValueError("Invalid numeric split feature or threshold")
            values = X[rows, feature]
            go_left = np.where(np.isnan(values), default_left[node], values <= thresholds[node])
            stack.append((int(right[node]), rows[~go_left]))
            stack.append((int(left[node]), rows[go_left]))
    return routes


def stored_tree(tree):
    left = np.asarray(tree["left_children"], dtype=np.int64)
    values = np.asarray(tree["split_conditions"], dtype=np.float64)
    features = np.asarray(tree["metagb_leaf_features"], dtype=np.int64)
    if len(left) != len(values) or len(left) != len(features):
        raise ValueError("Inconsistent native tree arrays")
    return [dict(leaf=int(node), features=[int(features[node])], theta=[float(values[node])],
                 affine=False, kind="stored", k=1, n_train=None)
            for node in np.flatnonzero(left == -1)]


def contribution(X, route, entries):
    out = np.zeros(len(X), dtype=np.float64)
    for entry in entries:
        mask = route == entry["leaf"]
        if not mask.any():
            continue
        features = entry["features"]
        if entry["kind"] == "constant":
            out[mask] = entry["theta"][0]
        else:
            out[mask] = design(X[mask], features, entry["affine"]) @ np.asarray(entry["theta"])
    return out


def anchor_entries(trees, num_features):
    fitted = []
    for tree in trees:
        entries = stored_tree(tree)
        for entry in entries:
            if not 0 <= entry["features"][0] < num_features:
                entry.update(kind="constant", features=[], k=0)
        fitted.append(entries)
    return fitted


def predict_frozen(X, routes, fitted, base_score):
    """Float64 replay for probe models; also used for anchor diagnostics."""
    prediction = np.full(len(X), base_score, dtype=np.float64)
    for index, entries in enumerate(fitted):
        prediction += contribution(X, routes[:, index], entries)
    return prediction


def predict_stored_native(X, routes, trees, base_score):
    """Replay stored scalar CrossGB leaves with native float32 arithmetic.

    The CPU predictor returns a float leaf product and adds each tree to a
    float output cursor in model order. Keep this path separate from float64
    probe fitting/prediction: casting a final float64 sum does not reproduce it.
    """
    X = np.asarray(X, dtype=np.float32)
    prediction = np.full(len(X), base_score, dtype=np.float32)
    for index, tree in enumerate(trees):
        leaves = np.asarray(routes[:, index], dtype=np.int64)
        values = np.asarray(tree["split_conditions"], dtype=np.float32)[leaves]
        features = np.asarray(tree["metagb_leaf_features"], dtype=np.int64)[leaves]
        products = values.copy()
        predictive = (features >= 0) & (features < X.shape[1])
        rows = np.flatnonzero(predictive)
        inputs = X[rows, features[rows]]
        # Native CrossGB treats a missing predictive feature as zero.
        products[rows] *= np.where(np.isnan(inputs), np.float32(0), inputs)
        prediction += products
    return prediction


def fit_variant(X_train, y_train, routes_train, trees, base_score, ridge, eta, k, affine):
    """Every variant owns its cursor and refreshes its residual after each tree."""
    cursor = np.full(len(X_train), base_score, dtype=np.float64)
    fitted = anchor_entries(trees, X_train.shape[1])
    for index, entries in enumerate(fitted):
        # All leaves within this tree fit the same pre-tree residual.
        residual = y_train - cursor
        for entry in entries:
            mask = routes_train[:, index] == entry["leaf"]
            entry["n_train"] = int(mask.sum())
            if entry["kind"] == "constant":
                entry["policy"] = "frozen_stored_constant"
                continue
            if not mask.any():
                entry["policy"] = "no_training_rows_keep_stored_slope"
                if affine:
                    entry.update(affine=True, theta=entry["theta"] + [0.0])
                continue
            features, theta = select_features(X_train[mask], residual[mask], entry["features"][0], k, ridge, affine)
            entry.update(features=features, theta=(eta * theta).tolist(), affine=affine,
                         kind="refit", k=len(features), policy="training_only_refit")
        cursor += contribution(X_train, routes_train[:, index], entries)
    return fitted


def parameter_counts(fitted):
    entries = [e for tree in fitted for e in tree]
    predictive = [e for e in entries if e["kind"] != "constant"]
    ks = [e["k"] for e in predictive]
    return {"leaf_count": len(entries), "predictive_leaf_count": len(predictive),
            "constant_leaf_count": len(entries) - len(predictive),
            "leaves_refit": sum(e["kind"] == "refit" for e in entries),
            "no_training_rows_fallbacks": sum(e.get("policy") == "no_training_rows_keep_stored_slope" for e in entries),
            "nonbias_slope_count": sum(ks), "affine_intercept_count": sum(e["affine"] for e in entries),
            "total_scalar_parameters_including_base": 1 + sum(len(e["theta"]) for e in entries),
            "actual_k_mean_predictive_leaves": float(np.mean(ks)) if ks else 0.0,
            "actual_k_histogram": {str(k): ks.count(k) for k in sorted(set(ks))}}


def run_probe(X, y, splits, routes, trees, base_score, ridge, eta):
    """Pure fixed-route core. Held-out labels are consulted only for metrics."""
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(eta) or eta <= 0:
        raise ValueError("Saved ridge and learning rate must be finite and positive")
    X, y, splits = validate_arrays(X, y, splits)
    ntrees = len(trees)
    if not ntrees:
        raise ValueError("Saved model contains no trees")
    for name in SPLITS:
        route = np.asarray(routes[name])
        if route.shape != (len(splits[name]), ntrees) or not np.isfinite(route).all() or not np.equal(route, route.astype(np.int64)).all():
            raise ValueError(f"Invalid {name} native leaf routes")
        for index, tree in enumerate(trees):
            legal = np.flatnonzero(np.asarray(tree["left_children"]) == -1)
            if not np.isin(route[:, index], legal).all():
                raise ValueError(f"Nonleaf route in {name} tree {index}")
    fitted_models, metrics, predictions = {}, {}, {}
    for name in VARIANTS:
        affine = name.startswith("affine")
        start = clock_pair()
        if name == "stored_anchor":
            fitted = anchor_entries(trees, X.shape[1])
            fit_timing = {"wall_seconds": None, "cpu_seconds": None, "reason": "preexisting anchor; original training timing not measured"}
        else:
            fitted = fit_variant(X[splits["train"]], y[splits["train"]], routes["train"], trees,
                                 base_score, ridge, eta, int(name[-1]), affine)
            fit_timing = elapsed(start)
        start = clock_pair()
        pred, split_timing = {}, {}
        for split in SPLITS:
            split_start = clock_pair()
            if name == "stored_anchor":
                pred[split] = predict_stored_native(X[splits[split]], routes[split], trees, base_score)
            else:
                pred[split] = predict_frozen(X[splits[split]], routes[split], fitted, base_score)
            split_timing[split] = elapsed(split_start)
        prediction_timing = elapsed(start)
        predictions[name] = pred
        fitted_models[name] = fitted
        metrics[name] = {"rmse": {s: float(np.sqrt(np.mean((pred[s] - y[splits[s]]) ** 2))) for s in SPLITS},
                         "fit_timing": fit_timing, "prediction_timing": split_timing,
                         "prediction_timing_all_splits": prediction_timing,
                         "counts": parameter_counts(fitted)}
    return {"metrics": metrics, "leaf_models": fitted_models}, predictions


def import_controlled_xgb(root):
    package = (root / "python-package") if (root / "python-package" / "xgboost").is_dir() else root
    if not (package / "xgboost").is_dir():
        raise ValueError("--root must contain python-package/xgboost or xgboost")
    sys.path.insert(0, str(package.resolve()))
    import xgboost as xgb
    if not Path(xgb.__file__).resolve().is_relative_to(package.resolve()):
        raise RuntimeError("Imported unintended xgboost package")
    return xgb


def load_saved_metadata(model_dir):
    """Support worker metadata and legacy archives without inventing defaults."""
    result = json.loads((model_dir / "result.json").read_text())
    sources = ("effective_params", "estimator_params", "params")
    selected = {}
    values = {}
    for field, aliases in (("ridge", ("metagb_lambda_l2",)), ("eta", ("learning_rate", "eta"))):
        for source in sources:
            params = result.get(source, {})
            if not isinstance(params, dict):
                continue
            found = next((key for key in aliases if params.get(key) is not None), None)
            if found is not None:
                values[field] = scalar(params[found])
                selected[field] = source + "." + found
                break
        else:
            raise ValueError(f"Saved result metadata missing {aliases}; checked {sources}")
    model_path = next((model_dir / name for name in ("model.ubj", "model.json") if (model_dir / name).is_file()), None)
    if model_path is None:
        raise FileNotFoundError("Saved model directory needs model.ubj or model.json")
    return result, values["ridge"], values["eta"], model_path, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "data", "model-dir", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    result, ridge, eta, model_path, parameter_sources = load_saved_metadata(args.model_dir)
    xgb = import_controlled_xgb(args.root)
    booster = xgb.Booster(model_file=str(model_path))
    learner = json.loads(booster.save_raw(raw_format="json"))["learner"]
    if learner["objective"]["name"] != "reg:squarederror":
        raise ValueError("Only reg:squarederror models are supported")
    if learner["gradient_booster"]["name"] != "gbtree":
        raise ValueError("Only native gbtree models are supported")
    trees = learner["gradient_booster"]["model"]["trees"]
    nfeature = int(learner["learner_model_param"]["num_feature"])
    base = scalar(learner["learner_model_param"]["base_score"])
    if int(learner["learner_model_param"].get("num_target", "1")) != 1:
        raise ValueError("Only single-target models are supported")
    with np.load(args.data, allow_pickle=False) as arrays:
        X, y, splits = validate_arrays(arrays["X"], arrays["y"], {s: arrays[s] for s in SPLITS})
    if X.shape[1] != nfeature:
        raise ValueError("NPZ feature count does not match saved model")
    start = clock_pair()
    routes, native = {}, {}
    for split in SPLITS:
        data = xgb.DMatrix(X[splits[split]].astype(np.float32))
        routes[split] = crossgb_leaf_routes(X[splits[split]], trees)
        native[split] = np.asarray(booster.predict(data), dtype=np.float64).reshape(-1)
    native_timing = elapsed(start)
    # Replay before costly refits. Fail closed if stored-leaf semantics differ.
    anchor = anchor_entries(trees, nfeature)
    differences = {s: float(np.max(np.abs(predict_stored_native(X[splits[s]], routes[s], trees, base) - native[s]))) for s in SPLITS}
    float64_differences = {s: float(np.max(np.abs(predict_frozen(X[splits[s]], routes[s], anchor, base) - native[s]))) for s in SPLITS}
    if max(differences.values()) > 1e-5:
        raise RuntimeError(f"Stored anchor replay exceeds absolute tolerance 1e-5: {differences}")
    report, predictions = run_probe(X, y, splits, routes, trees, base, ridge, eta)
    report.update(schema_version=2, status="completed_fixed_route_probe", model_dir=str(args.model_dir.resolve()),
                  model_file=str(model_path.resolve()), parameter_sources=parameter_sources,
                  data=str(args.data.resolve()), controlled_root=str(args.root.resolve()), xgboost_version=xgb.__version__,
                  task_id=result.get("task_id"), trees=len(trees), num_features=nfeature,
                  split_sizes={s: len(splits[s]) for s in SPLITS}, base_score=base, ridge_lambda_l2=ridge,
                  learning_rate=eta, route_frozen=True, route_and_native_anchor_timing=native_timing,
                  replay={"tolerance_abs": 1e-5, "max_abs_difference": differences,
                          "comparison": "float32 stored native replay versus booster.predict; gates probe execution",
                          "float64_max_abs_difference": float64_differences,
                          "float64_comparison": "float64 stored-leaf diagnostic versus booster.predict; does not gate execution"},
                  policy={"routing": "JSON numeric CrossGB scalar predictor semantics: float32 value <= float32 threshold; missing follows default_left; categorical rejected",
                          "fit": "each variant's own pre-tree training residual, squared error; eta applied once",
                          "selection": "training-only greedy penalized SSE; stored feature first; ascending-index tie-break; no HPO",
                          "k": "number of nonbias input features; affine always adds an unpenalized intercept in each refitted predictive leaf",
                          "constant_leaves": "frozen stored constants for every variant",
                          "unoccupied_training_leaves": "retain stored slope; affine adds frozen zero intercept; recorded fallback",
                          "positive_ridge": "saved metagb_lambda_l2 from recorded parameter_sources; no default or retuning",
                          "input_precision": "float32 native inputs; float64 refit arithmetic on the same rounded values",
                          "stored_anchor_metrics": "RMSE and delta baseline use stored native replay: float32 base, leaf products and sequential tree accumulation; no refit",
                          "probe_prediction_precision": "all slope/affine refits and their predictions retain float64 arithmetic; float64 stored replay is diagnostic only",
                          "timing": "fit excludes routing; prediction is replay across all three splits and excludes shared native routing"},
                  limitations=["Fixed k1 routes and stored-first feature selection may favor k1; not a full k>1 algorithm.",
                               "Affine leaf models are a separate approximation and are not the original algorithm.",
                               "Raw-input ridge refits need not reproduce coefficients learned using bin approximations; only the stored anchor must replay.",
                               "Probe fit timings exclude original structure training; native routing is shared and reported separately."])
    for name in VARIANTS:
        report["metrics"][name]["delta_rmse_vs_stored_anchor"] = {s: report["metrics"][name]["rmse"][s] - report["metrics"]["stored_anchor"]["rmse"][s] for s in SPLITS}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "predictions.npz", **{f"{name}__{s}": predictions[name][s] for name in VARIANTS for s in SPLITS},
                        **{f"rows__{s}": splits[s] for s in SPLITS}, **{f"native_anchor__{s}": native[s] for s in SPLITS})
    (args.output / "capacity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
