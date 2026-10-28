#!/usr/bin/env python3
"""N5 capacity probe: refit CrossGB leaf coefficients with k = 1, 2, 3 variables.

Scope, per `rebuttal_server_config_20261025/configs/e6_capacity.json`:
  * the tree structure and every routing decision are FROZEN -- this is a
    "fixed-route capacity probe", not a full k>1 CrossGB;
  * coefficients obey `theta = -(Z^T H Z + Lambda)^-1 Z^T g` with H = 1
    (squared error) and Lambda frozen to the model's own `metagb_lambda_l2`;
  * feature subsets are selected on TRAINING rows only (greedy forward, order
    recorded), never on rows that carry outer-test labels;
  * no hyperparameter is retuned here.

Three families of rows are produced on purpose, so the capacity effect is not
mixed with the refit rule's own approximation error:
  * `k1_stored` -- the archive's own single-variable weights (anchor; equals
                   the original model by construction),
  * `k1_rule`   -- the same ridge rule at k = 1 (validation row),
  * `k2_rule`, `k3_rule` -- the same rule with one/two extra variables.

Leaf semantics come from the controlled source (`src/predictor/cpu_predictor.cc`):
a MetaGB leaf contributes `LeafValue * x[MetaGBLeafFeature]`, and a constant
leaf (feature == NumFeatures) contributes `LeafValue`.  Leaf values are stored
in `split_conditions` and already carry the learning rate.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "rebuttal_formal_20261026"


def parse_scalar(value):
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return float(text.split(",")[0])


def ridge_fit(Z, g, ridge):
    """Ridge coefficients theta = -(Z^T Z + Lambda)^-1 Z^T g and the residual SSE."""
    import numpy as np
    gram = Z.T @ Z
    if ridge > 0:
        gram = gram + ridge * np.eye(Z.shape[1])
    theta = -np.linalg.solve(gram, Z.T @ g)
    residual = g + Z @ theta
    return theta, float(residual @ residual)


def greedy_features(X_leaf, g, start, want, ridge, limit):
    """Training-only greedy forward selection, starting from the stored variable."""
    import numpy as np
    chosen = [int(start)]
    _, best = ridge_fit(X_leaf[:, chosen], g, ridge)
    while len(chosen) < want:
        gain, picked = 0.0, None
        for candidate in range(limit):
            if candidate in chosen:
                continue
            trial = chosen + [candidate]
            _, sse = ridge_fit(X_leaf[:, trial], g, ridge)
            if best - sse > gain + 1e-12:
                gain, picked = best - sse, candidate
        if picked is None:
            break
        chosen.append(picked)
        _, best = ridge_fit(X_leaf[:, chosen], g, ridge)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="configs/cgb_tasks/*.json (or absolute)")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with model.ubj and result.json")
    parser.add_argument("--data", type=Path, help="Defaults to FORMAL/task['data_npz']")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--outer-fold", type=int, help="Defaults to the task's outer fold")
    parser.add_argument("--lambda-l2", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--xgboost-python-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.xgboost_python_package.resolve()))
    import numpy as np
    import xgboost as xgb
    if not Path(xgb.__file__).resolve().is_relative_to(args.xgboost_python_package.resolve()):
        raise RuntimeError("Imported an unintended xgboost package")

    task_path = args.task if args.task.is_absolute() else (FORMAL / args.task)
    task = json.loads(task_path.read_text())
    fold = args.outer_fold if args.outer_fold is not None else int(task["outer_fold"])
    data_path = args.data or (FORMAL / task["data_npz"])
    with np.load(data_path, allow_pickle=False) as arrays:
        train = arrays[f"outer{fold}_train"]
        test = arrays[f"outer{fold}_test"]
        X_all = np.asarray(arrays["X"], dtype=np.float64)
        y_all = np.asarray(arrays["y"], dtype=np.float64).ravel()
    X_train, y_train = X_all[train], y_all[train]
    X_test, y_test = X_all[test], y_all[test]

    result_path = args.model_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    params = result.get("params", {})
    ridge = args.lambda_l2 if args.lambda_l2 is not None else float(params.get("metagb_lambda_l2", 0.0))
    eta = args.learning_rate if args.learning_rate is not None else float(params.get("learning_rate", 1.0))

    booster = xgb.Booster(model_file=str(args.model_dir / "model.ubj"))
    raw = json.loads(booster.save_raw(raw_format="json"))["learner"]
    trees = raw["gradient_booster"]["model"]["trees"]
    param = raw["learner_model_param"]
    base_score = parse_scalar(param["base_score"])
    num_feature = int(param["num_feature"])

    tables = []
    for tree in trees:
        left = np.asarray(tree["left_children"], dtype=np.int64)
        value = np.asarray(tree["split_conditions"], dtype=np.float64)
        feature = np.asarray(tree["metagb_leaf_features"], dtype=np.int64)
        is_leaf = left == -1
        predictive = is_leaf & (feature >= 0) & (feature < num_feature)
        constant = is_leaf & ~predictive
        tables.append((value, feature, predictive, constant))

    def leaf_ids(X):
        ids = np.asarray(booster.predict(xgb.DMatrix(X.astype(np.float32)), pred_leaf=True))
        return ids.reshape(len(X), -1) if ids.ndim > 1 else ids.reshape(len(X), 1)

    train_leaves, test_leaves = leaf_ids(X_train), leaf_ids(X_test)
    native_test = np.asarray(booster.predict(xgb.DMatrix(X_test.astype(np.float32))), dtype=np.float64)

    variants = ["k1_stored", "k1_rule"] + [f"k{k}_rule" for k in args.k if k > 1]
    n_train, n_test = len(X_train), len(X_test)
    rows_train, rows_test = np.arange(n_train), np.arange(n_test)
    constant_train = np.zeros(n_train)
    constant_test = np.zeros(n_test)
    predicted = {name: {"train": np.zeros(n_train), "test": np.zeros(n_test)} for name in variants}
    refit_log = {name: [] for name in variants}

    cursor = np.full(n_train, base_score)
    for index, (value, feature, predictive, constant) in enumerate(tables):
        node_train = train_leaves[:, index].astype(np.int64)
        node_test = test_leaves[:, index].astype(np.int64)
        gradient = cursor - y_train
        constant_train += np.where(constant[node_train], value[node_train], 0.0)
        constant_test += np.where(constant[node_test], value[node_test], 0.0)
        f_train = np.clip(feature[node_train], 0, num_feature - 1)
        f_test = np.clip(feature[node_test], 0, num_feature - 1)
        stored_train = np.where(predictive[node_train], value[node_train] * X_train[rows_train, f_train], 0.0)
        stored_test = np.where(predictive[node_test], value[node_test] * X_test[rows_test, f_test], 0.0)
        predicted["k1_stored"]["train"] += stored_train
        predicted["k1_stored"]["test"] += stored_test

        for leaf in np.unique(node_train):
            leaf = int(leaf)
            if not predictive[leaf]:
                continue
            mask = node_train == leaf
            mask_test = node_test == leaf
            if mask.sum() < 2 or not mask_test.any():
                # Too few training rows to refit: every variant keeps the stored weight.
                for name in variants[1:]:
                    predicted[name]["train"][mask] += value[leaf] * X_train[mask, feature[leaf]]
                    predicted[name]["test"][mask_test] += value[leaf] * X_test[mask_test, feature[leaf]]
                continue
            g = gradient[mask]
            for name in variants[1:]:
                want = 1 if name == "k1_rule" else int(name[1])
                chosen = greedy_features(X_train[mask], g, int(feature[leaf]), want, ridge, num_feature)
                theta, _ = ridge_fit(X_train[mask][:, chosen], g, ridge)
                predicted[name]["train"][mask] += eta * (X_train[mask][:, chosen] @ theta)
                predicted[name]["test"][mask_test] += eta * (X_test[mask_test][:, chosen] @ theta)
                refit_log[name].append({"tree": index, "leaf": leaf, "variables": chosen, "k": len(chosen)})
        cursor = cursor + np.where(predictive[node_train],
                                   value[node_train] * X_train[rows_train, f_train],
                                   np.where(constant[node_train], value[node_train], 0.0))

    report = {
        "status": "capacity_probe",
        "task_id": result.get("task_id", task.get("task_id")),
        "variant": result.get("variant"),
        "outer_fold": fold,
        "trees": len(trees),
        "num_feature": num_feature,
        "base_score": base_score,
        "ridge_lambda_l2": ridge,
        "learning_rate": eta,
        "route_frozen": True,
        "selection": "training-only greedy forward selection starting at the stored variable; order recorded per leaf",
        "replay_check": {"max_abs_test_difference": float(np.max(np.abs(
            (base_score + constant_test + predicted["k1_stored"]["test"]) - native_test)))},
        "metrics": {},
    }
    for name in variants:
        train_pred = base_score + constant_train + predicted[name]["train"]
        test_pred = base_score + constant_test + predicted[name]["test"]
        entries = refit_log.get(name, [])
        report["metrics"][name] = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
            "test_rmse": float(np.sqrt(np.mean((test_pred - y_test) ** 2))),
            "mean_variables_per_leaf": float(np.mean([e["k"] for e in entries])) if entries else 1.0,
            "leaves_refit": len(entries),
        }
    anchor = report["metrics"]["k1_stored"]["test_rmse"]
    for name in report["metrics"]:
        report["metrics"][name]["delta_test_vs_k1_stored"] = report["metrics"][name]["test_rmse"] - anchor
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "capacity.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
