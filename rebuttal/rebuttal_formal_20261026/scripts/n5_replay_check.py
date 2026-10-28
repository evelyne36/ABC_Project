#!/usr/bin/env python3
"""Replay a fitted CrossGB model leaf by leaf and verify the k=1 coefficient identity.

The N5 capacity probe replaces each CrossGB leaf output `v * x[f]` with an
affine model on k variables.  Before any coefficient is refitted on the
archive, two things have to be demonstrated:

  1. **leaf replay** reproduces the model's own prediction, i.e. we know
     exactly which leaf value, which variable and which scaling the stored
     model uses (`src/predictor/cpu_predictor.cc`: a MetaGB leaf contributes
     `LeafValue * x[MetaGBLeafFeature]`, and a constant leaf whose feature
     equals NumFeatures contributes `LeafValue`);
  2. **the coefficient rule** `theta = -(Z^T H Z + Lambda)^-1 Z^T g` applied at
     k=1 reproduces the stored single-variable weight, so the same rule can be
     trusted at k=2/3.

Nothing is fitted here: this script is read-only over an existing `model.ubj`
and its frozen split indices.  Its report is the acceptance gate for the
capacity probe.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")


def parse_scalar(value):
    """XGBoost stores some parameters as vector strings such as "[0E0]"."""
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return float(text.split(",")[0])


def find_key(node, wanted):
    """Depth-first search for every value stored under `wanted`."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == wanted and not isinstance(value, (dict, list)):
                found.append(value)
            found.extend(find_key(value, wanted))
    elif isinstance(node, list):
        for value in node:
            found.extend(find_key(value, wanted))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model.ubj written by run_cgb_task.py")
    parser.add_argument("--data", type=Path, required=True, help="NPZ with X, y and outer{fold}_train/test indices")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--xgboost-python-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-fallback", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override the learning rate; the stored leaf values already carry it")
    parser.add_argument("--lambda-l2", type=float, default=None,
                        help="Override metagb_lambda_l2 (the leaf ridge penalty used at grow time)")
    args = parser.parse_args()

    sys.path.insert(0, str(args.xgboost_python_package.resolve()))
    import numpy as np
    import xgboost as xgb
    if not Path(xgb.__file__).resolve().is_relative_to(args.xgboost_python_package.resolve()):
        raise RuntimeError("Imported an unintended xgboost package")

    booster = xgb.Booster(model_file=str(args.model))
    raw = json.loads(booster.save_raw(raw_format="json"))
    learner = raw["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    param = learner["learner_model_param"]
    base_score = parse_scalar(param["base_score"])
    num_feature = int(param["num_feature"])
    learning_rate = find_key(learner, "learning_rate")
    eta = args.learning_rate if args.learning_rate is not None else (parse_scalar(learning_rate[0]) if learning_rate else 1.0)
    lambdas = find_key(json.loads(booster.save_config()), "metagb_lambda_l2")
    if args.lambda_l2 is not None:
        ridge = args.lambda_l2
    else:
        ridge = parse_scalar(lambdas[0]) if lambdas else args.lambda_fallback

    with np.load(args.data, allow_pickle=False) as arrays:
        rows = arrays[f"outer{args.outer_fold}_{args.split}"]
        X = np.asarray(arrays["X"][rows], dtype=np.float64)
        y = np.asarray(arrays["y"][rows], dtype=np.float64).ravel()

    dmatrix = xgb.DMatrix(X.astype(np.float32))
    native = np.asarray(booster.predict(dmatrix), dtype=np.float64)
    leaf_ids = np.asarray(booster.predict(dmatrix, pred_leaf=True))
    if leaf_ids.ndim == 1:
        leaf_ids = leaf_ids.reshape(-1, 1)
    if leaf_ids.shape[1] != len(trees):
        raise RuntimeError(f"pred_leaf returned {leaf_ids.shape[1]} trees, model has {len(trees)}")

    leaves = []
    for tree in trees:
        left = tree["left_children"]
        leaves.append([node for node, child in enumerate(left) if child == -1])

    def replay(scaling):
        pred = np.full(len(X), base_score, dtype=np.float64)
        for index, tree in enumerate(trees):
            weight = np.asarray(tree["split_conditions"], dtype=np.float64)
            feature = np.asarray(tree["metagb_leaf_features"], dtype=np.int64)
            node = leaf_ids[:, index].astype(np.int64)
            value = weight[node]
            feat = feature[node]
            contribution = np.where(feat < num_feature,
                                    value * X[np.arange(len(X)), np.clip(feat, 0, num_feature - 1)],
                                    value)
            pred += scaling * contribution
        return pred

    replay_report = {}
    for name, scaling in [("no_extra_scaling", 1.0), ("learning_rate", eta)]:
        difference = replay(scaling) - native
        replay_report[name] = {
            "scaling": scaling,
            "max_abs_difference": float(np.max(np.abs(difference))) if len(difference) else 0.0,
            "mean_abs_difference": float(np.mean(np.abs(difference))) if len(difference) else 0.0,
        }
    accepted = min(replay_report, key=lambda k: replay_report[k]["max_abs_difference"])
    scaling = replay_report[accepted]["scaling"]

    # k=1 coefficient identity: for every predictive leaf, re-solve the ridge
    # weights from the residual gradient before that tree and compare with the
    # stored value.  h = 1 for squared error.
    rel_errors = []
    per_tree = []
    cursor = np.full(len(X), base_score, dtype=np.float64)
    for index, tree in enumerate(trees):
        weight = np.asarray(tree["split_conditions"], dtype=np.float64)
        feature = np.asarray(tree["metagb_leaf_features"], dtype=np.int64)
        node = leaf_ids[:, index].astype(np.int64)
        gradient = cursor - y
        tree_errors = []
        predictive = 0
        for leaf in leaves[index]:
            rows_here = node == leaf
            count = int(rows_here.sum())
            feat = int(feature[leaf])
            if count == 0 or feat >= num_feature:
                continue
            predictive += 1
            x = X[rows_here, feat]
            g = gradient[rows_here]
            denominator = float(np.sum(x * x)) + ridge
            if denominator <= 0.0:
                continue
            solved = eta * (-float(np.sum(g * x)) / denominator)
            stored = float(weight[leaf])
            error = abs(solved - stored) / max(1e-9, abs(stored))
            tree_errors.append(error)
            rel_errors.append(error)
        value = weight[node]
        feat = feature[node]
        contribution = np.where(feat < num_feature,
                                value * X[np.arange(len(X)), np.clip(feat, 0, num_feature - 1)],
                                value)
        cursor = cursor + scaling * contribution
        per_tree.append({"tree": index, "predictive_leaves": predictive,
                         "max_relative_error": max(tree_errors) if tree_errors else None,
                         "median_relative_error": float(np.median(tree_errors)) if tree_errors else None})

    feature_ids = sorted({int(f) for tree in trees for f in np.asarray(tree["metagb_leaf_features"], dtype=np.int64)
                          if 0 <= int(f) < num_feature})
    report = {
        "status": "replay_checked",
        "model": str(args.model),
        "split": args.split,
        "outer_fold": args.outer_fold,
        "rows": int(len(X)),
        "trees": len(trees),
        "num_feature": num_feature,
        "base_score": base_score,
        "learning_rate_detected": eta,
        "metagb_lambda_l2": ridge,
        "leaves_total": int(sum(len(v) for v in leaves)),
        "leaves_predictive": int(sum(1 for tree in trees for node, child in enumerate(tree["left_children"])
                                     if child == -1 and 0 <= int(tree["metagb_leaf_features"][node]) < num_feature)),
        "features_used": feature_ids,
        "replay": replay_report,
        "replay_convention": accepted,
        "coefficient_identity_k1": {
            "rule": "theta = -sum(g*x) / (sum(x*x) + lambda), h=1 (squared error)",
            "leaves_checked": len(rel_errors),
            "max_relative_error": float(np.max(rel_errors)) if rel_errors else None,
            "median_relative_error": float(np.median(rel_errors)) if rel_errors else None,
            "per_tree": per_tree,
        },
        "verdict": {
            "replay_exact": replay_report[accepted]["max_abs_difference"] <= 1e-5,
            "k1_identity_holds": (bool(rel_errors) and float(np.max(rel_errors)) <= 1e-3),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ["trees", "replay_convention", "verdict"]}, indent=2))
    print(json.dumps({k: v for k, v in report["coefficient_identity_k1"].items() if k != "per_tree"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
