#!/usr/bin/env python3
"""Tiny Linux-only structural checks, separate from formal benchmark results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for key in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[key] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgboost-python-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux": raise RuntimeError("Native CGB checks belong on Linux")
    sys.path.insert(0, str(args.xgboost_python_package.resolve()))
    import numpy as np
    import xgboost as xgb
    assert Path(xgb.__file__).resolve().is_relative_to(args.xgboost_python_package.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(407)
    X = np.column_stack([rng.uniform(1, 3, 128), rng.uniform(-1, 1, 128)]).astype("float32")
    y = (3 * X[:, 0] + 0.1 * X[:, 1]).astype("float32")
    data = xgb.DMatrix(X, label=y)
    params = dict(objective="reg:squarederror", base_score=0, tree_method="hist", device="cpu", nthread=1,
        max_depth=2, max_bin=32, learning_rate=.4, subsample=1, colsample_bytree=1, enable_metagb=True,
        metagb_pred_colsample_bytree=1, metagb_score_agg=0, metagb_lambda_l2=.5,
        metagb_min_data_in_leaf=1, metagb_cond_pred_share=False, seed=271828)
    implicit = xgb.train(params, data, num_boost_round=2)
    explicit = xgb.train({**params, "metagb_split_only_use_bias": False}, data, num_boost_round=2)
    np.testing.assert_array_equal(implicit.predict(data), explicit.predict(data))
    leaf = xgb.train({**params, "metagb_split_only_use_bias": True}, data, num_boost_round=1)
    bias = xgb.train({**params, "metagb_split_only_use_bias": True, "metagb_leaf_only_use_bias": True}, data, num_boost_round=1)
    np.testing.assert_array_equal(leaf.predict(data, pred_leaf=True), bias.predict(data, pred_leaf=True))
    tree = json.loads(leaf.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    leaves = [i for i, child in enumerate(tree["left_children"]) if child == -1]
    assert any(tree["metagb_leaf_features"][i] < X.shape[1] for i in leaves), "No predictive leaf was fitted"
    assert not np.allclose(leaf.predict(data), bias.predict(data)), "Predictive leaves collapsed to the constant control"
    leaf.save_model(args.output / "tiny_leafonly.ubj")
    restored = xgb.Booster(model_file=args.output / "tiny_leafonly.ubj")
    np.testing.assert_array_equal(restored.predict(data), leaf.predict(data))
    report = {"passed": True, "native_library_sha256": hashlib.sha256(Path(xgb.core._LIB._name).read_bytes()).hexdigest(),
        "checks": ["default false preserves predictions", "first-tree bias-score routes agree with constant-leaf control",
            "predictive leaf selected and changes predictions", "model save/load preserves predictions"],
        "crossgb_fit_calls": 4, "scope": "tiny integration only; not formal evidence or an exhaustive correctness proof"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
