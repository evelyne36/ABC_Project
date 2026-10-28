#!/usr/bin/env python3
"""One candidate's four inner fits, or one selected outer refit, on Linux."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for variable in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def export_structure(booster, xgb, X, destination):
    import numpy as np
    trees = json.loads(booster.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
    routes = np.asarray(booster.predict(xgb.DMatrix(X), pred_leaf=True)).reshape(len(X), len(trees))
    rows = []
    for index, tree in enumerate(trees):
        paths = {}
        def visit(node, features):
            if tree["left_children"][node] == -1:
                paths[node] = features; return
            following = features | {int(tree["split_indices"][node])}
            visit(tree["left_children"][node], following)
            visit(tree["right_children"][node], following)
        visit(0, set())
        for node, path in paths.items():
            feature = tree.get("metagb_leaf_features", [X.shape[1]] * len(tree["left_children"]))[node]
            mask = routes[:, index] == node
            count = int(mask.sum())
            predictive = 0 <= feature < X.shape[1]
            rows.append({"tree": index, "leaf": node, "train_rows": count,
                "predictive_feature": feature if predictive else "bias",
                "path_features": json.dumps(sorted(path)), "outside_path": predictive and feature not in path,
                "stored_shrunk_leaf_weight": float(tree["split_conditions"][node]),
                "train_feature_std": float(X[mask, feature].std()) if predictive and count else 0.0})
    with (destination / "leaf_structure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    predictive_rows = [r for r in rows if r["predictive_feature"] != "bias"]
    return {"trees": len(trees), "leaves": len(rows), "predictive_leaves": len(predictive_rows),
        "outside_path_predictive_leaves": sum(r["outside_path"] for r in predictive_rows),
        "outside_path_fraction_among_predictive_leaves": sum(r["outside_path"] for r in predictive_rows) / len(predictive_rows) if predictive_rows else None,
        "sample_weighted_outside_path_fraction_among_predictive_leaves": sum(r["train_rows"] for r in predictive_rows if r["outside_path"]) / sum(r["train_rows"] for r in predictive_rows) if predictive_rows and sum(r["train_rows"] for r in predictive_rows) else None,
        "scope": "descriptive training-route statistics; not a causal attribution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["cpu_activity", "kin8nm", "energy_efficiency"])
    parser.add_argument("--method", choices=["xgb_cpu", "current_metagb"], required=True)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--params-file", required=True, type=Path)
    parser.add_argument("--xgboost-python-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["candidate", "final"], required=True)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux": raise RuntimeError("Formal CGB/XGB fits are reserved for Linux")
    import numpy as np
    sys.path.insert(0, str(ROOT / "rebuttal_baselines_20261025"))
    sys.path.insert(0, str(ROOT / "rebuttal_execution_20261025/source/exp"))
    from baselines.protocol import load_protocol
    from metagb_ablation.modeling import runtime_args
    from random300.utils.runner import build_estimator, import_local_xgboost
    from run_cgb_task import current_source_hashes, helper_hashes
    directory = ROOT / "rebuttal_baselines_20261025/inputs" / args.dataset
    X, y, folds, manifest = load_protocol(directory)
    fold = next(f for f in folds if f["outer_fold"] == args.outer_fold)
    params = json.loads(args.params_file.read_text())
    xgb = import_local_xgboost(args.xgboost_python_package)
    assert Path(xgb.__file__).resolve().is_relative_to(args.xgboost_python_package.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_args(model_seed=42, n_jobs=1, categorical_handling="onehot")
    evaluation = fold["inner"] if args.mode == "candidate" else [{"train": fold["train"], "validation": fold["test"]}]
    metrics, costs = [], []
    for split in evaluation:
        train, validation = split["train"], split["validation"]
        assert not np.intersect1d(train, validation).size
        model = build_estimator(xgb, args.method, params, runtime, "regression", 0)
        (args.output / "progress.json").write_text(json.dumps({"fits_started": len(metrics) + 1, "fits_completed": len(metrics)}))
        start, cpu = time.perf_counter(), time.process_time()
        model.fit(X[train], y[train], verbose=False)
        fit, cpu_fit = time.perf_counter() - start, time.process_time() - cpu
        prediction = model.predict(X[validation])
        metrics.append(float(np.sqrt(np.mean((prediction - y[validation]) ** 2))))
        costs.append({"fit_wall_sec": fit, "fit_cpu_sec": cpu_fit})
        (args.output / "progress.json").write_text(json.dumps({"fits_started": len(metrics), "fits_completed": len(metrics)}))
    report = {"status": "success", "mode": args.mode, "dataset": args.dataset, "method": args.method,
        "outer_fold": args.outer_fold, "params": params, "params_sha256": digest(args.params_file),
        "rmse": float(np.mean(metrics)), "fold_metrics": metrics, "costs": costs,
        "model_fit_calls": len(evaluation), "crossgb_fit_calls": len(evaluation) if args.method == "current_metagb" else 0,
        "input_sha256": manifest["data_npz_sha256"], "native_library_sha256": digest(Path(xgb.core._LIB._name)),
        "controlled_source": current_source_hashes(),
        "helper_hashes": helper_hashes(ROOT / "rebuttal_execution_20261025/source/exp"),
        "driver_sha256": digest(Path(__file__)),
        "evaluation_scope": "four inner validation folds only; no outer test metric" if args.mode == "candidate" else "held-out outer test after frozen selection"}
    if args.mode == "final":
        np.savez_compressed(args.output / "predictions.npz", sample_id=fold["test"], y_true=y[fold["test"]], prediction=prediction)
        model.get_booster().save_model(args.output / "model.ubj")
        report["model_sha256"] = digest(args.output / "model.ubj")
        report["predictions_sha256"] = digest(args.output / "predictions.npz")
        if args.diagnostics:
            report["structure"] = export_structure(model.get_booster(), xgb, X[fold["train"]], args.output)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ["status", "dataset", "method", "outer_fold", "mode", "rmse"]}))


if __name__ == "__main__": main()
