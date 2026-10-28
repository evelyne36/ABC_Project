#!/usr/bin/env python3
"""One real model fit in one process; train labels never include evaluation data."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

# Set before importing NumPy/Numba/native runtimes. Parallelism is across jobs.
for variable in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from baselines import make_regressor
from baselines.provenance import source_record,adapter_fingerprint


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["linxgboost", "pilot", "gbdt_pl", "tao"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPZ: X_train, y_train, X_predict; optional y_evaluate")
    parser.add_argument("--output", type=Path, required=True, help="A new result directory; existing directories are protected")
    parameters = parser.add_mutually_exclusive_group()
    parameters.add_argument("--params", default="{}", help="JSON dictionary using upstream parameter names")
    parameters.add_argument("--params-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true", help="Fit StandardScaler on X_train only")
    args = parser.parse_args()
    params = json.loads(args.params_file.read_text() if args.params_file else args.params)
    if not isinstance(params, dict):
        parser.error("Parameters must be a JSON object")
    with np.load(args.input, allow_pickle=False) as data:
        X = data["X_train"].copy()
        y = data["y_train"].copy()
        predict_X = data["X_predict"].copy()
        evaluate_y = data["y_evaluate"].copy() if "y_evaluate" in data.files else None
    if evaluate_y is not None and (evaluate_y.ndim != 1 or len(evaluate_y) != len(predict_X) or not np.isfinite(evaluate_y).all()):
        raise ValueError("y_evaluate must be finite, one-dimensional and match X_predict")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"method":args.method, "status":"started", "seed":args.seed,
              "input_sha256":sha256(args.input), "train_rows":len(X), "predict_rows":len(predict_X),
              "feature_count":X.shape[1], "requested_params":params, "model_fit_calls":1, "crossgb_fit_calls":0,
              "fit_timing_scope":"includes first-use author import/JIT; do not interpret as warmed kernel timing",
              "environment":{"python":platform.python_version(), "system":platform.system(), "machine":platform.machine(),
                             "versions":{n:importlib.metadata.version(n) for n in ["numpy","scipy","pandas","scikit-learn","numba"]}}}
    report["source"] = source_record(args.method)
    report["adapter_fingerprint"] = adapter_fingerprint()
    try:
        model = make_regressor(args.method,params,seed=args.seed,standardize=args.standardize)
        wall, cpu = time.perf_counter(), time.process_time()
        model.fit(X,y)
        report["fit_wall_sec"] = time.perf_counter()-wall
        report["fit_process_cpu_sec"] = time.process_time()-cpu
        wall, cpu = time.perf_counter(), time.process_time()
        prediction = model.predict(predict_X)
        report["predict_wall_sec"] = time.perf_counter()-wall
        report["predict_process_cpu_sec"] = time.process_time()-cpu
        report["metadata"] = model.metadata_
        if args.method == "gbdt_pl":
            report["native_library_sha256"] = sha256(model.model_.library_path)
        if model.scaler_ is not None:
            report["scaler_train_mean"] = model.scaler_.mean_.tolist()
            report["scaler_train_scale"] = model.scaler_.scale_.tolist()
        if evaluate_y is not None:
            report["rmse"] = float(np.sqrt(np.mean((prediction-evaluate_y)**2)))
            report["evaluation_scope"] = "evaluate-only labels; never passed to model.fit"
        if hasattr(model,"training_log_"):
            (args.output/"author_training.log").write_text(model.training_log_)
        np.savez_compressed(args.output/"predictions.npz",prediction=prediction)
        report["prediction_sha256"] = sha256(args.output/"predictions.npz")
        report["status"] = "success"
        (args.output/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:report[k] for k in ["method","status","fit_wall_sec","predict_wall_sec"]},indent=2))
        return 0
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        (args.output/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        (args.output/"error.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
