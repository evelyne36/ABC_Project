#!/usr/bin/env python3
"""One frozen-backbone scalar fit per Linux process; no hyperparameter search."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for variable in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "rebuttal_formal_20261026"
ENDPOINT = {"xgb": "xgb", "bias_p00_sa0": "bias_p00_sa0", "split_p10_sa0": "split_p10_sa0", "full_p10_sa0": "full_p10_sa0", "leafonly": "full_p10_sa0"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--variant", choices=list(ENDPOINT), required=True)
    parser.add_argument("--xgboost-python-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("CGB fits are reserved for the authorized Linux server")
    import numpy as np
    from sklearn.metrics import log_loss
    source = ROOT / "rebuttal_execution_20261025/source/exp"
    sys.path.insert(0, str(source))
    from metagb_ablation.modeling import build_variant_estimator, runtime_args
    from metagb_ablation.protocols import build_variants
    from metagb_ablation.io_utils import canonical_hash, index_hash
    from random300.utils.runner import import_local_xgboost
    task = json.loads(args.task.read_text())
    data = FORMAL / task["data_npz"]
    assert digest(data) == task["data_npz_sha256"]
    assert canonical_hash(task["backbone"]) == task["backbone_hash"]
    xgb = import_local_xgboost(args.xgboost_python_package)
    module = Path(xgb.__file__).resolve()
    if not module.is_relative_to(args.xgboost_python_package.resolve()):
        raise RuntimeError("Imported an unintended xgboost package")
    library = Path(xgb.core._LIB._name).resolve()
    library_hash = digest(library)
    if args.variant == "leafonly":
        if args.gate is None:
            parser.error("LeafOnly requires the completed 28-fit R0/E0 gate")
        gate = json.loads(args.gate.read_text())
        if not gate.get("passed") or gate.get("native_library_sha256") != library_hash:
            raise RuntimeError("R0/E0 did not pass for this exact native library")
        if gate.get("controlled_source") != current_source_hashes():
            raise RuntimeError("Controlled source changed after the gate")
        if gate.get("driver_sha256") != digest(Path(__file__)):
            raise RuntimeError("Driver changed after the gate")
        if gate.get("helper_hashes") != helper_hashes(source):
            raise RuntimeError("Model construction helpers changed after the gate")
    variant = next(v for v in build_variants() if v.variant_id == ENDPOINT[args.variant])
    model_args = runtime_args(model_seed=task["model_seed"], n_jobs=1, categorical_handling="ordinal")
    model, params = build_variant_estimator(xgb, variant, task["backbone"], model_args, task["task_type"], task["n_classes"])
    # Do not change the legacy default-off path when checking archived endpoints.
    if args.variant == "leafonly":
        model.set_params(metagb_split_only_use_bias=True)
        params["metagb_split_only_use_bias"] = True
    with np.load(data, allow_pickle=False) as arrays:
        train = arrays[f"outer{task['outer_fold']}_train"]
        test = arrays[f"outer{task['outer_fold']}_test"]
        assert index_hash(train) == task["archived_split"]["outer_train_hash"]
        assert index_hash(test) == task["archived_split"]["outer_test_hash"]
        X, y = arrays["X"], arrays["y"]
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    args.output.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(X_train, y_train, verbose=False)
    fit_wall, fit_cpu = time.perf_counter() - wall, time.process_time() - cpu
    booster = model.get_booster()
    config = booster.save_config()
    if args.variant == "leafonly":
        entries = []
        def visit(item):
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == "metagb_split_only_use_bias": entries.append(value)
                    visit(value)
            elif isinstance(item, list):
                for value in item: visit(value)
        visit(json.loads(config))
        if not entries or not all(str(v).lower() in {"1", "true"} for v in entries):
            raise RuntimeError("Native configuration did not accept LeafOnly; refuse to report a mislabeled Full run")
    prediction = model.predict(X_test)
    saved = {"sample_id": test, "y_true": y_test, "prediction": prediction}
    if task["task_type"] == "regression":
        metric = float(np.sqrt(np.mean((prediction - y_test) ** 2)))
        compare = prediction
    else:
        probability = model.predict_proba(X_test)
        saved["probability"] = probability
        metric = float(log_loss(y_test, probability, labels=np.arange(task["n_classes"])))
        compare = probability
    np.savez_compressed(args.output / "predictions.npz", **saved)
    booster.save_model(args.output / "model.ubj")
    (args.output / "native_config.json").write_text(config)
    report = {"status": "success", "task_id": task["task_id"], "variant": args.variant,
        "task_file_sha256": digest(args.task), "input_sha256": task["data_npz_sha256"],
        "metric_name": task["primary_metric"], "metric": metric, "fit_wall_sec": fit_wall, "fit_cpu_sec": fit_cpu,
        "native_library_sha256": library_hash, "native_library": str(library), "python_package": str(module),
        "controlled_source": current_source_hashes(), "helper_hashes": helper_hashes(source),
        "driver_sha256": digest(Path(__file__)), "params": params, "runtime_params": json_safe(model.get_params()),
        "crossgb_fit_calls": int(args.variant != "xgb"), "model_fit_calls": 1,
        "prediction_sha256": digest(args.output / "predictions.npz"), "model_sha256": digest(args.output / "model.ubj")}
    if args.variant != "leafonly":
        expected = task["historical_predictions"][args.variant]
        old = np.asarray(expected["probability"] if task["task_type"] != "regression" else expected["prediction"])
        report.update(archive_max_abs_prediction_difference=float(np.max(np.abs(compare - old))),
            archive_metric_difference=metric - task["historical_metrics"][args.variant],
            archive_prediction_pass=bool(np.allclose(compare, old, rtol=1e-6, atol=1e-6)),
            archive_prediction_tolerance={"rtol": 1e-6, "atol": 1e-6})
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ["status", "task_id", "variant", "metric", "fit_wall_sec"]}))


def current_source_hashes():
    directory = ROOT / "rebuttal_execution_20261025/native/controlled/src/tree"
    return {name: digest(directory / name) for name in ["param.h", "updater_metagb_hist.cc"]}


def helper_hashes(source):
    return {str(p.relative_to(source)): digest(p) for p in [source / "metagb_ablation/modeling.py", source / "metagb_ablation/protocols.py", source / "random300/utils/runner.py"]}


def json_safe(value):
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value


if __name__ == "__main__":
    main()
