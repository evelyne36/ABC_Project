#!/usr/bin/env python3
"""One fit per process, with sealed test evaluation and auditable timing.

NPZ: X (already preprocessed), y, integer train/val/test row indices, and optional
scalar task_type. Alternatively y_train/y_val/y_test avoid loading a combined y
array. Search never indexes test targets or predicts test features. No early
stopping, feature transformation, or parameter selection occurs in this worker.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

import numpy as np

METHODS = ("CrossGB", "XGB", "LGB_constant", "LGB_linear", "PL", "LinXGBoost", "LeafOnly")


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def metadata_json(value):
    """Encode metadata sentinels without changing estimator kwargs or metrics."""
    if isinstance(value, np.ndarray):
        return metadata_json(value.tolist())
    if isinstance(value, np.generic):
        return metadata_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: metadata_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [metadata_json(item) for item in value]
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def locate(root, relative):
    for candidate in (root / relative, root / "rebuttal" / relative):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing {relative} under explicit root {root}")


def load_data(path, evaluate_test=False):
    """Read only train/val targets unless final evaluation is explicitly enabled."""
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        if X.ndim != 2 or X.shape[1] < 1 or not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must be a numeric two-dimensional matrix with features")
        indices = {}
        for split in ("train", "val", "test"):
            raw = data[split]
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{split} must contain one-dimensional integer indices")
            idx = raw.astype(np.int64)
            if len(idx) == 0 or idx.min() < 0 or idx.max() >= len(X):
                raise ValueError(f"{split} indices empty or outside X")
            if len(np.unique(idx)) != len(idx):
                raise ValueError(f"{split} contains repeated rows")
            indices[split] = idx
        combined = np.concatenate(list(indices.values()))
        if len(np.unique(combined)) != len(combined):
            raise ValueError("train/val/test indices overlap")
        task = data["task_type"].item() if "task_type" in data else "regression"
        task = task.decode("utf-8") if isinstance(task, bytes) else str(task)
        if task not in ("regression", "binary", "multiclass"):
            raise ValueError(f"Unsupported task_type: {task}")
        splits = ("train", "val", "test") if evaluate_test else ("train", "val")
        targets = {}
        if "y_train" in data and "y_val" in data:
            for split in splits:
                targets[split] = data["y_" + split]
        else:
            # A combined NPZ member must be decompressed as a whole. Test target
            # positions are never indexed, validated, hashed, or used in search.
            y = data["y"]
            if y.ndim != 1 or len(y) != len(X):
                raise ValueError("y must have one target per X row")
            for split in splits:
                targets[split] = y[indices[split]]
        for split, target in targets.items():
            if target.ndim != 1 or len(target) != len(indices[split]):
                raise ValueError(f"Invalid {split} target dimensions")
            if not np.issubdtype(target.dtype, np.number) or not np.isfinite(target).all():
                raise ValueError(f"{split} targets must be finite numeric values")
        if task != "regression":
            classes = np.unique(targets["train"])
            if not np.array_equal(classes, np.arange(len(classes))):
                raise ValueError("Classification targets must be encoded 0..C-1 on training rows")
            if (task == "binary" and len(classes) != 2) or len(classes) < 2:
                raise ValueError("task_type and training class count disagree")
            for split, target in targets.items():
                if not np.isin(target, classes).all():
                    raise ValueError(f"Unknown class in {split}")
        return X, targets, indices, task


def make_model(root, method, task, params, seed):
    p = dict(params)
    lineage = {"python": sys.version, "platform": platform.platform()}
    if method in ("PL", "LinXGBoost"):
        if task != "regression":
            raise ValueError(f"{method} adapter supports regression only")
        base = locate(root, "rebuttal_baselines_20261025")
        sys.path.insert(0, str(base))
        from baselines.estimators import AuthorRegressor
        from baselines.provenance import source_record, adapter_fingerprint
        native_method = "gbdt_pl" if method == "PL" else "linxgboost"
        lineage.update(author_source=source_record(native_method),
                       adapter_sha256=adapter_fingerprint(), package_path=str(base))
        if method == "PL":
            p["num_threads"] = 1
        else:
            p["random_state"] = seed
        return AuthorRegressor(native_method, p, seed, False), lineage, p
    depth = p.pop("depth", 3)
    ridge = p.pop("ridge", 1.0)
    trees = p.pop("trees", 100)
    pred_fraction = p.pop("p", 1.0)
    if method in ("CrossGB", "XGB", "LeafOnly"):
        package = locate(root, "rebuttal_execution_20261025/native/controlled/python-package")
        sys.path.insert(0, str(package))
        import xgboost as xgb
        if not Path(xgb.__file__).resolve().is_relative_to(package):
            raise RuntimeError("XGBoost was imported outside the controlled package")
        library = Path(xgb.core._LIB._name).resolve()
        lineage.update(package_path=str(Path(xgb.__file__).resolve()), version=xgb.__version__,
                       native_library=str(library), native_sha256=sha256(library))
        kw = dict(n_estimators=trees, max_depth=depth, reg_lambda=ridge, max_bin=63,
                  learning_rate=0.1, tree_method="hist", verbosity=0, subsample=1.,
                  colsample_bytree=1.)
        if method in ("CrossGB", "LeafOnly"):
            kw.update(metagb_pred_colsample_bytree=pred_fraction, metagb_lambda_l2=ridge,
                      metagb_score_agg=0, metagb_min_data_in_leaf=1, metagb_cond_pred_share=False,
                      metagb_leaf_only_use_bias=False, metagb_hist_subtract=False)
        kw.update(p)
        kw.update(n_jobs=1, nthread=1, random_state=seed, enable_metagb=method != "XGB")
        if method in ("CrossGB", "LeafOnly"):
            kw["metagb_split_only_use_bias"] = method == "LeafOnly"
        klass = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    else:
        import lightgbm as lgb
        library = Path(lgb.basic._LIB._name).resolve()
        lineage.update(package_path=str(Path(lgb.__file__).resolve()), version=lgb.__version__,
                       native_library=str(library), native_sha256=sha256(library))
        kw = dict(n_estimators=trees, max_depth=depth, num_leaves=2 ** depth,
                  reg_lambda=ridge, linear_lambda=ridge, min_child_samples=10,
                  max_bin=63, learning_rate=0.1, verbosity=-1, deterministic=True,
                  force_col_wise=True)
        kw.update(p)
        kw.update(n_jobs=1, num_threads=1, random_state=seed, linear_tree=method == "LGB_linear")
        klass = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    # Reject alternate aliases that could override thread/seed policy internally.
    for key in ("nthread", "num_thread", "num_threads", "n_jobs"):
        if key in kw:
            kw[key] = 1
    for key in ("seed", "random_seed"):
        if key in kw:
            kw[key] = seed
    return klass(**kw), lineage, kw


def timed(record, name, action):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        return action()
    finally:
        record[name + "_wall_sec"] = time.perf_counter() - wall
        record[name + "_cpu_sec"] = time.process_time() - cpu


def metrics(task, target, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction")
    if task == "regression":
        if prediction.shape != target.shape:
            raise ValueError("Regression prediction shape mismatch")
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        if not np.isfinite(rmse):
            raise ValueError("Nonfinite RMSE")
        return {"rmse": rmse}
    from sklearn.metrics import log_loss, roc_auc_score
    if prediction.ndim != 2 or prediction.shape[0] != len(target):
        raise ValueError("Classification prediction shape mismatch")
    out = {"logloss": float(log_loss(target, prediction, labels=np.arange(prediction.shape[1]))),
           "accuracy": float(np.mean(np.argmax(prediction, axis=1) == target))}
    if task == "binary":
        out["auc"] = float(roc_auc_score(target, prediction[:, 1])) if len(np.unique(target)) > 1 else None
        if out["auc"] is None:
            out["auc_unavailable_reason"] = "single_class_split"
    return out


def model_info(model, method, nfeatures):
    result = {"n_features": nfeatures}
    try:
        if method in ("CrossGB", "XGB", "LeafOnly"):
            booster = model.get_booster()
            config = booster.save_config()
            result["booster_config"] = json.loads(config)
            if method != "XGB" and not ("grow_metagb_histmaker" in config or '"enable_metagb":"1"' in config):
                raise RuntimeError("CrossGB native updater was not activated")
            trees = json.loads(booster.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
            result["tree_count"] = len(trees)
            result["leaf_count"] = sum(sum(c == -1 for c in t["left_children"]) for t in trees)
            result["predictive_leaf_count"] = sum(
                0 <= t["metagb_leaf_features"][i] < nfeatures
                for t in trees if "metagb_leaf_features" in t
                for i, child in enumerate(t["left_children"]) if child == -1)
        elif method.startswith("LGB"):
            dump = model.booster_.dump_model()
            result["tree_count"] = len(dump["tree_info"])
            result["leaf_count"] = sum(t["num_leaves"] for t in dump["tree_info"])
            result["feature_importance_split"] = model.booster_.feature_importance("split").tolist()
        else:
            result["author_metadata"] = model.metadata_
            for name in ("n_estimators", "num_trees", "num_leaves"):
                value = getattr(model.model_, name, None)
                if value is not None and isinstance(value, (int, float, np.generic)):
                    result[name] = value
    except Exception as error:
        if "not activated" in str(error):
            raise
        result["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def write_result(output, record):
    serializable = dict(record)
    for key in ("requested_params", "effective_params", "estimator_params", "lineage", "model_info"):
        if key in serializable:
            serializable[key] = metadata_json(serializable[key])
    serializable["metadata_nonfinite_policy"] = "metadata-only strings: NaN, +Infinity, -Infinity; metrics remain strict"
    tmp = output / "result.json.tmp"
    tmp.write_text(json.dumps(serializable, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n")
    tmp.replace(output / "result.json")


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = dict(schema_version=1, status="running", method=args.method, seed=args.seed,
                  root=str(Path(args.root).resolve()), data=str(Path(args.data).resolve()),
                  evaluate_test=args.evaluate_test, test_labels_used=False,
                  threads_per_fit=1, preprocessing="supplied_npz; none_in_worker",
                  worker_sha256=sha256(__file__), timings={}, metrics={})
    write_result(output, record)
    started = time.perf_counter()
    stage = "load_params"
    try:
        params = json.loads(Path(args.params).read_text())
        if not isinstance(params, dict):
            raise ValueError("Params JSON must contain an object")
        record["requested_params"] = params
        stage = "load_data"
        X, targets, indices, task = load_data(args.data, args.evaluate_test)
        record.update(task_type=task, n_features=X.shape[1], n_rows=len(X),
                      split_counts={k: len(v) for k, v in indices.items()},
                      split_hashes={k: array_hash(v) for k, v in indices.items()},
                      target_hashes={k: array_hash(v) for k, v in targets.items()})
        stage = "make_model"
        model, lineage, effective = make_model(Path(args.root).resolve(), args.method, task, params, args.seed)
        record.update(lineage=lineage, effective_params=effective)
        write_result(output, record)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            stage = "fit"
            timed(record["timings"], "fit", lambda: model.fit(X[indices["train"]], targets["train"]))
            stage = "model_inspection"
            record["model_info"] = model_info(model, args.method, X.shape[1])
            if hasattr(model, "get_params"):
                record["estimator_params"] = model.get_params(deep=False)
            if args.method in ("PL", "LinXGBoost"):
                metadata = getattr(model, "metadata_", {})
                if "native_library_sha256" in metadata:
                    record["lineage"].update(native_sha256=metadata["native_library_sha256"],
                                             native_library=metadata["native_library"])
                record["effective_params"] = metadata.get("effective_params", effective)
            predictions = {}
            for split in targets:
                stage = "predict_" + split
                predict = model.predict if task == "regression" else model.predict_proba
                pred = np.asarray(timed(record["timings"], "predict_" + split,
                                       lambda: predict(X[indices[split]])))
                record["metrics"][split] = metrics(task, targets[split], pred)
                for key, value in record["metrics"][split].items():
                    record[split + "_" + key] = value
                predictions[split] = pred
                if split == "test":
                    record["test_labels_used"] = True
            np.savez_compressed(output / "predictions.npz", **predictions,
                                **{k + "_indices": indices[k] for k in predictions})
        if args.save_model:
            stage = "save_model"
            try:
                if args.method in ("CrossGB", "XGB", "LeafOnly"):
                    model.save_model(output / "model.json")
                    record["model_path"] = str(output / "model.json")
                elif args.method.startswith("LGB"):
                    model.booster_.save_model(str(output / "model.txt"))
                    record["model_path"] = str(output / "model.txt")
                else:
                    with (output / "model.pkl").open("wb") as stream:
                        pickle.dump(model, stream)
                    record["model_path"] = str(output / "model.pkl")
            except Exception as error:
                record["model_save_error"] = f"{type(error).__name__}: {error}"
        record["status"] = "ok"
        returncode = 0
    except Exception as error:
        record.update(status="failed", failure_stage=stage, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        returncode = 1
    record["total_wall_sec"] = time.perf_counter() - started
    write_result(output, record)
    print(json.dumps({k: record[k] for k in ("status", "method", "seed")}, sort_keys=True))
    return returncode


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, help="Explicit Linux project root")
    cli.add_argument("--data", required=True, help="Preprocessed NPZ")
    cli.add_argument("--method", required=True, choices=METHODS)
    cli.add_argument("--params", required=True, help="Method-specific params JSON file")
    cli.add_argument("--output", required=True, help="Unique per-fit result directory")
    cli.add_argument("--seed", required=True, type=int)
    cli.add_argument("--evaluate-test", action="store_true")
    cli.add_argument("--save-model", action="store_true")
    return cli


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
