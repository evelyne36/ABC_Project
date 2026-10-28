#!/usr/bin/env python3
"""Target-only persistence/Ridge panel using a specified canonical TS source tree.

This is a new reference experiment, never a reconstruction of old tree results.
No downloads, native tree imports, inner-CV search, or hidden candidate truncation.
Scientific dependencies are imported only after CLI parsing; --help is stdlib-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace
import warnings

RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)
REQUIRED_ARGS = ("run_id", "seed", "split_protocol", "feature_mode", "target_mode",
                 "output_scope", "include_time_features", "max_raw_rows",
                 "max_features", "origin_state_channel_budget")
SOURCE_FILES = (
    "exp/timeseries_extra_random300/utils/data.py",
    "exp/timeseries_extra_random300/utils/splits.py",
    "exp/timeseries_extra_random300/utils/metrics.py",
    "exp/timeseries_extra_random300/scripts/run_timeseries_benchmark.py",
    "exp/random600/__init__.py",
    "exp/random600/utils/__init__.py",
    "exp/random600/utils/runner.py",
    "exp/random600/utils/candidates.py",
    "exp/random600/utils/metrics.py",
    "exp/random600/utils/parameter_audit.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def load_source(source_root):
    """Import original helpers, verify their origin, and avoid native tree loading."""
    root = Path(source_root).resolve()
    for relative in SOURCE_FILES:
        if not (root / relative).is_file():
            raise ValueError(f"Missing canonical source file: {root / relative}")
    exp = root / "exp"
    sys.path.insert(0, str(exp))
    names = {
        "data": "timeseries_extra_random300.utils.data",
        "splits": "timeseries_extra_random300.utils.splits",
        "metrics": "timeseries_extra_random300.utils.metrics",
        "runner": "timeseries_extra_random300.scripts.run_timeseries_benchmark",
    }
    loaded = {}
    for alias, name in names.items():
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(exp):
            raise ValueError(f"Source namespace already loaded from another tree: {name}: {origin}")
        loaded[alias] = module
    if any(name == "xgboost" or name.startswith("xgboost.") for name in sys.modules):
        raise ValueError("Native xgboost was imported; run this tool in a clean Python process")
    loaded["np"] = importlib.import_module("numpy")
    loaded["sklearn"] = importlib.import_module("sklearn")
    loaded["Ridge"] = importlib.import_module("sklearn.linear_model").Ridge
    return SimpleNamespace(**loaded)


def checked_args(payload):
    args = dict(payload.get("args", payload))
    missing = [key for key in REQUIRED_ARGS if key not in args]
    if missing:
        raise ValueError("Run args JSON missing explicit semantic fields: " + ", ".join(missing))
    if args["split_protocol"] != "fixed":
        raise ValueError("Only canonical fixed chronological split is supported; nested/rolling requires its own adapter")
    if args["output_scope"] != "target":
        raise ValueError("Only output_scope=target is supported; persistence must not broadcast a single target across channels")
    if args["feature_mode"] not in ("window", "window_summary"):
        raise ValueError("Only window/window_summary preserve the last-target lag position used by persistence")
    if args["target_mode"] not in ("direct", "sequence"):
        raise ValueError("target_mode must be direct or sequence")
    if not isinstance(args["include_time_features"], bool):
        raise ValueError("include_time_features must be a JSON boolean")
    for key in ("seed", "max_raw_rows", "max_features", "origin_state_channel_budget"):
        if isinstance(args[key], bool) or not isinstance(args[key], int):
            raise ValueError(f"{key} must be an integer")
    for key in ("max_raw_rows", "max_features", "origin_state_channel_budget"):
        if args[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not isinstance(args["run_id"], str) or not args["run_id"].strip():
        raise ValueError("run_id must be a nonempty string")
    # Tree search controls such as max_candidates are never used by this panel.
    return args


def checked_tasks(payload):
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Tasks JSON must contain a nonempty list (or {\"tasks\": [...]})")
    checked, seen = [], set()
    for task in tasks:
        if not isinstance(task, dict) or any(key not in task for key in ("dataset", "seq_len", "pred_len")):
            raise ValueError("Each task requires dataset, seq_len, pred_len")
        if not isinstance(task["dataset"], str) or not task["dataset"]:
            raise ValueError("dataset must be a nonempty string")
        for key in ("seq_len", "pred_len"):
            if isinstance(task[key], bool) or not isinstance(task[key], int) or task[key] < 1:
                raise ValueError(f"Task {key} must be a positive integer")
        identity = (task["dataset"], task["seq_len"], task["pred_len"])
        if identity in seen:
            raise ValueError(f"Duplicate task: {identity}")
        seen.add(identity)
        checked.append({key: task[key] for key in ("dataset", "seq_len", "pred_len")})
    return checked


def synthetic_series(api):
    np, data = api.np, api.data
    rng = np.random.default_rng(20261025)
    n = 192
    t = np.arange(n, dtype=np.float32)
    # Deliberately put the target in the middle, avoiding a last-channel-only test.
    values = np.column_stack((np.sin(t / 7), .02 * t + np.sin(t / 11) + rng.normal(0, .08, n),
                              np.cos(t / 5))).astype(np.float32)
    spec = data.DatasetSpec("synthetic_reference_smoke", "synthetic", "", "", "", "", "", "target", "h")
    timestamps = [str(value) for value in api.data.pd.date_range("2020-01-01", periods=n, freq="h")]
    return data.RawTimeSeries(spec.name, values, 1, ["first", "target", "last"], timestamps, spec)


def task_namespace(base, task):
    return argparse.Namespace(**{**base, **task})


def get_splits(api, series, args):
    count = api.data.supervised_row_count(series, args.seq_len, args.pred_len)
    train, val, test, fixed = api.splits.fixed_chronological_split(count, gap=api.runner.horizon_gap(args))
    api.splits.assert_time_ordered(train, val)
    api.splits.assert_time_ordered(val, test)
    return train, val, test, fixed


def make_fold(api, series, train, evaluation, args):
    return api.data.make_fold_arrays(
        series, train, evaluation, seq_len=args.seq_len, pred_len=args.pred_len,
        include_time_features=args.include_time_features, feature_mode=args.feature_mode,
        target_mode=args.target_mode, output_scope=args.output_scope,
        origin_state_channel_budget=args.origin_state_channel_budget)


def split_record(api, series, args, train, val, test, fixed, final_fold):
    record = {"dataset": series.name, "split_protocol": args.split_protocol, "seed": args.seed,
              "outer_fold": 1, "fixed_split": fixed.to_dict(), "seq_len": args.seq_len,
              "pred_len": args.pred_len, "feature_mode": args.feature_mode,
              "origin_state_channel_budget": args.origin_state_channel_budget,
              "target_mode": args.target_mode, "output_scope": args.output_scope,
              "input_features": int(final_fold.X_train.shape[1]),
              "output_dim": int(final_fold.y_train.shape[1]) if final_fold.y_train.ndim == 2 else 1}
    for name, indices in (("train", train), ("val", val), ("test", test)):
        record[f"{name}_row_min"] = int(indices.min())
        record[f"{name}_row_max"] = int(indices.max())
    record.update(scaler_fit_start=int(final_fold.scaler_fit_start), scaler_fit_end=int(final_fold.scaler_fit_end),
                  train_val_target_start=int(final_fold.train_target_start), train_val_target_end=int(final_fold.train_target_end),
                  test_target_start=int(final_fold.eval_target_start), test_target_end=int(final_fold.eval_target_end))
    for prefix, pos in (("train_val_target_start", final_fold.train_target_start),
                        ("train_val_target_end", final_fold.train_target_end - 1),
                        ("test_target_start", final_fold.eval_target_start),
                        ("test_target_end", final_fold.eval_target_end - 1)):
        record[prefix + "_time"] = api.runner.split_timestamp(series.timestamps, pos)
    return record


def compare_original(record, originals):
    keys = ("dataset", "seed", "seq_len", "pred_len")
    matching = [old for old in originals if all(old.get(key) == record[key] for key in keys)]
    if not matching:
        raise ValueError("No corresponding original split record for " + str({key: record[key] for key in keys}))
    # Several tree models may share the same task. Every matching archived record
    # must agree, so cherry-picking one model cannot hide inconsistent splits.
    compare_keys = tuple(record)
    for old in matching:
        missing = [key for key in compare_keys if key not in old]
        differences = {key: {"old": old.get(key), "new": record[key]} for key in compare_keys
                       if key in old and old[key] != record[key]}
        if missing or differences:
            raise ValueError("Original split mismatch: " + json.dumps({"model": old.get("model"), "missing": missing, "differences": differences}))
    return {"status": "all_matching_split_records_equal", "records_compared": len(matching),
            "data_identity_proven": False}


class PersistenceRegressor:
    def __init__(self, last_target_column):
        self.last_target_column = int(last_target_column)

    def fit(self, X, y, verbose=False):
        import numpy as np
        shape = np.asarray(y).shape
        self.return_1d = len(shape) == 1
        self.outputs = 1 if self.return_1d else shape[1]
        return self

    def predict(self, X):
        import numpy as np
        last = np.asarray(X, dtype=np.float32)[:, self.last_target_column]
        return last.copy() if self.return_1d else np.repeat(last[:, None], self.outputs, axis=1)


def build_model(api, model, params, series, args):
    if model == "persistence":
        return PersistenceRegressor((args.seq_len - 1) * series.n_features + series.target_index)
    if model == "ridge_cpu":
        # Match the canonical runner's independent estimator per output/horizon.
        return api.runner.ColumnwiseMultiOutputRegressor(
            lambda: api.Ridge(alpha=float(params["alpha"])))
    raise ValueError(f"Unknown reference model: {model}")


def target_positions(api, indices, args):
    np = api.np
    rows = np.asarray(indices, dtype=np.int64)
    if args.target_mode == "direct":
        return rows + args.seq_len + args.pred_len - 1
    return rows[:, None] + args.seq_len + np.arange(args.pred_len)[None, :]


def save_prediction(api, path, series, args, fold, train_rows, eval_rows, prediction):
    np = api.np
    if prediction.shape != fold.y_eval.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Prediction shape/finite check failed: {prediction.shape} vs {fold.y_eval.shape}")
    # Reconstruct only an affine output conversion using the same source scaler;
    # model selection always uses the original standardized RMSE.
    scaler = api.data.StandardScaler().fit(series.values[fold.scaler_fit_start:fold.scaler_fit_end])
    mean, scale = float(scaler.mean_[series.target_index]), float(scaler.scale_[series.target_index])
    positions = target_positions(api, eval_rows, args)
    truth_raw = series.values[positions, series.target_index]
    prediction_raw = prediction.astype(np.float64) * scale + mean
    np.savez_compressed(path, y_true=fold.y_eval, y_pred=prediction,
                        y_true_raw=truth_raw, y_pred_raw=prediction_raw,
                        supervised_train_rows=train_rows, supervised_eval_rows=eval_rows,
                        raw_target_positions=positions, target_mean=mean, target_scale=scale,
                        scaler_fit_start=fold.scaler_fit_start, scaler_fit_end=fold.scaler_fit_end)
    return api.metrics.regression_metrics(truth_raw, prediction_raw)


def run_task(api, series, args, output_dir, counter):
    np = api.np
    train, val, test, fixed = get_splits(api, series, args)
    final_train = np.concatenate([train, val])  # preserve the canonical internal gap
    val_fold = make_fold(api, series, train, val, args)
    # Final test arrays are built only after all validation selections below.
    candidates, summaries, selected = [], [], []
    task_id = f"{safe_name(series.name)}__s{args.seq_len}__h{args.pred_len}__seed{args.seed}"
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for model, configs in (("persistence", [{}]), ("ridge_cpu", [{"alpha": value} for value in RIDGE_ALPHAS])):
        scores = []
        local_rows = []
        for index, params in enumerate(configs, 1):
            row = {"run_id": args.run_id, "dataset": series.name, "seed": args.seed,
                   "seq_len": args.seq_len, "pred_len": args.pred_len, "split_protocol": "fixed",
                   "feature_mode": args.feature_mode, "target_mode": args.target_mode, "output_scope": "target",
                   "model": model, "candidate_index": index, "params_json": json.dumps(params, sort_keys=True),
                   "selected": False, "status": "failed", "val_rmse": "", "val_mae": "", "val_mse": "",
                   "inner_train_time_sec": "", "inner_predict_time_sec": "", "failed_elapsed_sec": "",
                   "prediction_file": "", "warnings_json": "[]", "error": ""}
            start = time.perf_counter()
            try:
                estimator = build_model(api, model, params, series, args)
                counter["estimator_fit_calls"] += 1
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    pred, fit_time, predict_time = api.runner.fit_predict(estimator, val_fold)
                row["warnings_json"] = json.dumps(list(dict.fromkeys(str(item.message) for item in observed)))
                metrics = api.metrics.regression_metrics(val_fold.y_eval, pred)
                relative = Path("predictions") / f"{task_id}__{model}__val_c{index:02d}.npz"
                save_prediction(api, output_dir / relative, series, args, val_fold, train, val, pred)
                row.update(status="ok", val_rmse=metrics["rmse"], val_mae=metrics["mae"], val_mse=metrics["mse"],
                           inner_train_time_sec=fit_time, inner_predict_time_sec=predict_time, prediction_file=str(relative))
                scores.append((metrics["rmse"], fit_time + predict_time, index, "ok"))
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", failed_elapsed_sec=time.perf_counter() - start)
                scores.append((float("nan"), float("inf"), index, "failed"))
            local_rows.append(row)
            write_csv(output_dir / f"{task_id}__candidate_progress.csv", candidates + local_rows)
        best = api.metrics.select_min_rmse(scores)
        candidates.extend(local_rows)
        if best < 1:
            raise RuntimeError(f"All candidates failed: {task_id} {model}; errors={local_rows}")
        local_rows[best - 1]["selected"] = True
        selected.append((model, configs[best - 1], local_rows[best - 1]))
        write_csv(output_dir / f"{task_id}__candidate_progress.csv", candidates)
    final_fold = make_fold(api, series, final_train, test, args)
    record = split_record(api, series, args, train, val, test, fixed, final_fold)
    record["validation_scaler_fit_start"] = int(val_fold.scaler_fit_start)
    record["validation_scaler_fit_end"] = int(val_fold.scaler_fit_end)
    record["horizon_gap"] = api.runner.horizon_gap(args)
    record["final_train_row_count"] = int(len(final_train))
    for model, params, selected_row in selected:
        estimator = build_model(api, model, params, series, args)  # fresh refit, no validation estimator reuse
        counter["estimator_fit_calls"] += 1
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            pred, fit_time, predict_time = api.runner.fit_predict(estimator, final_fold)
        metrics = api.metrics.regression_metrics(final_fold.y_eval, pred)
        relative = Path("predictions") / f"{task_id}__{model}__test.npz"
        raw_metrics = save_prediction(api, output_dir / relative, series, args, final_fold, final_train, test, pred)
        summaries.append({"run_id": args.run_id, "dataset": series.name, "seed": args.seed,
                          "seq_len": args.seq_len, "pred_len": args.pred_len, "model": model,
                          "status": "ok", "selected_candidate_index": selected_row["candidate_index"],
                          "params_json": json.dumps(params, sort_keys=True), "val_rmse": selected_row["val_rmse"],
                          "test_rmse": metrics["rmse"], "test_mae": metrics["mae"], "test_mse": metrics["mse"],
                          "test_rmse_raw": raw_metrics["rmse"], "test_mae_raw": raw_metrics["mae"],
                          "final_train_time_sec": fit_time, "final_predict_time_sec": predict_time,
                          "warnings_json": json.dumps(list(dict.fromkeys(str(item.message) for item in observed))),
                          "prediction_file": str(relative), "historical_reproduction_claim": False})
    return candidates, summaries, record


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True, type=Path, help="Canonical source tree containing exp/")
    result.add_argument("--output-dir", required=True, type=Path, help="Fresh output directory (no overwrite/resume)")
    result.add_argument("--run-args-json", type=Path, help="Original or explicitly proposed args; semantic fields are mandatory")
    result.add_argument("--tasks-json", type=Path, help="Explicit list of dataset/seq_len/pred_len tasks")
    result.add_argument("--registry", type=Path, help="Default SOURCE/exp/timeseries_extra_random300/configs/dataset_registry.json")
    result.add_argument("--series-root", type=Path, help="Root containing data/raw/<registry raw_path>; never downloads")
    result.add_argument("--original-split-manifest", type=Path, help="Optional archived {splits:[...]} to require exact split-field agreement")
    result.add_argument("--expected-data-hashes", type=Path, help="Optional JSON dataset -> raw-file SHA256")
    result.add_argument("--acknowledge-unverified-config", action="store_true", help="Allow NEW real-data fits from args not labeled recovered_original; never establishes old-table reproduction")
    result.add_argument("--plan-only", action="store_true", help="Preflight data/config/splits; zero estimator fits; exit 2 if blocked")
    result.add_argument("--synthetic-smoke", action="store_true", help="Use deterministic in-memory 192x3 series, target middle channel, horizons 1/6/12")
    return result


def main(argv=None):
    cli = parser().parse_args(argv)
    output = cli.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be fresh/empty: {output}")
    # One process, one numeric thread. The timing records are not matched to tree
    # jobs on other hardware, and exclude preprocessing, serializing, and startup.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    api = load_source(cli.source_root)
    source = cli.source_root.expanduser().resolve()
    adapter_root = source / "exp" / "timeseries_extra_random300"
    registry_path = (cli.registry or adapter_root / "configs" / "dataset_registry.json").expanduser().resolve()
    series_root = (cli.series_root or adapter_root).expanduser().resolve()
    if cli.synthetic_smoke:
        if any((cli.run_args_json, cli.tasks_json, cli.registry, cli.series_root, cli.original_split_manifest, cli.expected_data_hashes)):
            raise ValueError("--synthetic-smoke is isolated; do not combine with real-data/config arguments")
        base = checked_args({"run_id": "synthetic_reference_smoke", "seed": 42, "split_protocol": "fixed",
                             "feature_mode": "window_summary", "target_mode": "sequence", "output_scope": "target",
                             "include_time_features": True, "max_raw_rows": 0, "max_features": 0,
                             "origin_state_channel_budget": 32})
        tasks = [{"dataset": "synthetic_reference_smoke", "seq_len": 24, "pred_len": h} for h in (1, 6, 12)]
        provenance = "synthetic_development_only"
        series_by_name = {"synthetic_reference_smoke": synthetic_series(api)}
        registry, expected_hashes, originals = {}, {}, None
    else:
        if not cli.run_args_json or not cli.tasks_json:
            raise ValueError("Real-data plan/run requires --run-args-json AND --tasks-json; no inferred historical args")
        payload = read_json(cli.run_args_json)
        base = checked_args(payload)
        tasks = checked_tasks(read_json(cli.tasks_json))
        provenance = payload.get("provenance", "supplied_config_unverified_original")
        registry = api.data.load_registry(registry_path)
        expected_hashes = read_json(cli.expected_data_hashes) if cli.expected_data_hashes else {}
        originals = read_json(cli.original_split_manifest).get("splits") if cli.original_split_manifest else None
        if cli.original_split_manifest and not isinstance(originals, list):
            raise ValueError("Original split manifest requires a splits list")
        series_by_name = {}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"tool": "run_ts_reference_panel.py", "tool_sha256": sha256(__file__), "status": "preflight",
                "mode": "plan_only" if cli.plan_only else "execute", "synthetic": bool(cli.synthetic_smoke),
                "historical_reproduction_claim": False, "config_provenance": provenance,
                "source_root": str(source), "series_root": str(series_root), "run_args": base,
                "ridge_alphas": list(RIDGE_ALPHAS), "models": ["persistence", "ridge_cpu"],
                "selection_rule": "canonical select_min_rmse: validation RMSE, fit+predict seconds, candidate index",
                "refit_rule": "fresh estimator on concatenate(train_rows,val_rows), retaining internal row gap",
                "timing_unit": "recorded fit/predict seconds; excludes preprocessing/I/O/startup; not equal wallclock compute",
                "source_sha256": {relative: sha256(source / relative) for relative in SOURCE_FILES},
                "python": platform.python_version(), "numpy": api.np.__version__, "sklearn": api.sklearn.__version__,
                "thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                "estimator_fit_calls": 0, "planned_columnwise_base_estimator_fits": 0,
                "tasks": [], "input_files": {}, "notes": [
                    "New fixed-split target-only reference results; no rolling-origin uncertainty estimate.",
                    "Equal split fields do not establish identical raw values; data hashes must be checked separately.",
                    "Validation and test metrics use different canonical scaler fits; do not compare their magnitudes as a learning curve.",
                    "No tree comparison or compute-match claim is produced."]}
    for name, path in (("run_args_json", cli.run_args_json), ("tasks_json", cli.tasks_json), ("registry", None if cli.synthetic_smoke else registry_path),
                       ("original_split_manifest", cli.original_split_manifest), ("expected_data_hashes", cli.expected_data_hashes)):
        if path:
            manifest["input_files"][name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    prepared = []
    for task in tasks:
        entry = {**task, "status": "blocked", "errors": []}
        try:
            name = task["dataset"] if cli.synthetic_smoke else api.data.resolve_dataset_name(registry, task["dataset"])
            task = {**task, "dataset": name}
            entry.update(dataset=name)
            if name not in series_by_name:
                raw_path = api.data.raw_path_for(series_root, registry[name])
                if not raw_path.is_file():
                    raise FileNotFoundError(f"Required raw data absent: {raw_path}")
                actual_hash = sha256(raw_path)
                if cli.expected_data_hashes and name not in expected_hashes:
                    raise ValueError(f"Expected data hashes missing dataset: {name}")
                if name in expected_hashes and expected_hashes[name].lower() != actual_hash:
                    raise ValueError(f"Raw data hash mismatch: {name}")
                series = api.data.load_raw_series(series_root, registry[name])
                if base["max_raw_rows"]:
                    series.values = series.values[:base["max_raw_rows"]]
                    series.timestamps = series.timestamps[:base["max_raw_rows"]]
                series = api.runner.limit_series_features(series, base["max_features"])
                series_by_name[name] = series
                manifest["input_files"][f"dataset:{name}"] = {"path": str(raw_path), "sha256": actual_hash,
                                                               "expected_hash_verified": name in expected_hashes}
            series = series_by_name[name]
            args = task_namespace(base, task)
            train, val, test, fixed = get_splits(api, series, args)
            final_train = api.np.concatenate([train, val])
            # Preflight uses the exact source feature/scaler path. No estimator fit.
            final_fold = make_fold(api, series, final_train, test, args)
            record = split_record(api, series, args, train, val, test, fixed, final_fold)
            entry.update(status="ready", raw_rows=series.n_rows, channels=series.n_features,
                         target_index=series.target_index, target_column=series.columns[series.target_index],
                         horizon_gap=api.runner.horizon_gap(args), train_rows=len(train), val_rows=len(val), test_rows=len(test),
                         ridge_candidates=10, persistence_candidates=1, planned_panel_fits=13, split_record=record)
            if originals is not None:
                entry["original_split_check"] = compare_original(record, originals)
            if not cli.synthetic_smoke and provenance != "recovered_original" and not cli.acknowledge_unverified_config:
                entry["execution_gate"] = "Pass --acknowledge-unverified-config to run this as a NEW experiment, or recover original args and record provenance. Plan-only is allowed."
                if not cli.plan_only:
                    raise ValueError(entry["execution_gate"])
            prepared.append((series, args))
            del final_fold
        except Exception as exc:
            entry["status"] = "blocked"
            entry["errors"].append(f"{type(exc).__name__}: {exc}")
        manifest["tasks"].append(entry)
    blocked = [task for task in manifest["tasks"] if task["status"] != "ready"]
    manifest["status"] = "blocked" if blocked else ("plan_ready" if cli.plan_only else "ready_to_execute")
    write_json(output / "manifest.json", manifest)
    if blocked or cli.plan_only:
        print(json.dumps({"status": manifest["status"], "ready_tasks": len(tasks) - len(blocked), "blocked_tasks": len(blocked),
                          "estimator_fit_calls": 0, "manifest": str(output / "manifest.json")}, ensure_ascii=False))
        return 2 if blocked else 0
    all_candidates, all_summaries, all_records = [], [], []
    start = time.perf_counter()
    try:
        for series, args in prepared:
            outputs = args.pred_len if args.target_mode == "sequence" else 1
            manifest["planned_columnwise_base_estimator_fits"] += 11 * outputs
            rows, summaries, record = run_task(api, series, args, output, manifest)
            all_candidates.extend(rows)
            all_summaries.extend(summaries)
            all_records.append(record)
            write_csv(output / "candidates.csv", all_candidates)
            write_csv(output / "summary.csv", all_summaries)
            write_json(output / "split_manifest.json", {"splits": all_records})
        manifest["status"] = "completed_synthetic_smoke" if cli.synthetic_smoke else "completed_new_reference_panel"
        manifest["native_xgboost_imported"] = any(name == "xgboost" or name.startswith("xgboost.") for name in sys.modules)
        if manifest["native_xgboost_imported"]:
            raise RuntimeError("Unexpected native tree import")
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["execution_elapsed_sec"] = time.perf_counter() - start
        manifest["output_sha256"] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                                     if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "tasks": len(prepared), "candidate_rows": len(all_candidates),
                      "summary_rows": len(all_summaries), "estimator_fit_calls": manifest["estimator_fit_calls"],
                      "manifest": str(output / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
