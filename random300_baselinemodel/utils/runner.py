from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from random300_baselinemodel.utils.candidates import candidate_config_payload, candidate_sets, load_search_space
from random300_baselinemodel.utils.model_factory import build_estimator
from random300_baselinemodel.utils.schedule import create_ordered_tasks, safe_name, select_datasets
from tabular_extra_random300.utils.metrics import (
    classification_metrics,
    finite_mean,
    labels_from_proba,
    mean_std,
    normalize_proba,
    primary_metric_direction,
    primary_metric_name,
    primary_metric_value,
    regression_metrics,
    select_best_index,
)
from tabular_extra_random300.utils.runner import (
    inner_split_indices,
    limit_rows_for_smoke,
    load_dataset,
    split_indices,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def update_task_status_file(path: Path, task_id: str, status: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tasks": {}}
    payload.setdefault("tasks", {})[task_id] = status
    tasks = payload.get("tasks", {})
    payload["completed"] = sum(1 for value in tasks.values() if value != "pending")
    payload["failed"] = sum(1 for value in tasks.values() if str(value).startswith("failed"))
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, payload)


def fit_predict_timed(model: Any, X_train: Any, y_train: Any, X_eval: Any, task_type: str, n_classes: int) -> Tuple[Any, Any, float, float]:
    start = time.perf_counter()
    try:
        model.fit(X_train, y_train, verbose=False)
    except (TypeError, ValueError) as exc:
        if "verbose" not in str(exc):
            raise
        model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    if task_type == "regression":
        pred = model.predict(X_eval)
        proba = None
    else:
        try:
            proba = normalize_proba(model.predict_proba(X_eval), n_classes)
            pred = labels_from_proba(proba) if proba is not None else model.predict(X_eval)
        except Exception:
            proba = None
            pred = model.predict(X_eval)
    predict_time = time.perf_counter() - start
    return pred, proba, fit_time, predict_time


def evaluate_metrics(task_type: str, y_true: Any, pred: Any, proba: Any, n_classes: int) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred, proba, task_type, n_classes)


def _nan_metrics() -> Dict[str, float]:
    return {
        "rmse": float("nan"),
        "logloss": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "auc": float("nan"),
    }


def run_outer_task(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    args = argparse.Namespace(**task["args"])
    task_log = Path(task["task_log"])
    append_log(task_log, f"start task_id={task['task_id']}")
    data = load_dataset(Path(task["dataset_path"]), task["task_type"], task["categorical_handling"])
    data, original_rows = limit_rows_for_smoke(data, int(getattr(args, "max_rows_per_dataset", 0) or 0), args.seed)
    primary_name = primary_metric_name(task["task_type"])
    direction = primary_metric_direction(task["task_type"])
    append_log(
        task_log,
        "config "
        f"model={task['model']} categorical={task['categorical_handling']} task_type={task['task_type']} "
        f"dataset={task['dataset']} outer_fold={task['outer_index'] + 1}/{args.outer_folds} "
        f"inner_folds={args.inner_folds} candidates={len(task['candidates'])} rows={len(data.y)} "
        f"original_rows={original_rows} features={data.X.shape[1]} classes={data.n_classes} "
        f"primary_metric={primary_name} direction={direction}",
    )

    outer_splits = split_indices(data, args)
    train_idx, test_idx = outer_splits[task["outer_index"]]
    X_train_outer, X_test_outer = data.X[train_idx], data.X[test_idx]
    y_train_outer, y_test_outer = data.y[train_idx], data.y[test_idx]
    inner_splits = inner_split_indices(data, X_train_outer, y_train_outer, args, task["outer_index"])

    rows: List[Dict[str, Any]] = []
    scores: List[Tuple[float, float, float, float, int, str]] = []
    task_start = time.perf_counter()
    for candidate_index, params in enumerate(task["candidates"], start=1):
        val_primary_values: List[float] = []
        val_rmses: List[float] = []
        val_loglosses: List[float] = []
        val_accuracies: List[float] = []
        val_balanced_accuracies: List[float] = []
        val_aucs: List[float] = []
        inner_train_time = 0.0
        inner_predict_time = 0.0
        status = "ok"
        for inner_train_idx, inner_val_idx in inner_splits:
            try:
                estimator = build_estimator(task["model"], params, args, task["task_type"], data.n_classes)
                pred_val, proba_val, fit_time, pred_time = fit_predict_timed(
                    estimator,
                    X_train_outer[inner_train_idx],
                    y_train_outer[inner_train_idx],
                    X_train_outer[inner_val_idx],
                    task["task_type"],
                    data.n_classes,
                )
                metrics = evaluate_metrics(task["task_type"], y_train_outer[inner_val_idx], pred_val, proba_val, data.n_classes)
                val_primary_values.append(primary_metric_value(task["task_type"], metrics))
                val_rmses.append(metrics.get("rmse", float("nan")))
                val_loglosses.append(metrics.get("logloss", float("nan")))
                val_accuracies.append(metrics.get("accuracy", float("nan")))
                val_balanced_accuracies.append(metrics.get("balanced_accuracy", float("nan")))
                val_aucs.append(metrics.get("auc", float("nan")))
                inner_train_time += fit_time
                inner_predict_time += pred_time
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
                val_primary_values = [float("nan")]
                inner_train_time = 0.0
                inner_predict_time = 0.0
                break
        val_primary_mean, val_primary_std = mean_std(val_primary_values)
        scores.append((val_primary_mean, val_primary_std, inner_train_time, inner_predict_time, candidate_index, status))
        rows.append(
            {
                "run_id": args.run_id,
                "model": task["model"],
                "categorical_handling": task["categorical_handling"],
                "task_type": task["task_type"],
                "dataset": task["dataset"],
                "outer_fold": task["outer_index"] + 1,
                "candidate_index": candidate_index,
                "candidate_id": str(params.get("candidate_id", candidate_index)),
                "source": str(params.get("source", "")),
                "selected": False,
                "status": status,
                "primary_metric": primary_name,
                "primary_metric_direction": direction,
                "val_primary_metric_mean": val_primary_mean,
                "val_primary_metric_std": val_primary_std,
                "val_rmse_mean": finite_mean(val_rmses),
                "val_logloss_mean": finite_mean(val_loglosses),
                "val_accuracy_mean": finite_mean(val_accuracies),
                "val_balanced_accuracy_mean": finite_mean(val_balanced_accuracies),
                "val_auc_mean": finite_mean(val_aucs),
                "inner_train_time_sec": inner_train_time,
                "inner_predict_time_sec": inner_predict_time,
                "final_train_time_sec": float("nan"),
                "final_predict_time_sec": float("nan"),
                "test_primary_metric": float("nan"),
                "test_rmse": float("nan"),
                "test_logloss": float("nan"),
                "test_accuracy": float("nan"),
                "test_balanced_accuracy": float("nan"),
                "test_auc": float("nan"),
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        if candidate_index == 1 or candidate_index == len(task["candidates"]) or status != "ok" or candidate_index % max(1, args.progress_every) == 0:
            append_log(task_log, f"candidate {candidate_index}/{len(task['candidates'])} {primary_name}={val_primary_mean:.6g} status={status}")

    best_index = select_best_index(scores, direction)
    if best_index < 0:
        summary = _summary_base(args, task, data, original_rows, primary_name, direction, len(task["candidates"]))
        summary.update({"status": "no_valid_candidate", "task_elapsed_sec": time.perf_counter() - task_start})
        return rows, summary

    best_params = task["candidates"][best_index - 1]
    best_score = scores[best_index - 1]
    try:
        final_estimator = build_estimator(task["model"], best_params, args, task["task_type"], data.n_classes)
        test_pred, test_proba, final_train_time, final_predict_time = fit_predict_timed(
            final_estimator, X_train_outer, y_train_outer, X_test_outer, task["task_type"], data.n_classes
        )
        test_metrics = evaluate_metrics(task["task_type"], y_test_outer, test_pred, test_proba, data.n_classes)
        test_primary = primary_metric_value(task["task_type"], test_metrics)
        final_status = "ok"
    except Exception as exc:
        final_train_time = float("nan")
        final_predict_time = float("nan")
        test_metrics = _nan_metrics()
        test_primary = float("nan")
        final_status = f"failed:{type(exc).__name__}:{exc}"

    for row in rows:
        if row["candidate_index"] == best_index:
            row.update(
                {
                    "selected": True,
                    "status": final_status,
                    "final_train_time_sec": final_train_time,
                    "final_predict_time_sec": final_predict_time,
                    "test_primary_metric": test_primary,
                    "test_rmse": test_metrics.get("rmse", float("nan")),
                    "test_logloss": test_metrics.get("logloss", float("nan")),
                    "test_accuracy": test_metrics.get("accuracy", float("nan")),
                    "test_balanced_accuracy": test_metrics.get("balanced_accuracy", float("nan")),
                    "test_auc": test_metrics.get("auc", float("nan")),
                }
            )
            break

    summary = _summary_base(args, task, data, original_rows, primary_name, direction, len(task["candidates"]))
    summary.update(
        {
            "selected_candidate_index": best_index,
            "selected_candidate_id": str(best_params.get("candidate_id", best_index)),
            "val_primary_metric_mean": best_score[0],
            "val_primary_metric_std": best_score[1],
            "test_primary_metric": test_primary,
            "inner_train_time_sec": best_score[2],
            "inner_predict_time_sec": best_score[3],
            "final_train_time_sec": final_train_time,
            "final_predict_time_sec": final_predict_time,
            "total_train_time_sec": best_score[2] + final_train_time,
            "total_predict_time_sec": best_score[3] + final_predict_time,
            "task_elapsed_sec": time.perf_counter() - task_start,
            "test_rmse": test_metrics.get("rmse", float("nan")),
            "test_logloss": test_metrics.get("logloss", float("nan")),
            "test_accuracy": test_metrics.get("accuracy", float("nan")),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy", float("nan")),
            "test_auc": test_metrics.get("auc", float("nan")),
            "status": final_status,
            "params_json": json.dumps(best_params, sort_keys=True),
        }
    )
    append_log(task_log, f"finish best={best_index} test_{primary_name}={test_primary:.6g} status={final_status}")
    return rows, summary


def _summary_base(args: Any, task: Dict[str, Any], data: Any, original_rows: int, primary_name: str, direction: str, candidate_count: int) -> Dict[str, Any]:
    return {
        "run_id": args.run_id,
        "model": task["model"],
        "categorical_handling": task["categorical_handling"],
        "task_type": task["task_type"],
        "dataset": task["dataset"],
        "outer_fold": task["outer_index"] + 1,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "primary_metric": primary_name,
        "primary_metric_direction": direction,
        "rows_original": original_rows,
        "rows_used": int(len(data.y)),
        "max_rows_per_dataset": int(getattr(args, "max_rows_per_dataset", 0) or 0),
        "features_used": int(data.X.shape[1]),
        "candidate_count": candidate_count,
    }


def build_aggregate(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not summary_rows:
        return []
    df = pd.DataFrame(summary_rows)
    metric_cols = [
        "val_primary_metric_mean",
        "test_primary_metric",
        "inner_train_time_sec",
        "inner_predict_time_sec",
        "final_train_time_sec",
        "final_predict_time_sec",
        "total_train_time_sec",
        "total_predict_time_sec",
        "task_elapsed_sec",
        "test_rmse",
        "test_logloss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auc",
    ]
    available = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(["model", "categorical_handling", "task_type", "dataset"], as_index=False)[available].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.to_flat_index()]
    return grouped.to_dict(orient="records")


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    search_space = load_search_space(Path(args.search_space_json) if args.search_space_json else None)
    selected_candidates = candidate_sets(args.models, args.max_candidates, args.seed, search_space)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)
    write_json(results_dir / "candidate_configs.json", candidate_config_payload(selected_candidates))
    write_json(
        results_dir / "parameter_audit.json",
        {
            "models": args.models,
            "categorical_handlings": args.categorical_handlings,
            "max_candidates": args.max_candidates,
            "search_space_json": args.search_space_json,
            "notes": "Baseline models use defaults plus simple random-search candidates. Deep model hyperparameters remain conservative pending final protocol discussion.",
        },
    )

    tasks = create_ordered_tasks(args, selected_candidates)
    for task in tasks:
        if not Path(task["dataset_path"]).exists():
            raise FileNotFoundError(f"Dataset not found: {task['dataset_path']}")
        task["task_log"] = str(logs_dir / "tasks" / f"{task['task_id']}.log")
        task["args"] = vars(args)

    task_status_path = logs_dir / "task_status.json"
    write_json(task_status_path, {"run_id": args.run_id, "total": len(tasks), "completed": 0, "failed": 0, "tasks": {task["task_id"]: "pending" for task in tasks}})
    candidate_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    worker_count = min(max(1, args.task_workers), len(tasks))
    start = time.perf_counter()

    if worker_count == 1:
        for index, task in enumerate(tasks, start=1):
            rows, summary = run_outer_task(task)
            candidate_rows.extend(rows)
            summary_rows.append(summary)
            update_task_status_file(task_status_path, task["task_id"], str(summary.get("status")))
            print(f"[{index}/{len(tasks)}] {task['task_id']} status={summary.get('status')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(run_outer_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                try:
                    rows, summary = future.result()
                except Exception as exc:
                    rows = []
                    summary = {
                        "run_id": args.run_id,
                        "model": task["model"],
                        "categorical_handling": task["categorical_handling"],
                        "task_type": task["task_type"],
                        "dataset": task["dataset"],
                        "outer_fold": task["outer_index"] + 1,
                        "status": f"failed:{type(exc).__name__}:{exc}",
                    }
                candidate_rows.extend(rows)
                summary_rows.append(summary)
                update_task_status_file(task_status_path, task["task_id"], str(summary.get("status")))
                print(f"[{index}/{len(tasks)}] {task['task_id']} status={summary.get('status')}", flush=True)

    sort_key = lambda row: (
        row.get("model", ""),
        row.get("categorical_handling", ""),
        row.get("task_type", ""),
        row.get("dataset", ""),
        row.get("outer_fold", 0),
        row.get("candidate_index", 0),
    )
    candidate_rows.sort(key=sort_key)
    summary_rows.sort(key=sort_key)
    write_csv(results_dir / "candidates.csv", candidate_rows)
    write_csv(results_dir / "summary.csv", summary_rows)
    write_csv(results_dir / "aggregate.csv", build_aggregate(summary_rows))

    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.perf_counter() - start,
        "models": args.models,
        "categorical_handlings": args.categorical_handlings,
        "task_types": args.task_types,
        "max_candidates": args.max_candidates,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "run_all_outer_folds": bool(args.run_all_outer_folds or args.mode == "full"),
        "task_workers": args.task_workers,
        "n_jobs": args.n_jobs,
        "selected_datasets": {
            "regression": select_datasets(args, "regression") if "regression" in args.task_types else [],
            "binary": select_datasets(args, "binary") if "binary" in args.task_types else [],
            "multiclass": select_datasets(args, "multiclass") if "multiclass" in args.task_types else [],
        },
        "results_dir": str(results_dir),
        "logs_dir": str(logs_dir),
        "summary_rows": len(summary_rows),
        "candidate_rows": len(candidate_rows),
        "failed_summary_rows": sum(1 for row in summary_rows if str(row.get("status", "")).startswith("failed") or row.get("status") == "no_valid_candidate"),
    }
    write_json(results_dir / "run_metadata.json", metadata)
    return metadata


def finite_or_none(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(numeric):
        return numeric
    return None
