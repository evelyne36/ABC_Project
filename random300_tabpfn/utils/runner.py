from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from random300_tabpfn.utils.candidates import candidate_config_payload, candidate_sets, load_search_space
from random300_tabpfn.utils.metrics import (
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
from random300_tabpfn.utils.parameter_audit import build_parameter_audit


DEFAULT_REGRESSION_DATASETS = [
    "regression__kin8nm.csv",
    "regression__energy_efficiency.csv",
    "regression__naval_propulsion_plant.csv",
    "regression__cpu_activity.csv",
    "regression__space_ga.csv",
    "regression__Moneyball.csv",
]


DEFAULT_TASK_DATASETS = {
    "regression": DEFAULT_REGRESSION_DATASETS,
    "binary": [
        "binary__banknote-authentication.csv",
        "binary__credit-approval.csv",
        "binary__breast-w.csv",
        "binary__ilpd.csv",
        "binary__haberman.csv",
        "binary__kc1.csv",
        "binary__electricity.csv",
    ],
    "multiclass": [
        "multiclass__vehicle.csv",
        "multiclass__segment.csv",
        "multiclass__satimage.csv",
        "multiclass__vowel.csv",
        "multiclass__optdigits.csv",
    ],
}

DATASET_INSTANCE_COUNTS: Dict[Tuple[str, str], int] = {
    ("regression", "regression__Moneyball.csv"): 114,
    ("binary", "binary__haberman.csv"): 306,
    ("binary", "binary__ilpd.csv"): 583,
    ("binary", "binary__credit-approval.csv"): 653,
    ("binary", "binary__breast-w.csv"): 683,
    ("regression", "regression__energy_efficiency.csv"): 768,
    ("multiclass", "multiclass__vehicle.csv"): 846,
    ("multiclass", "multiclass__vowel.csv"): 990,
    ("binary", "binary__banknote-authentication.csv"): 1372,
    ("binary", "binary__kc1.csv"): 2109,
    ("multiclass", "multiclass__segment.csv"): 2310,
    ("regression", "regression__space_ga.csv"): 3107,
    ("multiclass", "multiclass__optdigits.csv"): 5620,
    ("multiclass", "multiclass__satimage.csv"): 6430,
    ("regression", "regression__kin8nm.csv"): 8192,
    ("regression", "regression__cpu_activity.csv"): 8192,
    ("regression", "regression__naval_propulsion_plant.csv"): 11934,
    ("binary", "binary__electricity.csv"): 38474,
}


@dataclass
class LoadedDataset:
    X: Any
    y: Any
    task_type: str
    n_classes: int
    class_labels: List[str]


@dataclass
class CandidateRow:
    run_id: str
    task_type: str
    dataset: str
    model: str
    outer_fold: int
    candidate_index: int
    candidate_id: str
    backbone_id: str
    source: str
    pred_feature_fraction: float
    metagb_score_agg: float
    selected: bool
    status: str
    primary_metric: str
    primary_metric_direction: str
    val_primary_metric_mean: float
    val_primary_metric_std: float
    val_rmse_mean: float
    val_logloss_mean: float
    val_accuracy_mean: float
    val_balanced_accuracy_mean: float
    val_auc_mean: float
    inner_train_time_sec: float
    inner_predict_time_sec: float
    final_train_time_sec: float
    final_predict_time_sec: float
    test_primary_metric: float
    test_rmse: float
    test_logloss: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_auc: float
    params_json: str


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


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


def import_local_xgboost(python_package: Path) -> Any:
    python_package = python_package.expanduser().resolve()
    if not python_package.exists():
        raise FileNotFoundError(f"Local XGBoost python package not found: {python_package}")
    for module_name in list(sys.modules):
        if module_name == "xgboost" or module_name.startswith("xgboost."):
            del sys.modules[module_name]
    if str(python_package) not in sys.path:
        sys.path.insert(0, str(python_package))
    import xgboost as xgb

    return xgb


def dataset_dir_for_task(args: argparse.Namespace, task_type: str) -> Path:
    mapping = {
        "regression": args.regression_dir,
        "binary": args.binary_dir,
        "multiclass": args.multiclass_dir,
    }
    return Path(mapping[task_type]).expanduser().resolve()


def select_datasets(args: argparse.Namespace, task_type: str) -> List[str]:
    requested = getattr(args, f"{task_type}_datasets")
    if requested == ["all"]:
        selected = list(DEFAULT_TASK_DATASETS[task_type])
    else:
        selected = list(requested)
    skip_all = set(getattr(args, "skip_datasets", []) or [])
    skip_task = set(getattr(args, f"skip_{task_type}_datasets", []) or [])
    skip = skip_all | skip_task
    return [dataset for dataset in selected if dataset not in skip]


def preprocess_features(feature_df: Any, target: Any, categorical_handling: str) -> Tuple[Any, Any]:
    non_missing = target.notna()
    feature_df = feature_df.loc[non_missing].copy()
    target = target.loc[non_missing].copy()

    if categorical_handling == "drop":
        feature_df = feature_df.select_dtypes(include=[np.number])
        feature_df = feature_df.dropna()
        target = target.loc[feature_df.index]
    elif categorical_handling == "onehot":
        feature_df = feature_df.dropna()
        target = target.loc[feature_df.index]
        categorical_cols = list(feature_df.select_dtypes(exclude=[np.number]).columns)
        if categorical_cols:
            feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)
    elif categorical_handling == "ordinal":
        categorical_cols = list(feature_df.select_dtypes(exclude=[np.number]).columns)
        for column in categorical_cols:
            feature_df[column], _ = pd.factorize(feature_df[column], sort=True)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
        combined = feature_df.copy()
        combined["__target__"] = target
        combined = combined.dropna()
        target = combined["__target__"]
        feature_df = combined.drop(columns=["__target__"])
    else:
        raise ValueError("categorical_handling must be one of: drop, onehot, ordinal")

    if len(feature_df) == 0:
        raise ValueError("No rows remain after preprocessing")
    if feature_df.shape[1] == 0:
        raise ValueError("No feature columns remain after preprocessing")
    return feature_df, target


def load_dataset(csv_path: Path, task_type: str, categorical_handling: str) -> LoadedDataset:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must have at least one feature and one target: {csv_path}")

    feature_df, target = preprocess_features(df.iloc[:, :-1], df.iloc[:, -1], categorical_handling)
    X = feature_df.to_numpy(dtype=np.float32, copy=True)
    if task_type == "regression":
        y = pd.to_numeric(target, errors="coerce")
        valid = y.notna()
        X = X[valid.to_numpy()]
        y_arr = y.loc[valid].to_numpy(dtype=np.float32, copy=True)
        return LoadedDataset(X=X, y=y_arr, task_type=task_type, n_classes=0, class_labels=[])

    y_arr, labels = pd.factorize(target, sort=True)
    n_classes = int(len(labels))
    if task_type == "binary" and n_classes != 2:
        raise ValueError(f"Binary dataset must have exactly 2 classes, got {n_classes}: {csv_path}")
    if task_type == "multiclass" and n_classes <= 2:
        raise ValueError(f"Multiclass dataset must have >2 classes, got {n_classes}: {csv_path}")
    return LoadedDataset(
        X=X,
        y=y_arr.astype(np.int64, copy=False),
        task_type=task_type,
        n_classes=n_classes,
        class_labels=[str(item) for item in labels],
    )


def objective_for_task(task_type: str, args: argparse.Namespace) -> str:
    if task_type == "regression":
        return "reg:squarederror"
    if task_type == "binary":
        return args.binary_objective
    return args.multiclass_objective


def build_estimator(
    xgb: Any,
    model_name: str,
    params: Dict[str, Any],
    args: argparse.Namespace,
    task_type: str,
    n_classes: int,
) -> Any:
    common = {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": 1.0,
        "max_bin": int(params.get("max_bin", 255)),
        "tree_method": "hist",
        "device": "cpu",
        "objective": objective_for_task(task_type, args),
        "random_state": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "verbosity": 0,
    }
    if "reg_lambda" in params:
        common["reg_lambda"] = float(params["reg_lambda"])
    is_classifier = task_type in {"binary", "multiclass"}
    if is_classifier:
        common["eval_metric"] = "logloss" if task_type == "binary" else "mlogloss"
        if task_type == "multiclass":
            common["num_class"] = int(n_classes)

    if model_name == "xgb_cpu":
        common["reg_lambda"] = float(params.get("reg_lambda", args.xgb_reg_lambda))
        common.update(
            {
                "reg_alpha": 0.0,
                "grow_policy": "depthwise",
                "colsample_bylevel": 1.0,
                "colsample_bynode": 1.0,
            }
        )
        return xgb.XGBClassifier(**common) if is_classifier else xgb.XGBRegressor(**common)

    if model_name in {"current_metagb", "current_metagb_pred0"}:
        common.update(
            {
                "enable_metagb": True,
                "metagb_pred_colsample_bytree": float(params.get("pred_feature_fraction", 1.0)),
                "metagb_hist_subtract": bool(args.metagb_hist_subtract),
                "metagb_score_agg": int(params.get("metagb_score_agg", args.metagb_score_agg)),
                "metagb_lambda_l2": float(params.get("metagb_lambda_l2", args.metagb_lambda_l2)),
                "metagb_min_data_in_leaf": 1,
                "metagb_cond_pred_share": False,
                "metagb_leaf_only_use_bias": False,
            }
        )
        if is_classifier:
            common["metagb_pred_value_type"] = args.metagb_pred_value_type
            return xgb.XGBClassifier(**common)
        return xgb.XGBRegressor(**common)

    raise ValueError(f"Unknown model: {model_name}")


def fit_predict_timed(model: Any, X_train: Any, y_train: Any, X_eval: Any, task_type: str, n_classes: int) -> Tuple[Any, Any, float, float]:
    start = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    if task_type == "regression":
        pred = model.predict(X_eval)
        proba = None
    else:
        try:
            proba = normalize_proba(model.predict_proba(X_eval), n_classes)
            pred = labels_from_proba(proba) if proba is not None else np.asarray(model.predict(X_eval), dtype=np.int64)
        except Exception:
            proba = None
            pred = np.asarray(model.predict(X_eval), dtype=np.int64)
    predict_time = time.perf_counter() - start
    return pred, proba, fit_time, predict_time


def split_indices(data: LoadedDataset, args: argparse.Namespace) -> List[Tuple[Any, Any]]:
    if data.task_type == "regression":
        splitter = KFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        return list(splitter.split(data.X))
    splitter = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    return list(splitter.split(data.X, data.y))


def inner_split_indices(data: LoadedDataset, X_train: Any, y_train: Any, args: argparse.Namespace, outer_index: int) -> List[Tuple[Any, Any]]:
    if data.task_type == "regression":
        splitter = KFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_index)
        return list(splitter.split(X_train))
    splitter = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_index)
    return list(splitter.split(X_train, y_train))


def selected_outer_fold_indices(args: argparse.Namespace) -> List[int]:
    if args.run_all_outer_folds or args.mode == "full":
        return list(range(args.outer_folds))
    if args.outer_fold_index < 0 or args.outer_fold_index >= args.outer_folds:
        raise ValueError("--outer-fold-index is zero-based and must be in [0, outer_folds)")
    return [args.outer_fold_index]


def evaluate_metrics(task_type: str, y_true: Any, pred: Any, proba: Any, n_classes: int) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred, proba, task_type, n_classes)


def run_outer_task(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    args = argparse.Namespace(**task["args"])
    task_log = Path(task["task_log"])
    append_log(task_log, f"start task_id={task['task_id']}")
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    data = load_dataset(Path(task["dataset_path"]), task["task_type"], args.categorical_handling)
    primary_name = primary_metric_name(task["task_type"])
    direction = primary_metric_direction(task["task_type"])
    objective = objective_for_task(task["task_type"], args)
    append_log(
        task_log,
        "config "
        f"task_type={task['task_type']} dataset={task['dataset']} model={task['model']} "
        f"outer_fold={task['outer_index'] + 1}/{args.outer_folds} inner_folds={args.inner_folds} "
        f"candidates={len(task['candidates'])} objective={objective} primary_metric={primary_name} "
        f"direction={direction} rows={len(data.y)} features={data.X.shape[1]} classes={data.n_classes}",
    )

    outer_splits = split_indices(data, args)
    train_idx, test_idx = outer_splits[task["outer_index"]]
    X_train_outer, X_test_outer = data.X[train_idx], data.X[test_idx]
    y_train_outer, y_test_outer = data.y[train_idx], data.y[test_idx]
    inner_splits = inner_split_indices(data, X_train_outer, y_train_outer, args, task["outer_index"])
    candidates = task["candidates"]
    append_log(
        task_log,
        f"split train_rows={len(y_train_outer)} test_rows={len(y_test_outer)} "
        f"inner_splits={len(inner_splits)}",
    )
    rows: List[CandidateRow] = []
    scores: List[Tuple[float, float, float, float, int, str]] = []
    task_start = time.perf_counter()

    for candidate_index, params in enumerate(candidates, start=1):
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
            X_inner_train = X_train_outer[inner_train_idx]
            y_inner_train = y_train_outer[inner_train_idx]
            X_inner_val = X_train_outer[inner_val_idx]
            y_inner_val = y_train_outer[inner_val_idx]
            try:
                estimator = build_estimator(xgb, task["model"], params, args, task["task_type"], data.n_classes)
                pred_val, proba_val, fit_time, pred_time = fit_predict_timed(
                    estimator, X_inner_train, y_inner_train, X_inner_val, task["task_type"], data.n_classes
                )
                metrics = evaluate_metrics(task["task_type"], y_inner_val, pred_val, proba_val, data.n_classes)
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
            CandidateRow(
                run_id=args.run_id,
                task_type=task["task_type"],
                dataset=task["dataset"],
                model=task["model"],
                outer_fold=task["outer_index"] + 1,
                candidate_index=candidate_index,
                candidate_id=str(params.get("candidate_id", candidate_index)),
                backbone_id=str(params.get("backbone_id", "")),
                source=str(params.get("source", "")),
                pred_feature_fraction=float(params.get("pred_feature_fraction", float("nan"))),
                metagb_score_agg=float(params.get("metagb_score_agg", float("nan"))),
                selected=False,
                status=status,
                primary_metric=primary_name,
                primary_metric_direction=direction,
                val_primary_metric_mean=val_primary_mean,
                val_primary_metric_std=val_primary_std,
                val_rmse_mean=finite_mean(val_rmses),
                val_logloss_mean=finite_mean(val_loglosses),
                val_accuracy_mean=finite_mean(val_accuracies),
                val_balanced_accuracy_mean=finite_mean(val_balanced_accuracies),
                val_auc_mean=finite_mean(val_aucs),
                inner_train_time_sec=inner_train_time,
                inner_predict_time_sec=inner_predict_time,
                final_train_time_sec=float("nan"),
                final_predict_time_sec=float("nan"),
                test_primary_metric=float("nan"),
                test_rmse=float("nan"),
                test_logloss=float("nan"),
                test_accuracy=float("nan"),
                test_balanced_accuracy=float("nan"),
                test_auc=float("nan"),
                params_json=json.dumps(params, sort_keys=True),
            )
        )
        if candidate_index == 1 or candidate_index == len(candidates) or candidate_index % max(1, args.progress_every) == 0 or status != "ok":
            append_log(
                task_log,
                f"candidate {candidate_index}/{len(candidates)} {primary_name}={val_primary_mean:.6g} status={status}",
            )

    best_index = select_best_index(scores, direction)
    if best_index < 0:
        return [asdict(row) for row in rows], {
            "run_id": args.run_id,
            "task_type": task["task_type"],
            "dataset": task["dataset"],
            "model": task["model"],
            "outer_fold": task["outer_index"] + 1,
            "candidate_count": len(candidates),
            "status": "no_valid_candidate",
        }

    best_params = candidates[best_index - 1]
    best_score = scores[best_index - 1]
    append_log(
        task_log,
        f"selected candidate_index={best_index} candidate_id={best_params.get('candidate_id', best_index)} "
        f"val_{primary_name}={best_score[0]:.6g} val_std={best_score[1]:.6g}",
    )
    try:
        final_estimator = build_estimator(xgb, task["model"], best_params, args, task["task_type"], data.n_classes)
        test_pred, test_proba, final_train_time, final_predict_time = fit_predict_timed(
            final_estimator, X_train_outer, y_train_outer, X_test_outer, task["task_type"], data.n_classes
        )
        test_metrics = evaluate_metrics(task["task_type"], y_test_outer, test_pred, test_proba, data.n_classes)
        test_primary = primary_metric_value(task["task_type"], test_metrics)
        final_status = "ok"
    except Exception as exc:
        final_train_time = float("nan")
        final_predict_time = float("nan")
        test_metrics = {"rmse": float("nan"), "logloss": float("nan"), "accuracy": float("nan"), "balanced_accuracy": float("nan"), "auc": float("nan")}
        test_primary = float("nan")
        final_status = f"failed:{type(exc).__name__}:{exc}"

    for row in rows:
        if row.candidate_index == best_index:
            row.selected = True
            row.status = final_status
            row.final_train_time_sec = final_train_time
            row.final_predict_time_sec = final_predict_time
            row.test_primary_metric = test_primary
            row.test_rmse = test_metrics.get("rmse", float("nan"))
            row.test_logloss = test_metrics.get("logloss", float("nan"))
            row.test_accuracy = test_metrics.get("accuracy", float("nan"))
            row.test_balanced_accuracy = test_metrics.get("balanced_accuracy", float("nan"))
            row.test_auc = test_metrics.get("auc", float("nan"))
            break

    summary = {
        "run_id": args.run_id,
        "task_type": task["task_type"],
        "dataset": task["dataset"],
        "model": task["model"],
        "outer_fold": task["outer_index"] + 1,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "objective": objective,
        "primary_metric": primary_name,
        "primary_metric_direction": direction,
        "candidate_count": len(candidates),
        "selected_candidate_index": best_index,
        "selected_candidate_id": str(best_params.get("candidate_id", best_index)),
        "selected_backbone_id": str(best_params.get("backbone_id", "")),
        "selected_pred_feature_fraction": float(best_params.get("pred_feature_fraction", float("nan"))),
        "selected_metagb_score_agg": float(best_params.get("metagb_score_agg", float("nan"))),
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
    append_log(
        task_log,
        f"finish best={best_index} test_{primary_name}={test_primary:.6g} "
        f"final_train_sec={final_train_time:.6g} final_predict_sec={final_predict_time:.6g} "
        f"elapsed_sec={summary['task_elapsed_sec']:.6g} status={final_status}",
    )
    return [asdict(row) for row in rows], summary


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
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {"tasks": {}}
    payload.setdefault("tasks", {})[task_id] = status
    tasks = payload.get("tasks", {})
    payload["completed"] = sum(1 for value in tasks.values() if value != "pending")
    payload["failed"] = sum(1 for value in tasks.values() if str(value).startswith("failed"))
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, payload)


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
    grouped = df.groupby(["task_type", "dataset", "model"], as_index=False)[available].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.to_flat_index()]
    return grouped.to_dict(orient="records")


def build_pred_fraction_analysis(candidate_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidate_rows:
        return []
    df = pd.DataFrame(candidate_rows)
    df = df[(df["model"] == "current_metagb") & df["pred_feature_fraction"].notna()]
    if df.empty:
        return []
    cols = ["val_primary_metric_mean", "inner_train_time_sec", "inner_predict_time_sec"]
    grouped = df.groupby(
        ["task_type", "dataset", "outer_fold", "pred_feature_fraction", "metagb_score_agg"],
        as_index=False,
    )[cols].agg(["mean", "std", "min"])
    grouped.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.to_flat_index()]
    return grouped.to_dict(orient="records")


def sort_tasks_for_dataset_completion(tasks: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    model_order = {model: index for index, model in enumerate(args.models)}
    dataset_first_position: Dict[Tuple[str, str], int] = {}
    for position, task in enumerate(tasks):
        dataset_key = (task["task_type"], task["dataset"])
        dataset_first_position.setdefault(dataset_key, position)

    def task_sort_key(position_and_task: Tuple[int, Dict[str, Any]]) -> Tuple[Any, ...]:
        position, task = position_and_task
        dataset_key = (task["task_type"], task["dataset"])
        instance_count = DATASET_INSTANCE_COUNTS.get(dataset_key)
        is_unknown_dataset = instance_count is None
        return (
            is_unknown_dataset,
            instance_count if instance_count is not None else sys.maxsize,
            dataset_first_position[dataset_key],
            int(task["outer_index"]),
            model_order.get(task["model"], sys.maxsize),
            position,
        )

    return [task for _, task in sorted(enumerate(tasks), key=task_sort_key)]


def create_tasks(args: argparse.Namespace, candidates_by_model: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        dataset_dir = dataset_dir_for_task(args, task_type)
        for dataset_name in select_datasets(args, task_type):
            dataset_path = dataset_dir / dataset_name
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            for outer_index in selected_outer_fold_indices(args):
                for model in args.models:
                    task_id = f"{safe_name(task_type)}__{safe_name(dataset_name)}__{safe_name(model)}__out{outer_index + 1}"
                    tasks.append(
                        {
                            "task_id": task_id,
                            "task_type": task_type,
                            "dataset": dataset_name,
                            "dataset_path": str(dataset_path),
                            "model": model,
                            "outer_index": outer_index,
                            "candidates": candidates_by_model[model],
                            "task_log": str(Path(args.logs_dir) / "tasks" / f"{task_id}.log"),
                            "args": vars(args),
                        }
                    )
    return sort_tasks_for_dataset_completion(tasks, args)


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    search_space = load_search_space(Path(args.search_space_json))
    config_path = Path(args.xgb_config) if args.xgb_config else None
    selected_candidates, candidate_warnings = candidate_sets(
        search_space, args.max_candidates, args.seed, config_path, args.models
    )
    write_json(Path(args.results_dir) / "candidate_configs.json", candidate_config_payload(selected_candidates))
    write_json(Path(args.results_dir) / "parameter_audit.json", build_parameter_audit(args, search_space, candidate_warnings))

    tasks = create_tasks(args, selected_candidates)
    task_status_path = Path(args.logs_dir) / "task_status.json"
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
                        "task_type": task["task_type"],
                        "dataset": task["dataset"],
                        "model": task["model"],
                        "outer_fold": task["outer_index"] + 1,
                        "status": f"failed:{type(exc).__name__}:{exc}",
                    }
                candidate_rows.extend(rows)
                summary_rows.append(summary)
                update_task_status_file(task_status_path, task["task_id"], str(summary.get("status")))
                print(f"[{index}/{len(tasks)}] {task['task_id']} status={summary.get('status')}", flush=True)

    candidate_rows.sort(key=lambda row: (row.get("task_type", ""), row.get("dataset", ""), row.get("model", ""), row.get("outer_fold", 0), row.get("candidate_index", 0)))
    summary_rows.sort(key=lambda row: (row.get("task_type", ""), row.get("dataset", ""), row.get("model", ""), row.get("outer_fold", 0)))
    aggregate_rows = build_aggregate(summary_rows)
    pred_fraction_rows = build_pred_fraction_analysis(candidate_rows)

    results_dir = Path(args.results_dir)
    write_csv(results_dir / "candidates.csv", candidate_rows)
    write_csv(results_dir / "summary.csv", summary_rows)
    write_csv(results_dir / "aggregate.csv", aggregate_rows)
    write_csv(results_dir / "pred_fraction_analysis.csv", pred_fraction_rows)

    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.perf_counter() - start,
        "task_types": args.task_types,
        "models": args.models,
        "max_candidates": args.max_candidates,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "run_all_outer_folds": bool(args.run_all_outer_folds or args.mode == "full"),
        "n_jobs": args.n_jobs,
        "task_workers": args.task_workers,
        "xgb_config": str(config_path) if config_path else None,
        "candidate_warnings": candidate_warnings,
        "dataset_dirs": {
            "regression": str(Path(args.regression_dir).expanduser().resolve()),
            "binary": str(Path(args.binary_dir).expanduser().resolve()),
            "multiclass": str(Path(args.multiclass_dir).expanduser().resolve()),
        },
        "selected_datasets": {
            "regression": select_datasets(args, "regression") if "regression" in args.task_types else [],
            "binary": select_datasets(args, "binary") if "binary" in args.task_types else [],
            "multiclass": select_datasets(args, "multiclass") if "multiclass" in args.task_types else [],
        },
        "skip_datasets": list(getattr(args, "skip_datasets", []) or []),
        "skip_task_datasets": {
            "regression": list(getattr(args, "skip_regression_datasets", []) or []),
            "binary": list(getattr(args, "skip_binary_datasets", []) or []),
            "multiclass": list(getattr(args, "skip_multiclass_datasets", []) or []),
        },
        "xgboost_python_package": str(Path(args.xgboost_python_package).expanduser().resolve()),
        "results_dir": str(results_dir),
        "logs_dir": str(Path(args.logs_dir)),
        "summary_rows": len(summary_rows),
        "candidate_rows": len(candidate_rows),
    }
    write_json(results_dir / "run_metadata.json", metadata)
    return metadata
