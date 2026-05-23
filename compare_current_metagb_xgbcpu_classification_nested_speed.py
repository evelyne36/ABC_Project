from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("METAGB_PROJECT_ROOT", REPO_ROOT)).expanduser().resolve()
XGBOOST_PYTHON_PACKAGE = Path(
    os.environ.get("METAGB_XGBOOST_PYTHON_PACKAGE", PROJECT_ROOT / "xgboost-master" / "python-package")
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("METAGB_OUTPUT_DIR", PROJECT_ROOT / "results_current_metagb_vs_xgbcpu_classification")
).expanduser()


def default_dataset_dirs() -> List[Path]:
    raw_dirs = os.environ.get("METAGB_DATASET_DIRS")
    if raw_dirs:
        return [Path(item).expanduser() for item in raw_dirs.split(os.pathsep) if item]
    return [
        PROJECT_ROOT / "Exp1_54CV_ordinal" / "dataset" / "tabArena_binary",
        PROJECT_ROOT / "Exp1_54CV_ordinal" / "dataset" / "tabArena_multiclass",
        PROJECT_ROOT / "dataset" / "tabArena_binary",
        PROJECT_ROOT / "dataset" / "tabArena_multiclass",
    ]


DEFAULT_DATASET_DIRS = default_dataset_dirs()
CLASSIFICATION_DATASETS = ["diabetes.csv", "anneal.csv"]
#CLASSIFICATION_DATASETS = ["anneal.csv","diabetes.csv"]
CLASSIFICATION_DATASETS = ["anneal.csv"]

np = None
pd = None
StratifiedKFold = None
accuracy_score = None
balanced_accuracy_score = None
log_loss = None
roc_auc_score = None


def ensure_runtime_dependencies() -> None:
    global np, pd, StratifiedKFold
    global accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
    if np is not None:
        return

    import numpy as numpy_module
    import pandas as pandas_module
    from sklearn.metrics import (
        accuracy_score as sklearn_accuracy_score,
        balanced_accuracy_score as sklearn_balanced_accuracy_score,
        log_loss as sklearn_log_loss,
        roc_auc_score as sklearn_roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold as stratified_kfold_class

    np = numpy_module
    pd = pandas_module
    StratifiedKFold = stratified_kfold_class
    accuracy_score = sklearn_accuracy_score
    balanced_accuracy_score = sklearn_balanced_accuracy_score
    log_loss = sklearn_log_loss
    roc_auc_score = sklearn_roc_auc_score


def import_local_xgboost(python_package: Path):
    python_package = python_package.resolve()
    if not python_package.exists():
        raise FileNotFoundError(f"Local XGBoost python package not found: {python_package}")
    for module_name in list(sys.modules):
        if module_name == "xgboost" or module_name.startswith("xgboost."):
            del sys.modules[module_name]
    if str(python_package) not in sys.path:
        sys.path.insert(0, str(python_package))
    import xgboost as xgb

    return xgb


def default_cpu_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def sample_log_uniform(rng: Any, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_log_uniform_int(rng: Any, low: int, high: int) -> int:
    sampled = np.exp(rng.uniform(np.log(low), np.log(high)))
    return int(np.clip(np.round(sampled), low, high))


def attach_n_estimators(candidates: List[Dict[str, Any]], rng: Any) -> None:
    values = [200, 400, 600, 800]
    assigned = np.tile(values, int(np.ceil(len(candidates) / len(values))))[: len(candidates)]
    rng.shuffle(assigned)
    for params, n_estimators in zip(candidates, assigned):
        params["n_estimators"] = int(n_estimators)


def generate_exp1_style_candidates(num_candidates: int, seed: int) -> List[Dict[str, Any]]:
    ensure_runtime_dependencies()
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    rng = np.random.default_rng(seed + 3000)
    pred_feature_classes = [round(i * 0.1, 1) for i in range(11)]
    candidates: List[Dict[str, Any]] = []

    per_class = max(1, num_candidates // len(pred_feature_classes))
    for pred_feature_fraction in pred_feature_classes:
        for _ in range(per_class):
            if len(candidates) >= num_candidates:
                break
            candidates.append(
                {
                    "pred_feature_fraction": float(pred_feature_fraction),
                    "learning_rate": sample_log_uniform(rng, 0.005, 0.1),
                    "max_depth": sample_log_uniform_int(rng, 4, 10),
                    "min_child_weight": sample_log_uniform(rng, 0.001, 5.0),
                    "subsample": float(rng.uniform(0.6, 1.0)),
                    "colsample": float(rng.uniform(0.6, 1.0)),
                    "version": 3,
                    "max_bin": 255,
                }
            )

    while len(candidates) < num_candidates:
        candidates.append(
            {
                "pred_feature_fraction": float(rng.choice(pred_feature_classes)),
                "learning_rate": sample_log_uniform(rng, 0.005, 0.1),
                "max_depth": sample_log_uniform_int(rng, 4, 10),
                "min_child_weight": sample_log_uniform(rng, 0.001, 5.0),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample": float(rng.uniform(0.6, 1.0)),
                "version": 3,
                "max_bin": 255,
            }
        )

    attach_n_estimators(candidates, np.random.default_rng(seed + 4000))
    return candidates


def find_dataset_path(dataset_name: str, dataset_dirs: Sequence[Path]) -> Path:
    for dataset_dir in dataset_dirs:
        candidate = dataset_dir / dataset_name
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in dataset_dirs)
    raise FileNotFoundError(f"Dataset not found: {dataset_name}. Searched: {searched}")


@dataclass
class LoadedDataset:
    X: Any
    y: Any
    n_classes: int
    task_type: str
    class_labels: List[str]


def load_dataset(csv_path: Path, categorical_handling: str) -> LoadedDataset:
    ensure_runtime_dependencies()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must contain at least one feature and one target column: {csv_path}")

    feature_df = df.iloc[:, :-1].copy()
    target = df.iloc[:, -1].copy()
    non_missing = target.notna()
    feature_df = feature_df.loc[non_missing]
    target = target.loc[non_missing]

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
        feature_df = feature_df.dropna()
        target = target.loc[feature_df.index]
    else:
        raise ValueError("categorical_handling must be one of: drop, onehot, ordinal")

    if len(feature_df) == 0:
        raise ValueError(f"No rows remain after preprocessing: {csv_path}")
    if feature_df.shape[1] == 0:
        raise ValueError(f"No feature columns remain after preprocessing: {csv_path}")

    y, labels = pd.factorize(target, sort=True)
    n_classes = int(len(labels))
    if n_classes < 2:
        raise ValueError(f"Classification target must contain at least two classes: {csv_path}")

    X = feature_df.to_numpy(dtype=np.float32, copy=True)
    y = y.astype(np.int64, copy=False)
    task_type = "binary" if n_classes == 2 else "multiclass"
    return LoadedDataset(X=X, y=y, n_classes=n_classes, task_type=task_type, class_labels=[str(item) for item in labels])


def objective_for_task(task_type: str, args: argparse.Namespace) -> str:
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
):
    objective = objective_for_task(task_type, args)
    common = {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample"]),
        "min_child_weight": max(1, int(round(float(params["min_child_weight"])))),
        "max_bin": int(params.get("max_bin", 255)),
        "tree_method": "hist",
        "device": "cpu",
        "objective": objective,
        "eval_metric": "logloss" if task_type == "binary" else "mlogloss",
        "random_state": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "verbosity": 0,
    }
    if task_type == "multiclass":
        common["num_class"] = int(n_classes)

    if model_name == "xgb_cpu":
        common["reg_lambda"] = float(args.xgb_reg_lambda)
        return xgb.XGBClassifier(**common)

    if model_name == "current_metagb":
        common.update(
            {
                "enable_metagb": True,
                "metagb_pred_colsample_bytree": float(params["pred_feature_fraction"]),
                "metagb_cond_pred_share": bool(args.metagb_cond_pred_share),
                "metagb_leaf_only_use_bias": bool(args.metagb_leaf_only_use_bias),
                "metagb_score_agg": int(args.metagb_score_agg),
                "metagb_lambda_l2": float(args.metagb_lambda_l2),
            }
        )
        return xgb.XGBClassifier(**common)

    raise ValueError(f"Unknown model: {model_name}")


def normalize_proba(proba: Any, n_classes: int) -> Any:
    proba = np.asarray(proba)
    if proba.ndim == 1 and n_classes == 2:
        positive = np.clip(proba.astype(np.float64), 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - positive, positive])
    if proba.ndim == 2 and proba.shape[1] == n_classes:
        proba = proba.astype(np.float64, copy=False)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return np.clip(proba / row_sums, 1e-15, 1.0)
    return None


def labels_from_proba(proba: Any) -> Any:
    return np.asarray(proba).argmax(axis=1).astype(np.int64, copy=False)


def fit_predict_timed(model: Any, X_train: Any, y_train: Any, X_eval: Any, n_classes: int) -> Tuple[Any, Any, float, float]:
    fit_start = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    fit_time = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    try:
        proba = normalize_proba(model.predict_proba(X_eval), n_classes)
        pred = labels_from_proba(proba) if proba is not None else np.asarray(model.predict(X_eval), dtype=np.int64)
    except Exception:  # noqa: BLE001
        proba = None
        pred = np.asarray(model.predict(X_eval), dtype=np.int64)
    predict_time = time.perf_counter() - predict_start
    return pred, proba, fit_time, predict_time


def safe_logloss(y_true: Any, proba: Any, n_classes: int) -> float:
    if proba is None:
        return float("inf")
    try:
        return float(log_loss(y_true, proba, labels=list(range(n_classes))))
    except ValueError:
        return float("inf")


def safe_auc(y_true: Any, proba: Any, task_type: str) -> float:
    if proba is None:
        return float("nan")
    try:
        if task_type == "binary":
            if len(np.unique(y_true)) < 2:
                return float("nan")
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float("nan")
    except ValueError:
        return float("nan")


def classification_metrics(y_true: Any, pred: Any, proba: Any, task_type: str, n_classes: int) -> Dict[str, float]:
    return {
        "logloss": safe_logloss(y_true, proba, n_classes),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": safe_auc(y_true, proba, task_type),
    }


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("inf"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(np.mean(finite)), float(np.std(finite))


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def primary_metric_name(task_type: str) -> str:
    return "auc" if task_type == "binary" else "logloss"


def primary_metric_direction(task_type: str) -> str:
    return "maximize" if task_type == "binary" else "minimize"


def primary_metric_value(task_type: str, metrics: Dict[str, float]) -> float:
    return float(metrics["auc"] if task_type == "binary" else metrics["logloss"])


@dataclass
class CandidateRow:
    dataset: str
    task_type: str
    n_classes: int
    objective: str
    model: str
    outer_fold: int
    candidate_index: int
    selected: bool
    status: str
    primary_metric: str
    primary_metric_direction: str
    val_primary_metric_mean: float
    val_primary_metric_std: float
    val_logloss_mean: float
    val_logloss_std: float
    val_accuracy_mean: float
    val_balanced_accuracy_mean: float
    val_auc_mean: float
    inner_train_time_sec: float
    inner_predict_time_sec: float
    final_train_time_sec: float
    final_predict_time_sec: float
    test_logloss: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_auc: float
    params_json: str


def selected_outer_fold_indices(args: argparse.Namespace) -> List[int]:
    if args.run_all_outer_folds or args.mode == "full":
        return list(range(args.outer_folds))
    if args.outer_fold_index < 0 or args.outer_fold_index >= args.outer_folds:
        raise ValueError("--outer-fold-index is zero-based and must be in [0, outer_folds)")
    return [args.outer_fold_index]


def run_benchmark(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_runtime_dependencies()
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    dataset_dirs = [Path(path).expanduser() for path in args.dataset_dirs]
    datasets = CLASSIFICATION_DATASETS if args.datasets == ["all"] else args.datasets
    candidates = generate_exp1_style_candidates(args.max_candidates, args.seed)
    rows: List[CandidateRow] = []
    summary_rows: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        dataset_path = find_dataset_path(dataset_name, dataset_dirs)
        data = load_dataset(dataset_path, args.categorical_handling)
        objective = objective_for_task(data.task_type, args)
        primary_name = primary_metric_name(data.task_type)
        primary_direction = primary_metric_direction(data.task_type)
        outer_kf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        outer_splits = list(outer_kf.split(data.X, data.y))

        for outer_index in selected_outer_fold_indices(args):
            train_index, test_index = outer_splits[outer_index]
            X_train_outer, X_test_outer = data.X[train_index], data.X[test_index]
            y_train_outer, y_test_outer = data.y[train_index], data.y[test_index]
            inner_kf = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_index)
            inner_splits = list(inner_kf.split(X_train_outer, y_train_outer))

            for model_name in args.models:
                fold_start = time.perf_counter()
                print(
                    f"[{model_name}] dataset={dataset_name}, task={data.task_type}, "
                    f"primary={primary_name}, outer_fold={outer_index + 1}/{args.outer_folds}, "
                    f"candidates={len(candidates)}",
                    flush=True,
                )
                candidate_scores: List[Tuple[float, float, float, float, int, str]] = []

                for candidate_index, params in enumerate(candidates, start=1):
                    val_primary_metrics: List[float] = []
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
                            estimator = build_estimator(
                                xgb, model_name, params, args, data.task_type, data.n_classes
                            )
                            pred_val, proba_val, fit_time, pred_time = fit_predict_timed(
                                estimator, X_inner_train, y_inner_train, X_inner_val, data.n_classes
                            )
                            metrics = classification_metrics(
                                y_inner_val, pred_val, proba_val, data.task_type, data.n_classes
                            )
                            val_primary_metrics.append(primary_metric_value(data.task_type, metrics))
                            val_loglosses.append(metrics["logloss"])
                            val_accuracies.append(metrics["accuracy"])
                            val_balanced_accuracies.append(metrics["balanced_accuracy"])
                            val_aucs.append(metrics["auc"])
                            inner_train_time += fit_time
                            inner_predict_time += pred_time
                        except Exception as exc:  # noqa: BLE001
                            status = f"failed:{type(exc).__name__}"
                            val_primary_metrics = [float("nan")]
                            val_loglosses = [float("inf")]
                            val_accuracies = [float("nan")]
                            val_balanced_accuracies = [float("nan")]
                            val_aucs = [float("nan")]
                            inner_train_time = 0.0
                            inner_predict_time = 0.0
                            break

                    val_primary_mean, val_primary_std = mean_std(val_primary_metrics)
                    val_logloss_mean, val_logloss_std = mean_std(val_loglosses)
                    val_accuracy_mean = finite_mean(val_accuracies)
                    val_balanced_accuracy_mean = finite_mean(val_balanced_accuracies)
                    val_auc_mean = finite_mean(val_aucs)
                    candidate_scores.append(
                        (
                            val_primary_mean,
                            val_primary_std,
                            inner_train_time,
                            inner_predict_time,
                            candidate_index,
                            status,
                        )
                    )
                    rows.append(
                        CandidateRow(
                            dataset=dataset_name,
                            task_type=data.task_type,
                            n_classes=data.n_classes,
                            objective=objective,
                            model=model_name,
                            outer_fold=outer_index + 1,
                            candidate_index=candidate_index,
                            selected=False,
                            status=status,
                            primary_metric=primary_name,
                            primary_metric_direction=primary_direction,
                            val_primary_metric_mean=val_primary_mean,
                            val_primary_metric_std=val_primary_std,
                            val_logloss_mean=val_logloss_mean,
                            val_logloss_std=val_logloss_std,
                            val_accuracy_mean=val_accuracy_mean,
                            val_balanced_accuracy_mean=val_balanced_accuracy_mean,
                            val_auc_mean=val_auc_mean,
                            inner_train_time_sec=inner_train_time,
                            inner_predict_time_sec=inner_predict_time,
                            final_train_time_sec=float("nan"),
                            final_predict_time_sec=float("nan"),
                            test_logloss=float("nan"),
                            test_accuracy=float("nan"),
                            test_balanced_accuracy=float("nan"),
                            test_auc=float("nan"),
                            params_json=json.dumps(params, sort_keys=True),
                        )
                    )
                    if (
                        candidate_index == 1
                        or candidate_index == len(candidates)
                        or candidate_index % args.progress_every == 0
                        or status != "ok"
                    ):
                        progress_message = f"  candidate {candidate_index}/{len(candidates)}: "
                        if data.task_type == "binary":
                            progress_message += (
                                f"val_auc={val_auc_mean:.6g}, "
                                f"val_logloss={val_logloss_mean:.6g}, "
                                f"val_acc={val_accuracy_mean:.6g}, "
                            )
                        else:
                            progress_message += f"val_logloss={val_logloss_mean:.6g}, "
                        progress_message += f"status={status}"
                        print(progress_message, flush=True)

                valid_scores = [score for score in candidate_scores if math.isfinite(score[0])]
                if not valid_scores:
                    summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "task_type": data.task_type,
                            "n_classes": data.n_classes,
                            "objective": objective,
                            "primary_metric": primary_name,
                            "primary_metric_direction": primary_direction,
                            "model": model_name,
                            "outer_fold": outer_index + 1,
                            "status": "no_valid_candidate",
                        }
                    )
                    continue

                if primary_direction == "maximize":
                    valid_scores.sort(key=lambda item: (-item[0], item[2] + item[3], item[4]))
                else:
                    valid_scores.sort(key=lambda item: (item[0], item[2] + item[3], item[4]))
                best_primary_mean, best_primary_std, best_inner_train, best_inner_predict, best_index, _ = valid_scores[0]
                best_params = candidates[best_index - 1]

                try:
                    final_estimator = build_estimator(xgb, model_name, best_params, args, data.task_type, data.n_classes)
                    test_pred, test_proba, final_train_time, final_predict_time = fit_predict_timed(
                        final_estimator, X_train_outer, y_train_outer, X_test_outer, data.n_classes
                    )
                    test_metrics = classification_metrics(
                        y_test_outer, test_pred, test_proba, data.task_type, data.n_classes
                    )
                    final_status = "ok"
                except Exception as exc:  # noqa: BLE001
                    final_train_time = float("nan")
                    final_predict_time = float("nan")
                    test_metrics = {
                        "logloss": float("nan"),
                        "accuracy": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "auc": float("nan"),
                    }
                    final_status = f"failed:{type(exc).__name__}"
                test_primary_metric = primary_metric_value(data.task_type, test_metrics)

                best_row: Optional[CandidateRow] = None
                for row in reversed(rows):
                    if (
                        row.dataset == dataset_name
                        and row.model == model_name
                        and row.outer_fold == outer_index + 1
                        and row.candidate_index == best_index
                    ):
                        row.selected = True
                        row.final_train_time_sec = final_train_time
                        row.final_predict_time_sec = final_predict_time
                        row.test_logloss = test_metrics["logloss"]
                        row.test_accuracy = test_metrics["accuracy"]
                        row.test_balanced_accuracy = test_metrics["balanced_accuracy"]
                        row.test_auc = test_metrics["auc"]
                        row.status = final_status
                        best_row = row
                        break
                if best_row is None:
                    raise RuntimeError("Internal error: selected candidate row was not recorded")

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "task_type": data.task_type,
                        "n_classes": data.n_classes,
                        "objective": objective,
                        "primary_metric": primary_name,
                        "primary_metric_direction": primary_direction,
                        "model": model_name,
                        "outer_fold": outer_index + 1,
                        "outer_folds": args.outer_folds,
                        "inner_folds": args.inner_folds,
                        "selected_candidate_index": best_index,
                        "val_primary_metric_mean": best_primary_mean,
                        "val_primary_metric_std": best_primary_std,
                        "test_primary_metric": test_primary_metric,
                        "val_logloss_mean": best_row.val_logloss_mean,
                        "val_logloss_std": best_row.val_logloss_std,
                        "val_auc_mean": best_row.val_auc_mean,
                        "inner_train_time_sec": best_inner_train,
                        "inner_predict_time_sec": best_inner_predict,
                        "final_train_time_sec": final_train_time,
                        "final_predict_time_sec": final_predict_time,
                        "test_logloss": test_metrics["logloss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_auc": test_metrics["auc"],
                        "total_train_time_sec": best_inner_train + final_train_time,
                        "total_predict_time_sec": best_inner_predict + final_predict_time,
                        "status": final_status,
                        "params_json": json.dumps(best_params, sort_keys=True),
                    }
                )
                done_message = (
                    f"[{model_name}] done dataset={dataset_name}, outer_fold={outer_index + 1}, "
                    f"best={best_index}, "
                )
                if data.task_type == "binary":
                    done_message += (
                        f"val_auc={best_primary_mean:.6g}, "
                        f"test_auc={test_metrics['auc']:.6g}, "
                        f"test_logloss={test_metrics['logloss']:.6g}, "
                        f"test_acc={test_metrics['accuracy']:.6g}, "
                    )
                else:
                    done_message += (
                        f"val_logloss={best_primary_mean:.6g}, "
                        f"test_logloss={test_metrics['logloss']:.6g}, "
                    )
                done_message += f"elapsed={time.perf_counter() - fold_start:.1f}s"
                print(done_message, flush=True)

    return [asdict(row) for row in rows], summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not summary_rows:
        return []
    ensure_runtime_dependencies()
    df = pd.DataFrame(summary_rows)
    metric_cols = [
        "val_primary_metric_mean",
        "test_primary_metric",
        "val_logloss_mean",
        "val_auc_mean",
        "test_logloss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auc",
        "inner_train_time_sec",
        "inner_predict_time_sec",
        "final_train_time_sec",
        "final_predict_time_sec",
        "total_train_time_sec",
        "total_predict_time_sec",
    ]
    available = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(["dataset", "task_type", "model"], as_index=False)[available].agg(["mean", "std"])
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    return grouped.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-CV classification speed/performance comparison for embedded MetaGB and CPU XGBoost."
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_DATASET_DIRS],
        help="Directories containing tabArena_binary and tabArena_multiclass csv files.",
    )
    parser.add_argument("--xgboost-python-package", default=str(XGBOOST_PYTHON_PACKAGE))
    # parser.add_argument("--models", nargs="+", default=["xgb_cpu", "current_metagb"])
    parser.add_argument("--models", nargs="+", default=["current_metagb", "xgb_cpu"])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--outer-fold-index", type=int, default=0, help="zero-based outer fold for quick mode")
    parser.add_argument("--run-all-outer-folds", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means quick=5, full=200")
    parser.add_argument("--categorical-handling", choices=["drop", "onehot", "ordinal"], default="ordinal")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=default_cpu_jobs())
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--binary-objective", choices=["binary:logistic", "binary:logitraw"], default="binary:logistic")
    parser.add_argument("--multiclass-objective", choices=["multi:softprob", "multi:softmax"], default="multi:softprob")
    parser.add_argument("--xgb-reg-lambda", type=float, default=0.0)
    parser.add_argument("--metagb-lambda-l2", type=float, default=0.0)
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--metagb-cond-pred-share", action="store_true")
    parser.add_argument("--metagb-leaf-only-use-bias", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.datasets == ["all"]:
        args.run_all_outer_folds = args.mode == "full" or args.run_all_outer_folds
    if args.mode == "full":
        args.run_all_outer_folds = True
    if args.max_candidates <= 0:
        args.max_candidates = 5 if args.mode == "quick" else 200

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows, summary_rows = run_benchmark(args)
    aggregate_rows = build_aggregate(summary_rows)

    candidate_csv = output_dir / "current_metagb_vs_xgbcpu_classification_nested_candidates.csv"
    summary_csv = output_dir / "current_metagb_vs_xgbcpu_classification_nested_summary.csv"
    aggregate_csv = output_dir / "current_metagb_vs_xgbcpu_classification_nested_aggregate.csv"
    metadata_json = output_dir / "current_metagb_vs_xgbcpu_classification_nested_metadata.json"

    write_csv(candidate_csv, candidate_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(aggregate_csv, aggregate_rows)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": args.mode,
                "datasets": CLASSIFICATION_DATASETS if args.datasets == ["all"] else args.datasets,
                "models": args.models,
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "run_all_outer_folds": bool(args.run_all_outer_folds),
                "max_candidates": args.max_candidates,
                "categorical_handling": args.categorical_handling,
                "binary_objective": args.binary_objective,
                "multiclass_objective": args.multiclass_objective,
                "xgboost_python_package": args.xgboost_python_package,
                "candidate_csv": str(candidate_csv),
                "summary_csv": str(summary_csv),
                "aggregate_csv": str(aggregate_csv),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"candidate rows: {len(candidate_rows)}")
    print(f"summary rows: {len(summary_rows)}")
    print(f"aggregate rows: {len(aggregate_rows)}")
    print(f"outputs: {candidate_csv}, {summary_csv}, {aggregate_csv}, {metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
