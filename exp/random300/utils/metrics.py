from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)


PRIMARY_METRICS = {
    "regression": ("rmse", "minimize"),
    "binary": ("auc", "maximize"),
    "multiclass": ("logloss", "minimize"),
}


def primary_metric_name(task_type: str) -> str:
    return PRIMARY_METRICS[task_type][0]


def primary_metric_direction(task_type: str) -> str:
    return PRIMARY_METRICS[task_type][1]


def rmse(y_true: Any, y_pred: Any) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def normalize_proba(proba: Any, n_classes: int) -> Any:
    proba = np.asarray(proba)
    if proba.ndim == 1 and n_classes == 2:
        positive = np.clip(proba.astype(np.float64), 1e-15, 1.0 - 1e-15)
        return np.column_stack([1.0 - positive, positive])
    if proba.ndim == 2 and proba.shape[1] == n_classes:
        out = proba.astype(np.float64, copy=False)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return np.clip(out / row_sums, 1e-15, 1.0)
    return None


def labels_from_proba(proba: Any) -> Any:
    return np.asarray(proba).argmax(axis=1).astype(np.int64, copy=False)


def safe_logloss(y_true: Any, proba: Any, n_classes: int) -> float:
    if proba is None:
        return float("inf")
    try:
        return float(log_loss(y_true, proba, labels=list(range(n_classes))))
    except ValueError:
        return float("inf")


def safe_auc(y_true: Any, proba: Any, task_type: str) -> float:
    if proba is None or task_type != "binary":
        return float("nan")
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, proba[:, 1]))
    except ValueError:
        return float("nan")


def classification_metrics(y_true: Any, pred: Any, proba: Any, task_type: str, n_classes: int) -> Dict[str, float]:
    return {
        "logloss": safe_logloss(y_true, proba, n_classes),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": safe_auc(y_true, proba, task_type),
    }


def regression_metrics(y_true: Any, pred: Any) -> Dict[str, float]:
    return {"rmse": rmse(y_true, pred)}


def primary_metric_value(task_type: str, metrics: Dict[str, float]) -> float:
    name = primary_metric_name(task_type)
    return float(metrics[name])


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


def select_best_index(scores: List[Tuple[float, float, float, float, int, str]], direction: str) -> int:
    valid = [score for score in scores if math.isfinite(score[0])]
    if not valid:
        return -1
    if direction == "maximize":
        valid.sort(key=lambda item: (-item[0], item[2] + item[3], item[4]))
    else:
        valid.sort(key=lambda item: (item[0], item[2] + item[3], item[4]))
    return int(valid[0][4])
