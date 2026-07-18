from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def finite_std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return 0.0 if finite else float("nan")
    return float(np.std(finite))


def select_min_rmse(scores: list[tuple[float, float, int, str]]) -> int:
    valid = [score for score in scores if math.isfinite(score[0])]
    if not valid:
        return -1
    valid.sort(key=lambda item: (item[0], item[1], item[2]))
    return int(valid[0][2])

