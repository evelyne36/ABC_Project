from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from random300.utils.runner import (
    build_estimator,
    evaluate_metrics,
    fit_predict_timed,
    import_local_xgboost,
)

from .contracts import SHARED_BACKBONE_KEYS
from .io_utils import canonical_hash, file_sha256
from .protocols import VariantSpec


@dataclass(frozen=True)
class BackboneSelection:
    candidate_index: int
    candidate_id: str
    validation_primary_metric: float
    validation_primary_metric_std: float
    validation_fit_sec: float
    validation_predict_sec: float
    params: Dict[str, Any]


def primary_metric_name(task_type: str) -> str:
    return "rmse" if task_type == "regression" else "logloss"


def primary_metric_direction(task_type: str) -> str:
    del task_type
    return "minimize"


def primary_value(task_type: str, metrics: Dict[str, float]) -> float:
    return float(metrics[primary_metric_name(task_type)])


def runtime_args(
    *,
    model_seed: int,
    n_jobs: int,
    categorical_handling: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        binary_objective="binary:logistic",
        multiclass_objective="multi:softprob",
        seed=int(model_seed),
        n_jobs=int(n_jobs),
        xgb_reg_lambda=1.0,
        crossgb_lambda_l2=1.0,
        crossgb_hist_subtract=False,
        crossgb_score_agg=0,
        categorical_handling=categorical_handling,
    )


def crossgb_params_from_backbone(
    backbone: Dict[str, Any],
    variant: VariantSpec,
) -> Dict[str, Any]:
    if variant.variant_id == "xgb":
        return dict(backbone)
    if variant.pred_fraction is None or variant.score_agg is None:
        raise ValueError(f"CrossGB variant is missing private parameters: {variant}")
    params = {
        key: value
        for key, value in backbone.items()
        if key not in {"candidate_id", "source", "reg_lambda"}
    }
    params.update(
        {
            "candidate_id": variant.variant_id,
            "source": "xgb_selected_fixed_backbone",
            "backbone_id": backbone.get("backbone_id", backbone.get("candidate_id", "")),
            "xgb_selected_candidate_id": backbone.get("candidate_id", ""),
            "crossgb_lambda_l2": float(
                backbone.get("reg_lambda", backbone.get("lambda_l2", 1.0))
            ),
            "pred_feature_fraction": float(variant.pred_fraction),
            "crossgb_score_agg": int(variant.score_agg),
            "crossgb_leaf_only_use_bias": bool(variant.leaf_only_use_bias),
        }
    )
    return params


def assert_shared_backbone(
    backbone: Dict[str, Any],
    crossgb_params: Dict[str, Any],
) -> None:
    for key in SHARED_BACKBONE_KEYS:
        if backbone.get(key) != crossgb_params.get(key):
            raise AssertionError(
                f"Shared backbone mismatch for {key}: "
                f"{backbone.get(key)} != {crossgb_params.get(key)}"
            )
    xgb_lambda = float(backbone.get("reg_lambda", backbone.get("lambda_l2", 1.0)))
    crossgb_lambda = float(crossgb_params["crossgb_lambda_l2"])
    if xgb_lambda != crossgb_lambda:
        raise AssertionError(
            f"L2 mismatch: XGB reg_lambda={xgb_lambda}, CrossGB lambda={crossgb_lambda}"
        )


def build_variant_estimator(
    xgb: Any,
    variant: VariantSpec,
    backbone: Dict[str, Any],
    args: argparse.Namespace,
    task_type: str,
    n_classes: int,
) -> Tuple[Any, Dict[str, Any]]:
    if variant.variant_id == "xgb":
        params = dict(backbone)
        estimator = build_estimator(
            xgb,
            "xgb_cpu",
            params,
            args,
            task_type,
            n_classes,
        )
        return estimator, params
    params = crossgb_params_from_backbone(backbone, variant)
    assert_shared_backbone(backbone, params)
    estimator = build_estimator(
        xgb,
        "crossgb",
        params,
        args,
        task_type,
        n_classes,
    )
    estimator.set_params(
        crossgb_leaf_only_use_bias=bool(variant.leaf_only_use_bias)
    )
    return estimator, params


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("inf")


def _finite_std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0
    return float(np.std(finite, ddof=1))


def select_xgb_backbone(
    *,
    xgb: Any,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    task_type: str,
    n_classes: int,
    X_outer_train: Optional[np.ndarray] = None,
    y_outer_train: Optional[np.ndarray] = None,
    inner_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    prepared_inner_folds: Optional[
        Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = None,
) -> BackboneSelection:
    if not candidates:
        raise ValueError("At least one XGB backbone candidate is required")
    if prepared_inner_folds is None:
        if X_outer_train is None or y_outer_train is None or inner_splits is None:
            raise ValueError(
                "Provide either prepared_inner_folds or outer-train arrays and splits"
            )
        evaluation_folds = [
            (
                X_outer_train[inner_train],
                y_outer_train[inner_train],
                X_outer_train[inner_validation],
                y_outer_train[inner_validation],
            )
            for inner_train, inner_validation in inner_splits
        ]
    else:
        evaluation_folds = list(prepared_inner_folds)
    if not evaluation_folds:
        raise ValueError("At least one inner evaluation fold is required")
    best: BackboneSelection | None = None
    for candidate_index, candidate in enumerate(candidates, start=1):
        values: List[float] = []
        fit_total = 0.0
        predict_total = 0.0
        for X_inner_train, y_inner_train, X_inner_validation, y_inner_validation in (
            evaluation_folds
        ):
            estimator = build_estimator(
                xgb,
                "xgb_cpu",
                candidate,
                args,
                task_type,
                n_classes,
            )
            pred, proba, fit_sec, predict_sec = fit_predict_timed(
                estimator,
                X_inner_train,
                y_inner_train,
                X_inner_validation,
                task_type,
                n_classes,
            )
            metrics = evaluate_metrics(
                task_type,
                y_inner_validation,
                pred,
                proba,
                n_classes,
            )
            values.append(primary_value(task_type, metrics))
            fit_total += fit_sec
            predict_total += predict_sec
        selection = BackboneSelection(
            candidate_index=candidate_index,
            candidate_id=str(candidate.get("candidate_id", candidate_index)),
            validation_primary_metric=_finite_mean(values),
            validation_primary_metric_std=_finite_std(values),
            validation_fit_sec=fit_total,
            validation_predict_sec=predict_total,
            params=dict(candidate),
        )
        if best is None or selection.validation_primary_metric < best.validation_primary_metric:
            best = selection
    assert best is not None
    return best


def evaluate_variant(
    *,
    xgb: Any,
    variant: VariantSpec,
    backbone: Dict[str, Any],
    args: argparse.Namespace,
    task_type: str,
    n_classes: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    estimator, params = build_variant_estimator(
        xgb,
        variant,
        backbone,
        args,
        task_type,
        n_classes,
    )
    pred, proba, fit_sec, predict_sec = fit_predict_timed(
        estimator,
        X_train,
        y_train,
        X_test,
        task_type,
        n_classes,
    )
    metrics = evaluate_metrics(task_type, y_test, pred, proba, n_classes)
    return {
        "pred": np.asarray(pred),
        "proba": None if proba is None else np.asarray(proba),
        "metrics": metrics,
        "test_primary_metric": primary_value(task_type, metrics),
        "fit_sec": float(fit_sec),
        "predict_sec": float(predict_sec),
        "params": params,
        "params_hash": canonical_hash(params),
    }


def load_xgboost(python_package: Path) -> Any:
    return import_local_xgboost(python_package)


def xgboost_runtime_identity(python_package: Path) -> Dict[str, Any]:
    xgb = load_xgboost(python_package)
    package_file = Path(xgb.__file__).resolve()
    package_root = (python_package / "xgboost").resolve()
    if not package_file.is_relative_to(package_root):
        raise ValueError(
            "Loaded XGBoost package is outside the requested local package: "
            f"{package_file} not under {package_root}"
        )
    python_files = {
        str(path.relative_to(package_root)): file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    if not python_files:
        raise ValueError(f"No XGBoost Python sources found under {package_root}")
    library_name = getattr(getattr(xgb, "core", None), "_LIB", None)
    library_path_value = getattr(library_name, "_name", "")
    library_path = (
        Path(str(library_path_value)).resolve()
        if library_path_value
        else None
    )
    return {
        "version": str(getattr(xgb, "__version__", "unknown")),
        "python_package_root": str(package_root),
        "python_package_file": str(package_file),
        "python_package_file_sha256": file_sha256(package_file),
        "python_source_files": python_files,
        "python_source_combined_sha256": canonical_hash(python_files),
        "library_path": str(library_path) if library_path else "",
        "library_sha256": (
            file_sha256(library_path)
            if library_path is not None and library_path.exists()
            else ""
        ),
    }
