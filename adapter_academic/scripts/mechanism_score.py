from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class ScoreConfig:
    lambda_l2: float = 1.0
    max_depth: int = 3
    min_samples_leaf_fraction: float = 0.03
    min_samples_leaf_floor: int = 20
    sample_cap: int = 5000
    seed: int = 42
    weight_linear: float = 0.35
    weight_decoupling: float = 0.25
    weight_heterogeneity: float = 0.20
    weight_concentration: float = 0.10
    weight_support: float = 0.10
    strong_threshold: float = 0.65
    moderate_threshold: float = 0.40


def safe_divide(num: float, den: float, default: float = math.nan) -> float:
    if den == 0 or not math.isfinite(den):
        return default
    return num / den


def clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def bounded_fraction(value: float, low: float, high: float) -> float:
    if high <= low or not math.isfinite(value):
        return 0.0
    return clip01((value - low) / (high - low))


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    valid = std > 1e-12
    Z = np.zeros_like(X, dtype=np.float64)
    if np.any(valid):
        Z[:, valid] = (X[:, valid] - mean[valid]) / std[valid]
    return Z, valid


def projected_scores(X: np.ndarray, residual: np.ndarray, config: ScoreConfig) -> np.ndarray:
    if X.shape[0] < 3 or X.shape[1] == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    Z, valid = standardize(X)
    numerators = np.sum(Z * residual[:, None], axis=0) ** 2
    denominators = np.sum(Z * Z, axis=0) + config.lambda_l2
    scores = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators > 0)
    scores[~valid] = 0.0
    return scores


def alignment_ratio(scores: np.ndarray, residual: np.ndarray, config: ScoreConfig) -> float:
    residual_energy = float(np.sum(residual * residual) + config.lambda_l2)
    return clip01(float(np.max(scores)) / residual_energy if scores.size else 0.0)


def feature_hit(best_feature: int, true_predictive: set[int]) -> float:
    if not true_predictive:
        return math.nan
    return 1.0 if best_feature in true_predictive else 0.0


def score_arrays(
    X: np.ndarray,
    y: np.ndarray,
    config: ScoreConfig,
    true_predictive_features: set[int] | None = None,
) -> dict[str, Any]:
    true_predictive_features = true_predictive_features or set()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible row counts")

    rng = np.random.default_rng(config.seed)
    if X.shape[0] > config.sample_cap:
        idx = rng.choice(X.shape[0], size=config.sample_cap, replace=False)
        X_fit = X[idx]
        y_fit = y[idx]
    else:
        X_fit = X
        y_fit = y

    global_residual = y_fit - float(np.mean(y_fit))
    global_scores = projected_scores(X_fit, global_residual, config)
    global_alignment = alignment_ratio(global_scores, global_residual, config)

    min_leaf = max(config.min_samples_leaf_floor, int(config.min_samples_leaf_fraction * X_fit.shape[0]))
    tree = DecisionTreeRegressor(max_depth=config.max_depth, min_samples_leaf=min_leaf, random_state=config.seed)
    tree.fit(X_fit, y_fit)
    leaf_ids = tree.apply(X_fit)
    split_features = {int(feature) for feature in tree.tree_.feature if int(feature) >= 0}

    local_alignments: list[float] = []
    concentrations: list[float] = []
    decoupled_flags: list[float] = []
    hit_flags: list[float] = []
    best_features: list[int] = []

    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        if int(mask.sum()) < config.min_samples_leaf_floor:
            continue
        y_leaf = y_fit[mask]
        residual = y_leaf - float(np.mean(y_leaf))
        scores = projected_scores(X_fit[mask], residual, config)
        if scores.size == 0:
            continue
        best = int(np.argmax(scores))
        best_features.append(best)
        local_alignments.append(alignment_ratio(scores, residual, config))
        positive_sum = float(np.sum(scores[scores > 0]))
        concentrations.append(safe_divide(float(np.max(scores)), positive_sum, default=0.0))
        decoupled_flags.append(1.0 if best not in split_features else 0.0)
        hit = feature_hit(best, true_predictive_features)
        if math.isfinite(hit):
            hit_flags.append(hit)

    local_alignment = float(np.mean(local_alignments)) if local_alignments else 0.0
    heterogeneity = clip01(local_alignment - global_alignment)
    decoupling = float(np.mean(decoupled_flags)) if decoupled_flags else 0.0
    concentration = float(np.mean(concentrations)) if concentrations else 0.0
    signal_support = bounded_fraction(math.sqrt(max(local_alignment, 0.0)), low=0.05, high=0.55)
    feature_hit_rate = float(np.mean(hit_flags)) if hit_flags else math.nan

    components = {
        "component_linear_leaf_advantage": bounded_fraction(local_alignment, low=0.01, high=0.20),
        "component_cond_pred_decoupling": bounded_fraction(decoupling, low=0.20, high=0.80),
        "component_local_heterogeneity": bounded_fraction(heterogeneity, low=0.01, high=0.15),
        "component_predictive_concentration": bounded_fraction(concentration, low=0.20, high=0.70),
        "component_signal_support": signal_support,
    }
    weight_sum = (
        config.weight_linear
        + config.weight_decoupling
        + config.weight_heterogeneity
        + config.weight_concentration
        + config.weight_support
    )
    score = (
        config.weight_linear * components["component_linear_leaf_advantage"]
        + config.weight_decoupling * components["component_cond_pred_decoupling"]
        + config.weight_heterogeneity * components["component_local_heterogeneity"]
        + config.weight_concentration * components["component_predictive_concentration"]
        + config.weight_support * components["component_signal_support"]
    )
    score = clip01(score / weight_sum)
    return {
        "mechanism_alignment_score": score,
        "mechanism_band": score_band(score, config),
        "global_projected_alignment": global_alignment,
        "local_projected_alignment": local_alignment,
        "local_global_alignment_lift": safe_divide(local_alignment, global_alignment + 1e-12, default=0.0),
        "decoupled_leaf_fraction": decoupling,
        "predictive_score_concentration": concentration,
        "feature_hit_rate": feature_hit_rate,
        "tree_leaf_count": float(len(local_alignments)),
        "split_feature_count": float(len(split_features)),
        "best_predictive_features": ";".join(str(item) for item in sorted(set(best_features))),
        **components,
    }


def score_band(score: float, config: ScoreConfig) -> str:
    if score >= config.strong_threshold:
        return "STRONG_MECHANISM_MATCH"
    if score >= config.moderate_threshold:
        return "MODERATE_MECHANISM_MATCH"
    return "WEAK_MECHANISM_MATCH"


def score_proxy_row(row: pd.Series, config: ScoreConfig) -> dict[str, Any]:
    def f(name: str, default: float = math.nan) -> float:
        value = row.get(name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    projected_lift = f("projected_gradient_lift_mean")
    local_lift = f("local_linearity_lift")
    decoupled = f("decoupled_leaf_fraction")
    concentration = f("predictive_score_concentration_mean")
    local_corr = f("local_abs_corr_max_mean")

    components = {
        "component_linear_leaf_advantage": bounded_fraction(math.log1p(max(projected_lift - 1.0, 0.0)), low=0.0, high=math.log1p(7.0)),
        "component_cond_pred_decoupling": bounded_fraction(decoupled, low=0.20, high=0.80),
        "component_local_heterogeneity": bounded_fraction(math.log1p(max(local_lift - 1.0, 0.0)), low=0.0, high=math.log1p(0.8)),
        "component_predictive_concentration": bounded_fraction(concentration, low=0.25, high=0.65),
        "component_signal_support": bounded_fraction(local_corr, low=0.10, high=0.60),
    }
    weight_sum = (
        config.weight_linear
        + config.weight_decoupling
        + config.weight_heterogeneity
        + config.weight_concentration
        + config.weight_support
    )
    score = (
        config.weight_linear * components["component_linear_leaf_advantage"]
        + config.weight_decoupling * components["component_cond_pred_decoupling"]
        + config.weight_heterogeneity * components["component_local_heterogeneity"]
        + config.weight_concentration * components["component_predictive_concentration"]
        + config.weight_support * components["component_signal_support"]
    )
    score = clip01(score / weight_sum)
    return {
        "mechanism_alignment_score": score,
        "mechanism_band": score_band(score, config),
        **components,
    }
