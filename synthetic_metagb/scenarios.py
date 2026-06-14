from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    task_type: str
    hypothesis: str
    expected_winner: str
    generator: Callable[[np.random.Generator, int, float], Tuple[np.ndarray, np.ndarray]]


def _noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, size=n)


def global_linear(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    predictive = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 5))
    X = np.column_stack([predictive, nuisance])
    y = 3.0 * predictive + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def regime_slope(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    gate = rng.uniform(-1.0, 1.0, size=n)
    predictive = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 4))
    slope = np.where(gate < 0.0, -3.0, 3.0)
    X = np.column_stack([gate, predictive, nuisance])
    y = slope * predictive + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def feature_switch(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    gate = rng.uniform(-1.0, 1.0, size=n)
    pred_left = rng.uniform(-2.0, 2.0, size=n)
    pred_right = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 3))
    X = np.column_stack([gate, pred_left, pred_right, nuisance])
    y = np.where(gate < 0.0, 3.0 * pred_left, -2.5 * pred_right) + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def multiplicative(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    conditional = rng.uniform(-1.0, 1.0, size=n)
    predictive = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 4))
    X = np.column_stack([conditional, predictive, nuisance])
    y = 4.0 * conditional * predictive + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def pure_step_control(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 5))
    X = np.column_stack([x, nuisance])
    y = np.select([x < -0.7, x < 0.4, x < 1.2], [-3.0, 1.5, -1.0], default=3.0)
    y = y + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def curved_control(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 5))
    X = np.column_stack([x, nuisance])
    y = 2.0 * x * x + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def xor_classification(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    conditional = rng.uniform(-1.0, 1.0, size=n)
    predictive = rng.uniform(-1.0, 1.0, size=n)
    nuisance = rng.normal(size=(n, 4))
    margin = np.sign(conditional) * predictive
    margin = margin + _noise(rng, n, noise)
    X = np.column_stack([conditional, predictive, nuisance])
    y = (margin > 0.0).astype(np.int64)
    return X.astype(np.float32), y


def regime_logit(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    gate = rng.uniform(-1.0, 1.0, size=n)
    predictive = rng.normal(size=n)
    nuisance = rng.normal(size=(n, 4))
    margin = np.where(gate < 0.0, -3.0, 3.0) * predictive + _noise(rng, n, noise)
    probability = 1.0 / (1.0 + np.exp(-margin))
    X = np.column_stack([gate, predictive, nuisance])
    y = rng.binomial(1, probability).astype(np.int64)
    return X.astype(np.float32), y


SCENARIOS: Dict[str, Scenario] = {
    item.name: item
    for item in [
        Scenario(
            "global_linear",
            "regression",
            "A predictive feature is linear everywhere, so one MetaGB leaf can represent the signal.",
            "current_metagb",
            global_linear,
        ),
        Scenario(
            "regime_slope",
            "regression",
            "One conditional split exposes two opposite linear leaf responses.",
            "current_metagb",
            regime_slope,
        ),
        Scenario(
            "feature_switch",
            "regression",
            "A gate selects which feature should make the local prediction.",
            "current_metagb",
            feature_switch,
        ),
        Scenario(
            "multiplicative",
            "regression",
            "Tree partitions approximate one factor while leaf slopes model the other factor.",
            "current_metagb",
            multiplicative,
        ),
        Scenario(
            "pure_step_control",
            "regression",
            "The target is already piecewise constant, matching ordinary XGBoost leaves.",
            "xgb_cpu_or_tie",
            pure_step_control,
        ),
        Scenario(
            "curved_control",
            "regression",
            "A smooth univariate curve is not directly representable by a single feature-scaled leaf.",
            "xgb_cpu_or_tie",
            curved_control,
        ),
        Scenario(
            "xor_classification",
            "binary",
            "A conditional split turns the other feature into a signed leaf-level classifier.",
            "current_metagb",
            xor_classification,
        ),
        Scenario(
            "regime_logit",
            "binary",
            "The sign of a predictive feature's logit effect changes across a gate.",
            "current_metagb",
            regime_logit,
        ),
    ]
}


def scenario_names() -> List[str]:
    return list(SCENARIOS)

