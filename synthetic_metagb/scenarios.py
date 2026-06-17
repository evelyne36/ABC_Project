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
    family: str = "legacy"


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


def xor_3way(rng: np.random.Generator, n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x1 = rng.uniform(-1.0, 1.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    x3 = rng.uniform(-1.0, 1.0, size=n)
    nuisance = rng.normal(size=(n, 3))
    margin = np.sign(x1 * x2) * x3 + _noise(rng, n, noise)
    X = np.column_stack([x1, x2, x3, nuisance])
    y = (margin > 0.0).astype(np.int64)
    return X.astype(np.float32), y


def conditional_multiplicative(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    gate = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 3))
    coefficient = np.where(gate < 0.0, -2.0, 4.0)
    X = np.column_stack([gate, x1, x2, nuisance])
    y = coefficient * x1 * x2 + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def regime_slope_multi(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    gate = rng.uniform(-1.0, 1.0, size=n)
    predictive = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 4))
    region = np.digitize(gate, [-0.5, 0.0, 0.5])
    slopes = np.asarray([-4.0, -1.5, 1.5, 4.0])
    X = np.column_stack([gate, predictive, nuisance])
    y = slopes[region] * predictive + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def categorical_regime_slope(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    category = rng.integers(0, 4, size=n)
    one_hot = np.eye(4, dtype=np.float32)[category]
    predictive = rng.uniform(-2.0, 2.0, size=n)
    slopes = np.asarray([-4.0, -1.5, 1.5, 4.0])
    nuisance = rng.normal(size=(n, 1))
    X = np.column_stack([one_hot, predictive, nuisance])
    y = slopes[category] * predictive + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def categorical_feature_switch(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    category = rng.integers(0, 3, size=n)
    one_hot = np.eye(3, dtype=np.float32)[category]
    predictive = rng.uniform(-2.0, 2.0, size=(n, 3))
    coefficients = np.asarray([3.0, -2.5, 1.5])
    selected = predictive[np.arange(n), category]
    X = np.column_stack([one_hot, predictive])
    y = coefficients[category] * selected + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def mixed_gate_regime(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    category = rng.integers(0, 2, size=n)
    one_hot = np.eye(2, dtype=np.float32)[category]
    gate = rng.uniform(-1.0, 1.0, size=n)
    predictive = rng.uniform(-2.0, 2.0, size=n)
    nuisance = rng.normal(size=(n, 2))
    region = category * 2 + (gate >= 0.0).astype(np.int64)
    slopes = np.asarray([-4.0, -1.5, 1.5, 4.0])
    X = np.column_stack([one_hot, gate, predictive, nuisance])
    y = slopes[region] * predictive + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def sparse_multiplicative(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    relevant = rng.uniform(-2.0, 2.0, size=(n, 5))
    nuisance = rng.normal(size=(n, 1))
    X = np.column_stack([relevant, nuisance])
    y = 3.0 * relevant[:, 0] * relevant[:, 1] + 0.5 * relevant[:, 2] + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def sparse_regime_slope(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    relevant = rng.uniform(-2.0, 2.0, size=(n, 5))
    nuisance = rng.normal(size=(n, 1))
    slope = np.where(relevant[:, 0] < 0.0, -3.0, 3.0)
    X = np.column_stack([relevant, nuisance])
    y = slope * relevant[:, 1] + 0.25 * relevant[:, 2] + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def sparse_feature_switch(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    relevant = rng.uniform(-2.0, 2.0, size=(n, 5))
    nuisance = rng.normal(size=(n, 1))
    y = np.where(relevant[:, 0] < 0.0, 3.0 * relevant[:, 1], -2.5 * relevant[:, 2])
    X = np.column_stack([relevant, nuisance])
    y = y + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


def additive_nonlinear_control(
    rng: np.random.Generator, n: int, noise: float
) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-2.0, 2.0, size=(n, 3))
    nuisance = rng.normal(size=(n, 3))
    X = np.column_stack([x, nuisance])
    y = np.sin(2.0 * x[:, 0]) + 0.75 * x[:, 1] ** 2 + np.cos(x[:, 2])
    y = y + _noise(rng, n, noise)
    return X.astype(np.float32), y.astype(np.float32)


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
            "B_linear_in_leaves",
        ),
        Scenario(
            "regime_slope",
            "regression",
            "One conditional split exposes two opposite linear leaf responses.",
            "current_metagb",
            regime_slope,
            "B_linear_in_leaves",
        ),
        Scenario(
            "feature_switch",
            "regression",
            "A gate selects which feature should make the local prediction.",
            "current_metagb",
            feature_switch,
            "B_linear_in_leaves",
        ),
        Scenario(
            "multiplicative",
            "regression",
            "Tree partitions approximate one factor while leaf slopes model the other factor.",
            "current_metagb",
            multiplicative,
            "A_interaction",
        ),
        Scenario(
            "pure_step_control",
            "regression",
            "The target is already piecewise constant, matching ordinary XGBoost leaves.",
            "xgb_cpu_or_tie",
            pure_step_control,
            "E_negative_control",
        ),
        Scenario(
            "curved_control",
            "regression",
            "A smooth univariate curve is not directly representable by a single feature-scaled leaf.",
            "xgb_cpu_or_tie",
            curved_control,
            "E_negative_control",
        ),
        Scenario(
            "xor_classification",
            "binary",
            "A conditional split turns the other feature into a signed leaf-level classifier.",
            "current_metagb",
            xor_classification,
            "A_interaction",
        ),
        Scenario(
            "regime_logit",
            "binary",
            "The sign of a predictive feature's logit effect changes across a gate.",
            "current_metagb",
            regime_logit,
            "B_linear_in_leaves",
        ),
        Scenario(
            "xor_3way",
            "binary",
            "A higher-order conditional interaction exposes x3 as a signed leaf classifier.",
            "current_metagb",
            xor_3way,
            "A_interaction",
        ),
        Scenario(
            "conditional_multiplicative",
            "regression",
            "A gate changes the coefficient and sign of a multiplicative response.",
            "current_metagb",
            conditional_multiplicative,
            "A_interaction",
        ),
        Scenario(
            "regime_slope_multi",
            "regression",
            "Four regions expose four different linear leaf responses.",
            "current_metagb",
            regime_slope_multi,
            "B_linear_in_leaves",
        ),
        Scenario(
            "categorical_regime_slope",
            "regression",
            "A one-hot category defines the slope of a continuous predictive feature.",
            "current_metagb",
            categorical_regime_slope,
            "C_conditional_predictive",
        ),
        Scenario(
            "categorical_feature_switch",
            "regression",
            "A one-hot category selects which continuous feature predicts the target.",
            "current_metagb",
            categorical_feature_switch,
            "C_conditional_predictive",
        ),
        Scenario(
            "mixed_gate_regime",
            "regression",
            "Categorical and continuous gates jointly define a local linear response.",
            "current_metagb",
            mixed_gate_regime,
            "C_conditional_predictive",
        ),
        Scenario(
            "sparse_multiplicative",
            "regression",
            "Five relevant features contain a sparse multiplicative signal.",
            "current_metagb",
            sparse_multiplicative,
            "D_sparse_interaction",
        ),
        Scenario(
            "sparse_regime_slope",
            "regression",
            "Five relevant features contain a sparse gate and predictive feature.",
            "current_metagb",
            sparse_regime_slope,
            "D_sparse_interaction",
        ),
        Scenario(
            "sparse_feature_switch",
            "regression",
            "Five relevant features contain a sparse feature-switching response.",
            "current_metagb",
            sparse_feature_switch,
            "D_sparse_interaction",
        ),
        Scenario(
            "additive_nonlinear_control",
            "regression",
            "An additive nonlinear target is not specifically aligned with linear leaves.",
            "xgb_cpu_or_tie",
            additive_nonlinear_control,
            "E_negative_control",
        ),
    ]
}


def scenario_names() -> List[str]:
    return list(SCENARIOS)


def generate_instance(
    base_scenario: str,
    data_seed: int,
    n_samples: int,
    noise: float,
    extra_nuisance_features: int = 0,
    target_features: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(data_seed)
    X, y = SCENARIOS[base_scenario].generator(rng, n_samples, noise)
    nuisance_count = extra_nuisance_features
    if target_features is not None:
        if target_features < X.shape[1]:
            raise ValueError(
                f"target_features={target_features} is below base feature count {X.shape[1]}"
            )
        nuisance_count += target_features - X.shape[1]
    if nuisance_count > 0:
        nuisance = rng.normal(size=(n_samples, nuisance_count)).astype(np.float32)
        X = np.column_stack([X, nuisance]).astype(np.float32)
    return X, y
