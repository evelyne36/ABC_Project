from __future__ import annotations

import numpy as np

from synthetic_crossgb.run_benchmark import candidate_for_model
from synthetic_crossgb.protocols import main_experiment_instances
from synthetic_crossgb.scenarios import SCENARIOS, generate_instance


def test_all_scenarios_are_reproducible_and_finite() -> None:
    for scenario in SCENARIOS.values():
        first_X, first_y = scenario.generator(np.random.default_rng(123), 200, 0.1)
        second_X, second_y = scenario.generator(np.random.default_rng(123), 200, 0.1)
        assert first_X.shape[0] == 200
        assert first_y.shape == (200,)
        assert np.array_equal(first_X, second_X)
        assert np.array_equal(first_y, second_y)
        assert np.isfinite(first_X).all()
        assert np.isfinite(first_y).all()
        if scenario.task_type == "binary":
            assert set(np.unique(first_y)) == {0, 1}


def test_regime_slope_has_opposite_local_slopes() -> None:
    X, y = SCENARIOS["regime_slope"].generator(np.random.default_rng(9), 5000, 0.0)
    left_slope = np.polyfit(X[X[:, 0] < 0.0, 1], y[X[:, 0] < 0.0], deg=1)[0]
    right_slope = np.polyfit(X[X[:, 0] >= 0.0, 1], y[X[:, 0] >= 0.0], deg=1)[0]
    assert left_slope < -2.9
    assert right_slope > 2.9


def test_pred0_candidate_disables_predictive_features() -> None:
    base = {"pred_feature_fraction": 1.0, "max_depth": 2}
    pred0 = candidate_for_model(base, "crossgb_pred0")
    crossgb = candidate_for_model(base, "crossgb")
    assert pred0["pred_feature_fraction"] == 0.0
    assert crossgb["pred_feature_fraction"] == 1.0
    assert base["pred_feature_fraction"] == 1.0


def test_main_experiment_instances_have_exact_dimensions_and_reproduce() -> None:
    for instance in main_experiment_instances(n_samples=80):
        first_X, first_y = generate_instance(
            instance.base_scenario,
            42,
            instance.n_samples,
            instance.noise,
            target_features=instance.target_features,
        )
        second_X, second_y = generate_instance(
            instance.base_scenario,
            42,
            instance.n_samples,
            instance.noise,
            target_features=instance.target_features,
        )
        assert first_X.shape == (80, instance.dimension)
        assert np.array_equal(first_X, second_X)
        assert np.array_equal(first_y, second_y)


def test_noise_case_uses_stronger_gaussian_target_noise() -> None:
    clean_X, clean_y = generate_instance("regime_slope", 42, 5000, 0.0, target_features=10)
    default_X, default_y = generate_instance("regime_slope", 42, 5000, 0.15, target_features=10)
    noisy_X, noisy_y = generate_instance("regime_slope", 42, 5000, 0.75, target_features=10)
    assert np.array_equal(clean_X, default_X)
    assert np.array_equal(clean_X, noisy_X)
    assert np.std(noisy_y - clean_y) > 4.5 * np.std(default_y - clean_y)
