from __future__ import annotations

import numpy as np

from synthetic_metagb.run_benchmark import candidate_for_model
from synthetic_metagb.scenarios import SCENARIOS


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
    pred0 = candidate_for_model(base, "current_metagb_pred0")
    metagb = candidate_for_model(base, "current_metagb")
    assert pred0["pred_feature_fraction"] == 0.0
    assert metagb["pred_feature_fraction"] == 1.0
    assert base["pred_feature_fraction"] == 1.0
