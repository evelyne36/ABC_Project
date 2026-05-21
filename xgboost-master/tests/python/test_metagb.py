"""Smoke tests for the CPU MetaGB updater."""

import numpy as np
import pytest

import xgboost as xgb


def make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(1994)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    y = np.where(X[:, 0] <= 0, X[:, 1], -X[:, 1]).astype(np.float32)
    return X, y


def make_metagb_regressor(**kwargs: object) -> xgb.XGBRegressor:
    params = {
        "n_estimators": 3,
        "max_depth": 2,
        "min_child_weight": 1,
        "learning_rate": 0.3,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "enable_metagb": True,
        "metagb_lambda_l2": 0.0,
        "random_state": 1,
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


@pytest.mark.parametrize("score_agg", [0, 1, 2, 3])
def test_metagb_regressor_score_agg_smoke(score_agg: int) -> None:
    X, y = make_regression_data()

    model = make_metagb_regressor(metagb_score_agg=score_agg)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


@pytest.mark.parametrize(
    "extra_params",
    [
        {"metagb_leaf_only_use_bias": True},
        {"metagb_cond_pred_share": True},
    ],
)
def test_metagb_regressor_options_smoke(extra_params: dict[str, object]) -> None:
    X, y = make_regression_data()

    model = make_metagb_regressor(metagb_score_agg=0, **extra_params)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_metagb_model_round_trip(tmp_path) -> None:
    X, y = make_regression_data()
    model = make_metagb_regressor(metagb_score_agg=0)
    model.fit(X, y)
    before = model.predict(X)

    path = tmp_path / "metagb.json"
    model.save_model(path)
    loaded = xgb.XGBRegressor()
    loaded.load_model(path)
    after = loaded.predict(X)

    np.testing.assert_allclose(before, after)
