"""Smoke tests for the CPU MetaGB updater."""

import numpy as np
import pytest

import xgboost as xgb


def make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(1994)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    y = np.where(X[:, 0] <= 0, X[:, 1], -X[:, 1]).astype(np.float32)
    return X, y


def make_binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(2026)
    X = rng.normal(size=(160, 5)).astype(np.float32)
    margin = 1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3]
    y = (margin > np.median(margin)).astype(np.int32)
    return X, y


def make_multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(2027)
    X = rng.normal(size=(180, 6)).astype(np.float32)
    scores = np.column_stack(
        [
            X[:, 0] + 0.5 * X[:, 1],
            -X[:, 0] + X[:, 2],
            X[:, 3] - 0.25 * X[:, 4],
        ]
    )
    y = np.argmax(scores, axis=1).astype(np.int32)
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


def make_metagb_classifier(**kwargs: object) -> xgb.XGBClassifier:
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "min_child_weight": 1,
        "learning_rate": 0.25,
        "tree_method": "hist",
        "enable_metagb": True,
        "metagb_lambda_l2": 0.0,
        "metagb_score_agg": 0,
        "random_state": 1,
    }
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


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


def test_metagb_min_data_in_leaf_smoke() -> None:
    X, y = make_regression_data()

    model = make_metagb_regressor(metagb_score_agg=0, metagb_min_data_in_leaf=2)
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


@pytest.mark.parametrize("objective", ["binary:logistic", "binary:logitraw"])
def test_metagb_binary_classifier_smoke(objective: str) -> None:
    X, y = make_binary_data()

    model = make_metagb_classifier(objective=objective)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_metagb_binary_classifier_round_trip(tmp_path) -> None:
    X, y = make_binary_data()
    model = make_metagb_classifier(objective="binary:logistic")
    model.fit(X, y)
    before = model.predict_proba(X)

    path = tmp_path / "metagb_binary.json"
    model.save_model(path)
    loaded = xgb.XGBClassifier()
    loaded.load_model(path)
    after = loaded.predict_proba(X)

    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
def test_metagb_multiclass_classifier_smoke(objective: str) -> None:
    X, y = make_multiclass_data()

    model = make_metagb_classifier(objective=objective, num_class=3)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1, 2})

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_metagb_multiclass_classifier_round_trip(tmp_path) -> None:
    X, y = make_multiclass_data()
    model = make_metagb_classifier(objective="multi:softprob", num_class=3)
    model.fit(X, y)
    before = model.predict_proba(X)

    path = tmp_path / "metagb_multiclass.json"
    model.save_model(path)
    loaded = xgb.XGBClassifier()
    loaded.load_model(path)
    after = loaded.predict_proba(X)

    np.testing.assert_allclose(before, after)
