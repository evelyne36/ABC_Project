"""Behavior and serialization tests for the CPU CrossGB updater."""

import json

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


def make_crossgb_regressor(**kwargs: object) -> xgb.XGBRegressor:
    params = {
        "n_estimators": 3,
        "max_depth": 2,
        "min_child_weight": 1,
        "learning_rate": 0.3,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "enable_crossgb": True,
        "crossgb_lambda_l2": 0.0,
        "random_state": 1,
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def make_crossgb_classifier(**kwargs: object) -> xgb.XGBClassifier:
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "min_child_weight": 1,
        "learning_rate": 0.25,
        "tree_method": "hist",
        "enable_crossgb": True,
        "crossgb_lambda_l2": 0.0,
        "crossgb_score_agg": 0,
        "random_state": 1,
    }
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


@pytest.mark.parametrize("score_agg", [0, 1, 2, 3])
def test_crossgb_regressor_score_agg_smoke(score_agg: int) -> None:
    X, y = make_regression_data()

    model = make_crossgb_regressor(crossgb_score_agg=score_agg)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("score_agg", [0, 1, 2, 3])
def test_crossgb_score_agg_reselects_best_final_leaf_feature(
    score_agg: int, tmp_path
) -> None:
    predictive = np.linspace(-2.0, 2.0, 128, dtype=np.float32)
    X = np.column_stack([np.zeros_like(predictive), predictive])
    y = 3.0 * predictive
    model = make_crossgb_regressor(
        n_estimators=1,
        max_depth=2,
        learning_rate=1.0,
        max_bin=32,
        base_score=0.0,
        n_jobs=1,
        colsample_bytree=1.0,
        subsample=1.0,
        crossgb_pred_colsample_bytree=1.0,
        crossgb_score_agg=score_agg,
    )
    model.fit(X, y)

    path = tmp_path / f"score_agg_{score_agg}.json"
    model.save_model(path)
    tree = json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"][0]
    leaves = [i for i, left in enumerate(tree["left_children"]) if left == -1]
    leaf_pred_features = [tree["crossgb_leaf_features"][i] for i in leaves]

    # Feature 0 is the first sampled predictive feature but is constant. Every
    # final leaf must independently reselect feature 1, which explains y exactly.
    assert leaf_pred_features
    assert set(leaf_pred_features) == {1}


@pytest.mark.parametrize(
    "extra_params",
    [
        {"crossgb_leaf_only_use_bias": True},
        {"crossgb_cond_pred_share": True},
    ],
)
def test_crossgb_regressor_options_smoke(extra_params: dict[str, object]) -> None:
    X, y = make_regression_data()

    model = make_crossgb_regressor(crossgb_score_agg=0, **extra_params)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_crossgb_zero_pred_fraction_is_strictly_bias_only(tmp_path) -> None:
    X, y = make_regression_data()
    model = make_crossgb_regressor(
        n_estimators=2,
        crossgb_pred_colsample_bytree=0.0,
        crossgb_leaf_only_use_bias=False,
    )
    model.fit(X, y)

    path = tmp_path / "pred_fraction_zero.json"
    model.save_model(path)
    trees = json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"]
    bias_feature = X.shape[1]
    for tree in trees:
        leaves = [
            index
            for index, left_child in enumerate(tree["left_children"])
            if left_child == -1
        ]
        assert leaves
        assert {
            tree["crossgb_leaf_features"][index] for index in leaves
        } == {bias_feature}


def test_crossgb_min_data_in_leaf_smoke() -> None:
    X, y = make_regression_data()

    model = make_crossgb_regressor(crossgb_score_agg=0, crossgb_min_data_in_leaf=2)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_crossgb_model_round_trip(tmp_path) -> None:
    X, y = make_regression_data()
    model = make_crossgb_regressor(crossgb_score_agg=0)
    model.fit(X, y)
    before = model.predict(X)

    path = tmp_path / "crossgb.json"
    model.save_model(path)
    loaded = xgb.XGBRegressor()
    loaded.load_model(path)
    after = loaded.predict(X)

    np.testing.assert_allclose(before, after)


def test_crossgb_writes_canonical_fields(tmp_path) -> None:
    X, y = make_regression_data()
    model = make_crossgb_regressor(crossgb_score_agg=0)
    model.fit(X, y)
    expected = model.predict(X)

    canonical_path = tmp_path / "crossgb_canonical.json"
    model.save_model(canonical_path)
    config = json.loads(model.get_booster().save_config())
    booster_config = config["learner"]["gradient_booster"]
    tree_param = booster_config["tree_train_param"]
    assert tree_param["enable_crossgb"] == "1"
    assert booster_config["updater"][0]["name"] == "grow_crossgb_histmaker"

    payload = json.loads(canonical_path.read_text())
    trees = payload["learner"]["gradient_booster"]["model"]["trees"]
    assert trees
    for tree in trees:
        assert "crossgb_leaf_features" in tree
    loaded = xgb.XGBRegressor()
    loaded.load_model(canonical_path)
    np.testing.assert_allclose(expected, loaded.predict(X))


@pytest.mark.parametrize("objective", ["binary:logistic", "binary:logitraw"])
def test_crossgb_binary_classifier_smoke(objective: str) -> None:
    X, y = make_binary_data()

    model = make_crossgb_classifier(objective=objective)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_crossgb_binary_classifier_round_trip(tmp_path) -> None:
    X, y = make_binary_data()
    model = make_crossgb_classifier(objective="binary:logistic")
    model.fit(X, y)
    before = model.predict_proba(X)

    path = tmp_path / "crossgb_binary.json"
    model.save_model(path)
    loaded = xgb.XGBClassifier()
    loaded.load_model(path)
    after = loaded.predict_proba(X)

    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
def test_crossgb_multiclass_classifier_smoke(objective: str) -> None:
    X, y = make_multiclass_data()

    model = make_crossgb_classifier(objective=objective, num_class=3)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1, 2})

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_crossgb_multiclass_classifier_round_trip(tmp_path) -> None:
    X, y = make_multiclass_data()
    model = make_crossgb_classifier(objective="multi:softprob", num_class=3)
    model.fit(X, y)
    before = model.predict_proba(X)

    path = tmp_path / "crossgb_multiclass.json"
    model.save_model(path)
    loaded = xgb.XGBClassifier()
    loaded.load_model(path)
    after = loaded.predict_proba(X)

    np.testing.assert_allclose(before, after)
