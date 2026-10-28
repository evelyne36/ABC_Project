import numpy as np
import pytest

from baselines import make_regressor


PARAMS = {
    "linxgboost": {"n_estimators": 5, "max_depth": 2, "lbda": 0.1, "min_samples_leaf": 8},
    "pilot": {"max_depth": 3, "max_model_depth": 8, "min_sample_leaf": 8},
    "tao": {"depth": 4, "max_iter": 3, "C_dnode": 10},
    "gbdt_pl": {"num_trees": 12, "num_leaves": 4, "max_var": 3,
                "learning_rate": 0.2, "min_sum_hessians": 4, "num_bins": 31},
}


@pytest.fixture
def data():
    rng = np.random.default_rng(20261025)
    X = rng.uniform(-1, 1, size=(144, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + 4 * (X[:, 0] > 0.15)
    return X[:112], y[:112], X[112:], y[112:]


@pytest.mark.parametrize("method", list(PARAMS))
def test_adapter_trains_real_method_without_mutating_inputs_or_rng(method, data):
    X, y, Z, target = data
    X0, y0 = X.copy(), y.copy()
    model = make_regressor(method, PARAMS[method], seed=13, n_jobs=1)
    state = np.random.get_state()
    assert model.fit(X, y) is model
    actual = model.predict(Z)
    assert actual.shape == (len(Z),)
    assert np.isfinite(actual).all()
    assert np.sqrt(np.mean((actual - target) ** 2)) < 0.8 * np.sqrt(np.mean((y.mean() - target) ** 2))
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    restored = np.random.get_state()
    assert state[0] == restored[0] and state[2:] == restored[2:]
    np.testing.assert_array_equal(state[1], restored[1])
    np.testing.assert_allclose(model.predict(Z), actual, atol=1e-12)
    np.testing.assert_allclose(model.predict(Z[::-1]), actual[::-1], atol=1e-10)


def test_linxgboost_preserves_upstream_last_tree_semantics(data):
    X, y, Z, _ = data
    model = make_regressor("linxgboost", PARAMS["linxgboost"], seed=13).fit(X, y)
    trees = model.model_.trees
    eta = model.model_.learning_rate
    want = sum(eta * t.predict(Z) for t in trees[:-1]) + trees[-1].predict(Z)
    np.testing.assert_allclose(model.predict(Z), want)
    all_shrunk = sum(eta * t.predict(Z) for t in trees)
    assert not np.allclose(want, all_shrunk)


def test_tao_uses_official_constant_regression_leaf(data):
    X, y, Z, _ = data
    model = make_regressor("tao", {"depth": 1, "max_iter": 2}, seed=13).fit(X, y)
    np.testing.assert_allclose(model.predict(Z), np.repeat(y.mean(), len(Z)))
    assert model.metadata_["leaf_model"] == "constant"


@pytest.mark.parametrize("method", list(PARAMS))
def test_adapter_rejects_nonfinite_inputs_and_wrong_feature_count(method, data):
    X, y, Z, _ = data
    model = make_regressor(method, PARAMS[method])
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad, y)
    with pytest.raises(ValueError, match="one-dimensional"):
        model.fit(X, y.reshape(-1, 1))


def test_unknown_parameter_is_not_silently_ignored(data):
    X, y, _, _ = data
    with pytest.raises((TypeError, ValueError), match="parameter"):
        make_regressor("gbdt_pl", {"max_deth": 2}).fit(X, y)


def test_gbdt_pl_refuses_an_unidentified_native_library(tmp_path,monkeypatch,data):
    library=tmp_path/"unexpected.dylib"
    library.write_bytes(b"not the pinned native build")
    monkeypatch.setenv("GBDTPL_LIBRARY",str(library))
    X,y,_,_=data
    with pytest.raises(RuntimeError,match="provenance"):
        make_regressor("gbdt_pl",PARAMS["gbdt_pl"]).fit(X,y)
