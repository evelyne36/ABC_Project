"""LightGBM adapters: constant-leaf (`lgb_constant`) and `linear_tree` (`lgb_linear`).

Both variants are plain LightGBM fits from the official `lightgbm` wheel.  They
exist to answer the reviewer weakness that the rebuttal had no LightGBM
neighbour line, and specifically that `linear_tree=true` -- which the GBDT-PL
authors themselves describe as "an approximated version of GBDT-PL
implemented in LightGBM" -- had never been run here.

Scope statements that must travel with any number produced by this module:
  * `lgb_constant` is a *re-run* of an already archived method under the same
    nested protocol; it is the anchor for `lgb_linear`, not a new baseline.
  * `lgb_linear` is LightGBM's linear-tree approximation, NOT the authors'
    PL-Trees implementation (that is `gbdt_pl`).  It must never be described as
    "the GBDT-PL baseline".
  * The 300-configuration search space below is this rebuttal's own frozen
    space (structured like `configs/gbdt_pl_300.json`), not an author-paper
    exact search reproduction.

The class mimics the small surface `scripts/run_task.py` needs from
`baselines.estimators.AuthorRegressor`: `fit(X, y)`, `predict(X)`,
`metadata_`, `scaler_`, `model_`.
"""
from __future__ import annotations

import numpy as np

# method name -> linear_tree flag
METHODS = {"lgb_constant": False, "lgb_linear": True}

# Frozen default = the LightGBM starting configuration of this rebuttal's
# search space (kept in the same spirit as GBDT-PL's author start:
# 500 trees / 255 leaves / small L2).  Every key here is accepted by
# LGBMRegressor; unknown keys are rejected so a typo cannot silently no-op.
DEFAULTS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_child_samples": 100,
    "reg_lambda": 0.01,
    "reg_alpha": 0.01,
    "colsample_bytree": 1.0,
    "linear_lambda": 0.0,
}

FIXED = {
    "objective": "l2",
    "verbosity": -1,
    "num_threads": 1,
    "force_col_wise": True,
    "force_row_wise": False,
    "deterministic": True,
}

APPROXIMATION_NOTE = (
    "LightGBM linear_tree is documented by the GBDT-PL authors as an approximate "
    "implementation of GBDT-PL (upstream README); it is not the PL-Trees code and "
    "must not be reported as the GBDT-PL baseline."
)


def _as_matrix(X):
    array = np.asarray(X, dtype=np.float64, order="C")
    if array.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("X must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError("X must be finite")
    return array


def _as_target(y):
    array = np.asarray(y, dtype=np.float64, order="C")
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("y must be finite")
    return array


class LightGBMAdapter:
    def __init__(self, method, params=None, *, seed=42, standardize=False):
        import lightgbm

        if method not in METHODS:
            raise ValueError("unknown method: " + str(method))
        merged = dict(DEFAULTS)
        merged.update(params or {})
        unknown = sorted(set(merged) - set(DEFAULTS))
        if unknown:
            raise ValueError("unsupported LightGBM parameter(s): " + ", ".join(unknown))
        self._lgb = lightgbm
        self.method = method
        self.params = merged
        self.seed = int(seed)
        self.standardize = bool(standardize)
        self.linear_tree = METHODS[method]
        self.model_ = None
        self.scaler_ = None

    # -- sklearn-ish surface -------------------------------------------------
    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler

        matrix = _as_matrix(X)
        target = _as_target(y)
        if len(target) != len(matrix):
            raise ValueError("X and y disagree on the number of rows")
        if self.standardize:
            self.scaler_ = StandardScaler().fit(matrix)
            matrix = self.scaler_.transform(matrix)

        settings = dict(FIXED)
        settings.update(self.params)
        settings["random_state"] = self.seed
        settings["linear_tree"] = self.linear_tree
        if not self.linear_tree:
            # linear_lambda only exists for linear trees; dropping it keeps the
            # constant variant byte-identical to a plain LightGBM fit.
            settings.pop("linear_lambda", None)

        model = self._lgb.LGBMRegressor(**settings)
        model.fit(matrix, target)
        self.model_ = model
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("fit() before predict()")
        matrix = _as_matrix(X)
        if self.scaler_ is not None:
            matrix = self.scaler_.transform(matrix)
        return np.asarray(self.model_.predict(matrix), dtype=np.float64).ravel()

    # -- provenance ----------------------------------------------------------
    @property
    def metadata_(self):
        return {
            "implementation": "official_lightgbm_pip",
            "lightgbm_version": self._lgb.__version__,
            "leaf_model": "linear_tree_multivariate" if self.linear_tree else "constant",
            "linear_tree": self.linear_tree,
            "approximation_note": APPROXIMATION_NOTE if self.linear_tree else None,
            "prediction_semantics": "final_tree_shrunk",
            "thread_policy": "num_threads=1, deterministic=True, force_col_wise=True",
            "effective_params": {k: self.model_.get_params().get(k) for k in sorted(self.params)},
            "standardize_training_only": self.standardize,
        }


def make_regressor(method, params=None, *, seed=42, standardize=False):
    return LightGBMAdapter(method, params, seed=seed, standardize=standardize)
