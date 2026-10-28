def make_regressor(method, params=None, *, seed=42, n_jobs=1, standardize=False):
    # Keep source-download/provenance tools usable before science dependencies
    # are installed; import the actual adapters only when constructing a model.
    from .estimators import make_regressor as implementation
    return implementation(method, params, seed=seed, n_jobs=n_jobs, standardize=standardize)

__all__ = ["make_regressor"]
