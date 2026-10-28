"""Regression adapters for pinned author implementations; no replacement models."""
from contextlib import contextmanager, redirect_stdout
import ctypes as C
import importlib
import inspect
import hashlib
import json
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import types

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def vendor_root(method):
    roots = [p for p in (ROOT / "vendor" / method).iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one pinned {method} source directory; run scripts/fetch_sources.py")
    return roots[0]


@contextmanager
def author_imports(root, names):
    """Restore generic author module names after a single-process import/use."""
    saved = {k:v for k,v in sys.modules.items() if any(k == n or k.startswith(n+".") for n in names)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for k in list(sys.modules):
            if any(k == n or k.startswith(n+".") for n in names):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(root))


@contextmanager
def seeded(seed):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class AuthorRegressor:
    def __init__(self, method, params, seed, standardize):
        self.method, self.params, self.seed, self.standardize = method, dict(params), int(seed), bool(standardize)
        self.metadata_ = {"method": method, "task": "regression", "standardize_training_only": bool(standardize),
                          "threads_per_fit": 1, "test_labels_passed_to_fit": False}

    def _features(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional feature array")
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values")
        return np.array(X, dtype=np.float64, order="C", copy=True)

    def fit(self, X, y):
        X = self._features(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("y must have matching length and finite values")
        y = y.copy()
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        with seeded(self.seed):
            if self.method == "linxgboost":
                self._fit_linx(X, y)
            elif self.method == "pilot":
                self._fit_pilot(X, y)
            elif self.method == "tao":
                self._fit_tao(X, y)
            else:
                self.model_ = GBDTPLNative(self.params, self.seed).fit(X, y)
                self.metadata_.update(self.model_.metadata_)
        self.metadata_["effective_params"] = self.model_.params if self.method == "gbdt_pl" else self.effective_params_
        return self

    def _fit_linx(self, X, y):
        with author_imports(vendor_root("linxgboost"), ["linxgb", "node"]):
            upstream = importlib.import_module("linxgb")
            params = {"random_state": self.seed, "loss_function": "reg:squarederror", **self.params}
            if params["loss_function"] not in {"reg:squarederror", "reg:linear"}:
                raise ValueError("This adapter is regression-only; unsupported loss_function parameter")
            self.model_ = upstream.linxgb(**params).fit(X, y)
            self.effective_params_ = {key:getattr(self.model_,key) for key in inspect.signature(upstream.linxgb).parameters if hasattr(self.model_,key)}
        self.metadata_.update(implementation="author_python", leaf_model="multivariate_affine",
                              prediction_semantics="upstream_final_tree_unshrunk")

    def _fit_pilot(self, X, y):
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"build/matplotlib"))
        package_name = "_crossgb_author_pilot"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(vendor_root("pilot")/"pilot")]
            sys.modules[package_name] = package
        # Load the author's Python implementation directly, without executing
        # __init__.py's unconditional import of the separate C++ implementation.
        upstream = importlib.import_module(package_name+".Pilot")
        import numba
        previous_threads = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            self.model_ = upstream.PILOT(random_state=self.seed, **self.params)
            self.effective_params_ = {key:getattr(self.model_,key) for key in inspect.signature(upstream.PILOT).parameters if hasattr(self.model_,key)}
            self.model_.fit(X, y)
        finally:
            numba.set_num_threads(previous_threads)
        self.metadata_.update(implementation="author_python_numba", leaf_model="PILOT_node_model_selection",
                              import_compatibility="namespace_load_bypasses_optional_cpilot_import")

    def _fit_tao(self, X, y):
        params = {"depth": 4, "max_iter": 20, "C_dnode": 10.0, "alpha": 0.0,
                  "node_max_iter": 300, "tol": 1e-5, "strict_update": True, "postprocess": True,
                  "node_type": "linear", **self.params}
        allowed = {"depth", "max_iter", "C_dnode", "alpha", "node_max_iter", "tol", "strict_update", "postprocess", "node_type"}
        if set(params)-allowed:
            raise ValueError("Unsupported TAO parameter: "+str(sorted(set(params)-allowed)))
        if int(params["depth"]) < 1 or int(params["max_iter"]) < 1:
            raise ValueError("TAO depth and max_iter parameters must be positive")
        with author_imports(vendor_root("tao"), ["tao", "tao_tree", "utils", "models"]):
            tao = importlib.import_module("tao")
            self._tao_tree = importlib.import_module("tao_tree")
            utils = importlib.import_module("utils")
            _install_tao_node_shape_compat()
            self.model_ = self._tao_tree.get_complete_tree(X, y, depth=int(params["depth"]),
                init_type="median_split", node_type=params["node_type"], leaf_type="majority_regression")
            log = io.StringIO()
            with redirect_stdout(log):
                self.training_metrics_ = tao.tao_main(X, y, X, y, dtree=self.model_, loss=utils.mse_loss,
                    n_cpus=1, sequential=tuple(range(1, int(params["depth"])+1)),
                    max_iter=int(params["max_iter"]), tol=float(params["tol"]),
                    node_args={"C":float(params["C_dnode"]), "alpha":float(params["alpha"]),
                               "max_iter":int(params["node_max_iter"]), "scale":False},
                    leaf_args={}, postprocess=bool(params["postprocess"]), tree_logg_folder=None,
                    logg_progress=False, strict_update=bool(params["strict_update"]))
            self.training_log_ = log.getvalue()
        self.metadata_.update(implementation="official_2026_07_22_python", leaf_model="constant",
                              node_model=params["node_type"], paper_scope="TAO family; not NeurIPS2025 linear-leaf implementation")
        self.effective_params_ = {**params,"leaf_type":"majority_regression","init_type":"median_split"}

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before predict")
        X = self._features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training")
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        if self.method == "tao":
            result = self._tao_tree.predict(X, self.model_.get_root, target_dim=1)
        else:
            result = self.model_.predict(X)
        result = np.asarray(result, dtype=np.float64).reshape(-1)
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise RuntimeError("Upstream returned invalid predictions")
        return result


class GBDTPLNative:
    """Call the author's C API; one fit per worker process for bounded memory."""
    DEFAULTS = {"num_trees":100,"num_leaves":31,"min_sum_hessians":5.0,"l2_reg":0.1,"l1_reg":0.0,
                "objective":"l2","learning_rate":0.1,"eval_metric":"rmse","normalization":"no",
                "num_classes":1,"num_threads":1,"num_bins":63,"min_gain":0.0,"max_var":5,
                "grow_by":"leaf","leaf_type":"half_additive","max_depth":5,"verbose":1,
                "sparse_ratio":0.0,"boosting_type":"gbdt","goss_alpha":0.0,"goss_beta":0.0}

    def __init__(self, params, seed=42):
        self.seed = int(seed)
        if set(params)-set(self.DEFAULTS):
            raise ValueError("Unknown GBDT-PL parameter: "+str(sorted(set(params)-set(self.DEFAULTS))))
        self.params = {**self.DEFAULTS, **params}
        if self.params["objective"] != "l2" or self.params["num_classes"] != 1:
            raise ValueError("The current GBDT-PL adapter supports scalar regression only")
        if not 2 <= int(self.params["num_bins"]) <= 255:
            raise ValueError("num_bins parameter must lie in [2,255]")
        if int(self.params["num_threads"]) != 1:
            raise ValueError("Use one native thread per fit and parallel worker processes")
        if int(self.params["verbose"]) not in {1,2}:
            raise ValueError("verbose parameter must be 1 or 2: upstream verbose=0 dereferences an uninitialized metric")
        if self.params["boosting_type"] != "gbdt":
            raise ValueError("Only the verified gbdt boosting_type is enabled in this adapter")
        if self.params["grow_by"] not in {"leaf", "level"} or self.params["leaf_type"] not in {"linear", "half_additive", "constant"}:
            raise ValueError("Unsupported grow_by or leaf_type parameter")
        for key in ["num_trees", "num_leaves", "max_var", "max_depth"]:
            if int(self.params[key]) < 1:
                raise ValueError(key+" parameter must be positive")
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        self.library_path = Path(os.environ.get("GBDTPL_LIBRARY", str(ROOT/"build"/("liblineargbm"+suffix))))
        if not self.library_path.is_file():
            raise RuntimeError("GBDT-PL library missing; run scripts/build_gbdt_pl.py")
        build_file=ROOT/"reports/gbdt_pl_build.json"
        build=json.loads(build_file.read_text()) if build_file.is_file() else {}
        actual_hash=hashlib.sha256(self.library_path.read_bytes()).hexdigest()
        if actual_hash!=build.get("sha256"):
            raise RuntimeError("Native library provenance mismatch; rebuild with scripts/build_gbdt_pl.py")
        self.lib = C.CDLL(str(self.library_path))
        void = C.c_void_p
        self.lib.CreateLinearGBMBoosterConfig.argtypes = [C.POINTER(void)]
        self.lib.SetLinearGBMParams.argtypes = [void,C.c_char_p,C.c_char_p]
        self.lib.CreateLinearGBMDataMat.argtypes = [void,C.c_char_p,C.c_int,C.c_int,C.c_char_p,C.POINTER(void),void]
        self.lib.CreateLinearGBM.argtypes = [void,void,void,C.POINTER(void)]
        self.lib.Train.argtypes = [void]
        self.lib.LinearGBMPredict.argtypes = [void,void,C.POINTER(C.POINTER(C.c_double)),C.POINTER(C.c_int),C.c_int]
        self.metadata_ = {"implementation":"author_native_with_platform_compatibility", "leaf_model":self.params["leaf_type"],
                          "native_library":str(self.library_path), "prediction_iterations":"all_configured_trees",
                          "native_rng_seed":self.seed,
                          "native_library_sha256":actual_hash,
                          "native_compatibility":build.get("compatibility",[]),
                          "timing_scope":build.get("formal_timing_warning"),
                          "resource_policy":"one fit per worker process; upstream has no complete free-model API"}

    def _data(self, name, path, labels, reference=None):
        result = C.c_void_p()
        self.lib.CreateLinearGBMDataMat(self.config, name.encode(), 0 if labels else -1, -1,
            str(path).encode(), C.byref(result), reference or C.c_void_p())
        return result

    def fit(self, X, y):
        self.config = C.c_void_p()
        self.lib.CreateLinearGBMBoosterConfig(C.byref(self.config))
        for k,v in self.params.items():
            self.lib.SetLinearGBMParams(self.config, k.encode(), str(v).encode())
        with tempfile.TemporaryDirectory(prefix="gbdtpl_") as directory:
            path = Path(directory)/"train.csv"
            np.savetxt(path, np.column_stack([y,X]), delimiter=",", fmt="%.17g")
            self.train_data = self._data("train",path,True)
            self.eval_data = self._data("train_monitor",path,True,self.train_data)
            self.booster = C.c_void_p()
            self.lib.CreateLinearGBM(self.config,self.train_data,self.eval_data,C.byref(self.booster))
            # Upstream constructor calls srand(time(NULL)); reset immediately
            # before training. The fit CLI isolates this C RNG in its process.
            libc = C.CDLL(None)
            libc.srand.argtypes = [C.c_uint]
            libc.srand(C.c_uint(self.seed))
            self.lib.Train(self.booster)
        return self

    def predict(self, X):
        with tempfile.TemporaryDirectory(prefix="gbdtpl_predict_") as directory:
            path=Path(directory)/"features.csv"
            np.savetxt(path,X,delimiter=",",fmt="%.17g")
            data=self._data("predict",path,False,self.train_data)
            preds=C.POINTER(C.c_double)()
            count=C.c_int()
            self.lib.LinearGBMPredict(self.booster,data,C.byref(preds),C.byref(count),-1)
            if count.value != len(X):
                raise RuntimeError("Native GBDT-PL prediction length mismatch")
            return np.ctypeslib.as_array(preds,shape=(count.value,)).copy()


def make_regressor(method, params=None, *, seed=42, n_jobs=1, standardize=False):
    if method not in {"linxgboost", "pilot", "gbdt_pl", "tao"}:
        raise ValueError("Unknown baseline: "+str(method))
    if n_jobs != 1:
        raise ValueError("Use n_jobs=1 per fit; parallelize independent worker processes")
    return AuthorRegressor(method, params or {}, seed, standardize)

def _install_tao_node_shape_compat():
    """[crossgb-linux-compat] Shape-only normalization for TAO's strict-update rollback.

    TAO's exact-loss rollback restores the *model's own* previous parameters through
    models/decision_node.py NodeLinearModel.set_params, which asserts
    ``w.ndim == 2 and b.ndim == 1``. On some datasets the stored weight comes back 1-D
    (or the bias 2-D), so the rollback itself raises AssertionError and the whole refit
    process exits with status 1. We normalize the shapes of the *incoming* tuple and then
    delegate to the original method: values are untouched and predict() computes
    ``X @ w.T + b`` followed by ravel, which is identical for (D,) vs (1, D) weights and
    for (1, 1) vs (1,) biases. Author files stay byte-identical, so the pinned
    reports/sources.json fingerprint is unaffected; this wrapper lives in the adapter and
    is therefore covered by the per-run adapter_sha256.
    """
    try:
        node_module = importlib.import_module("models.decision_node")
    except Exception:
        return
    node_class = getattr(node_module, "NodeLinearModel", None)
    if node_class is None or getattr(node_class, "_crossgb_shape_compat", False):
        return
    original = node_class.set_params

    def set_params(self, param_tuple):
        weight, bias = param_tuple[0], param_tuple[1]
        if isinstance(weight, (int, float, np.floating)):
            weight = np.array([[float(weight)]])
        elif getattr(weight, "ndim", None) == 1:
            weight = weight.reshape(1, -1)
        if isinstance(bias, (int, float, np.floating)):
            bias = np.array([float(bias)])
        elif getattr(bias, "ndim", None) == 2:
            bias = bias.reshape(-1)
        return original(self, (weight, bias))

    node_class.set_params = set_params
    node_class._crossgb_shape_compat = True
