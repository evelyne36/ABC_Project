from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Optional, Type

import numpy as np


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Optional dependency '{module_name}' is required for this model") from exc


def _common_seed(args: Any) -> int:
    return int(getattr(args, "seed", 42))


def _common_jobs(args: Any) -> int:
    return int(getattr(args, "n_jobs", 1))


def build_estimator(model_name: str, params: Dict[str, Any], args: Any, task_type: str, n_classes: int) -> Any:
    if model_name == "linear":
        return _build_linear(params, args, task_type)
    if model_name == "extra_trees":
        return _build_forest("extra_trees", params, args, task_type)
    if model_name == "random_forest":
        return _build_forest("random_forest", params, args, task_type)
    if model_name == "lightgbm":
        return _build_lightgbm(params, args, task_type, n_classes)
    if model_name == "catboost":
        return _build_catboost(params, args, task_type, n_classes)
    if model_name == "torchmlp":
        return TorchMLPEstimator(task_type=task_type, n_classes=n_classes, random_state=_common_seed(args), **params)
    if model_name in {"realmlp", "tabm"}:
        return _build_pytabkit(model_name, params, args, task_type)
    raise ValueError(f"Unknown baseline model: {model_name}")


def _build_linear(params: Dict[str, Any], args: Any, task_type: str) -> Any:
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if task_type == "regression":
        alpha = float(params.get("alpha", 1.0))
        estimator = LinearRegression() if alpha <= 0.0 else Ridge(alpha=alpha, random_state=_common_seed(args))
    else:
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            solver="lbfgs",
            n_jobs=_common_jobs(args),
            random_state=_common_seed(args),
        )
    return make_pipeline(StandardScaler(), estimator)


def _build_forest(kind: str, params: Dict[str, Any], args: Any, task_type: str) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor

    common = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_features": params.get("max_features", "sqrt"),
        "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
        "max_depth": params.get("max_depth", None),
        "random_state": _common_seed(args),
        "n_jobs": _common_jobs(args),
    }
    if kind == "extra_trees":
        return ExtraTreesClassifier(**common) if task_type != "regression" else ExtraTreesRegressor(**common)
    return RandomForestClassifier(**common) if task_type != "regression" else RandomForestRegressor(**common)


def _build_lightgbm(params: Dict[str, Any], args: Any, task_type: str, n_classes: int) -> Any:
    lgb = _optional_import("lightgbm")
    common = {
        "n_estimators": int(params.get("n_estimators", 400)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "max_depth": int(params.get("max_depth", -1)),
        "num_leaves": int(params.get("num_leaves", 31)),
        "subsample": float(params.get("subsample", 0.9)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.9)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "random_state": _common_seed(args),
        "n_jobs": _common_jobs(args),
        "verbosity": -1,
    }
    if task_type == "regression":
        common["objective"] = "regression"
        return lgb.LGBMRegressor(**common)
    common["objective"] = "binary" if task_type == "binary" else "multiclass"
    if task_type == "multiclass":
        common["num_class"] = int(n_classes)
    return lgb.LGBMClassifier(**common)


def _build_catboost(params: Dict[str, Any], args: Any, task_type: str, n_classes: int) -> Any:
    catboost = _optional_import("catboost")
    common = {
        "iterations": int(params.get("iterations", 400)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "depth": int(params.get("depth", 6)),
        "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
        "random_seed": _common_seed(args),
        "thread_count": _common_jobs(args),
        "allow_writing_files": False,
        "verbose": False,
    }
    if task_type == "regression":
        common["loss_function"] = "RMSE"
        return catboost.CatBoostRegressor(**common)
    common["loss_function"] = "Logloss" if task_type == "binary" else "MultiClass"
    if task_type == "multiclass":
        common["classes_count"] = int(n_classes)
    return catboost.CatBoostClassifier(**common)


def _find_class(module_names: Iterable[str], class_names: Iterable[str]) -> Type[Any]:
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
    raise ImportError(
        "Could not find a compatible pytabkit estimator class. "
        "Install/verify pytabkit and update random300_baselinemodel.utils.model_factory if its API changed."
    )


def _build_pytabkit(model_name: str, params: Dict[str, Any], args: Any, task_type: str) -> Any:
    module_names = [
        "pytabkit.models.sklearn.sklearn_interfaces",
        "pytabkit.models.sklearn.interfaces",
        "pytabkit",
    ]
    if model_name == "realmlp":
        class_names = (
            ["RealMLP_TD_Regressor", "RealMLPRegressor", "RealMLP_Regressor"]
            if task_type == "regression"
            else ["RealMLP_TD_Classifier", "RealMLPClassifier", "RealMLP_Classifier"]
        )
    else:
        class_names = (
            ["TabM_D_Regressor", "TabMRegressor", "TabM_TD_Regressor", "TabM_Regressor"]
            if task_type == "regression"
            else ["TabM_D_Classifier", "TabMClassifier", "TabM_TD_Classifier", "TabM_Classifier"]
        )
    cls = _find_class(module_names, class_names)
    kwargs = pytabkit_constructor_kwargs(model_name, params, args)
    try:
        return cls(**kwargs)
    except TypeError:
        minimal = {"random_state": _common_seed(args)}
        return cls(**minimal)


def pytabkit_constructor_kwargs(model_name: str, params: Dict[str, Any], args: Any) -> Dict[str, Any]:
    common = {
        "device": "cpu",
        "random_state": _common_seed(args),
        "n_threads": _common_jobs(args),
        "verbosity": 0,
        "n_epochs": int(params.get("max_epochs", 100)),
        "batch_size": int(params.get("batch_size", 256)),
        "lr": float(params.get("learning_rate", 0.001)),
    }
    if model_name == "realmlp":
        common["wd"] = float(params.get("weight_decay", 0.0001))
    elif model_name == "tabm":
        common["weight_decay"] = float(params.get("weight_decay", 0.0001))
        common["tabm_k"] = int(params.get("k", 32))
        common["compile_model"] = False
        common["allow_amp"] = False
    else:
        raise ValueError(f"Unsupported pytabkit model: {model_name}")
    return common


class TorchMLPEstimator:
    def __init__(
        self,
        task_type: str,
        n_classes: int,
        random_state: int,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 100,
        weight_decay: float = 0.0001,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        self.task_type = task_type
        self.n_classes = int(n_classes)
        self.random_state = int(random_state)
        self.hidden_layers = list(hidden_layers or [256, 128])
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.weight_decay = float(weight_decay)
        self.device_name = device

    def fit(self, X: Any, y: Any, **_: Any) -> "TorchMLPEstimator":
        torch = _optional_import("torch")
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr) if self.task_type != "regression" else None
        output_dim = 1 if self.task_type == "regression" else len(self.classes_)
        layers = []
        input_dim = X_arr.shape[1]
        for hidden_dim in self.hidden_layers:
            layers.append(torch.nn.Linear(input_dim, int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            input_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.model_ = torch.nn.Sequential(*layers)
        device = torch.device(self.device_name)
        self.model_.to(device)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32, device=device)
        if self.task_type == "regression":
            y_tensor = torch.tensor(y_arr.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)
            loss_fn = torch.nn.MSELoss()
        else:
            class_to_index = {value: index for index, value in enumerate(self.classes_)}
            indexed = np.asarray([class_to_index[value] for value in y_arr], dtype=np.int64)
            y_tensor = torch.tensor(indexed, dtype=torch.long, device=device)
            loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        row_count = X_tensor.shape[0]
        for _epoch in range(self.max_epochs):
            order = torch.randperm(row_count, generator=generator).to(device)
            for start in range(0, row_count, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = self.model_(X_tensor[batch_idx])
                loss = loss_fn(pred, y_tensor[batch_idx])
                loss.backward()
                optimizer.step()
        return self

    def _forward_numpy(self, X: Any) -> Any:
        torch = _optional_import("torch")
        X_arr = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            tensor = torch.tensor(X_arr, dtype=torch.float32, device=device)
            return self.model_(tensor).detach().cpu().numpy()

    def predict(self, X: Any) -> Any:
        output = self._forward_numpy(X)
        if self.task_type == "regression":
            return output.reshape(-1)
        return self.classes_[np.argmax(output, axis=1)]

    def predict_proba(self, X: Any) -> Any:
        if self.task_type == "regression":
            raise AttributeError("Regression TorchMLPEstimator does not expose predict_proba")
        output = self._forward_numpy(X)
        output = output - np.max(output, axis=1, keepdims=True)
        exp = np.exp(output)
        return exp / np.sum(exp, axis=1, keepdims=True)
