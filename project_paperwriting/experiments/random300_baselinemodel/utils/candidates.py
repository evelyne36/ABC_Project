from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


BASELINE_MODEL_ORDER = [
    "linear",
    "extra_trees",
    "random_forest",
    "lightgbm",
    "catboost",
    "torchmlp",
]


DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "linear": {
        "default": {"alpha": 1.0, "C": 1.0, "max_iter": 1000},
        "space": {
            "alpha": {"type": "loguniform", "low": 0.0001, "high": 5.0},
            "C": {"type": "loguniform", "low": 0.2, "high": 10000.0},
            "max_iter": {"type": "choice", "values": [1000]},
        },
    },
    "extra_trees": {
        "default": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        "space": {
            "n_estimators": {"type": "choice", "values": [200, 400, 600, 800]},
            "max_features": {"type": "uniform", "low": 0.6, "high": 1.0},
            "min_samples_leaf": {"type": "logint", "low": 1, "high": 20},
            "max_depth": {"type": "choice", "values": [None, 4, 6, 8, 10]},
        },
    },
    "random_forest": {
        "default": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        "space": {
            "n_estimators": {"type": "choice", "values": [200, 400, 600, 800]},
            "max_features": {"type": "uniform", "low": 0.6, "high": 1.0},
            "min_samples_leaf": {"type": "logint", "low": 1, "high": 20},
            "max_depth": {"type": "choice", "values": [None, 4, 6, 8, 10]},
        },
    },
    "lightgbm": {
        "default": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
        "space": {
            "n_estimators": {"type": "choice", "values": [200, 400, 600, 800]},
            "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.1},
            "max_depth": {"type": "logint", "low": 4, "high": 10},
            "num_leaves": {"type": "choice", "values": [15, 31, 63, 127]},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_lambda": {"type": "loguniform", "low": 0.0001, "high": 5.0},
        },
    },
    "catboost": {
        "default": {"iterations": 400, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
        "space": {
            "iterations": {"type": "choice", "values": [200, 400, 600, 800]},
            "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.1},
            "depth": {"type": "logint", "low": 4, "high": 10},
            "l2_leaf_reg": {"type": "loguniform", "low": 0.0001, "high": 5.0},
        },
    },
    "torchmlp": {
        "default": {"hidden_layers": [256, 128], "learning_rate": 0.001, "batch_size": 256, "max_epochs": 100, "weight_decay": 0.0001},
        "space": {
            "hidden_layers": {"type": "choice", "values": [[128], [256, 128], [512, 256]]},
            "learning_rate": {"type": "loguniform", "low": 0.0001, "high": 0.003},
            "batch_size": {"type": "choice", "values": [128, 256, 512]},
            "max_epochs": {"type": "choice", "values": [80, 120, 200]},
            "weight_decay": {"type": "loguniform", "low": 0.0001, "high": 0.01},
        },
    },
}


def load_search_space(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_SEARCH_SPACE)
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_one(rng: random.Random, spec: Mapping[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "choice":
        return deepcopy(rng.choice(list(spec["values"])))
    if kind == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if kind == "loguniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if low <= 0.0 or high <= 0.0:
            raise ValueError(f"loguniform bounds must be positive: {spec}")
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if kind == "logint":
        low = int(spec["low"])
        high = int(spec["high"])
        if low <= 0 or high <= 0:
            raise ValueError(f"logint bounds must be positive: {spec}")
        sampled = math.exp(rng.uniform(math.log(low), math.log(high)))
        return int(min(high, max(low, round(sampled))))
    if kind == "int":
        return int(rng.randint(int(spec["low"]), int(spec["high"])))
    raise ValueError(f"Unsupported search-space type: {kind}")


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in candidate.items() if key not in {"candidate_id", "source"}}
    return json.dumps(payload, sort_keys=True)


def generate_model_candidates(model: str, max_candidates: int, seed: int, search_space: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if model not in search_space:
        raise ValueError(f"Unknown baseline model in search space: {model}")

    model_space = search_space[model]
    candidates: List[Dict[str, Any]] = []
    default_params = deepcopy(model_space.get("default", {}))
    default_params["candidate_id"] = f"{model}_default"
    default_params["source"] = "default"
    candidates.append(default_params)

    rng = random.Random(seed + 1009 * (BASELINE_MODEL_ORDER.index(model) + 1 if model in BASELINE_MODEL_ORDER else 31))
    seen = {_candidate_key(default_params)}
    attempts = 0
    while len(candidates) < max_candidates:
        attempts += 1
        if attempts > max(1000, max_candidates * 200):
            raise RuntimeError(f"Could not generate {max_candidates} unique candidates for {model}")
        sampled = {key: _sample_one(rng, spec) for key, spec in model_space.get("space", {}).items()}
        marker = _candidate_key(sampled)
        if marker in seen:
            continue
        seen.add(marker)
        sampled["candidate_id"] = f"{model}_{len(candidates):04d}"
        sampled["source"] = "sampled_simple_hpo"
        candidates.append(sampled)
    return candidates


def candidate_sets(
    models: Iterable[str],
    max_candidates: int,
    seed: int,
    search_space: Mapping[str, Any] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    space = deepcopy(search_space or DEFAULT_SEARCH_SPACE)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for model in models:
        out[model] = generate_model_candidates(model, max_candidates, seed, space)
    return out


def candidate_config_payload(candidates_by_model: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {model: candidates for model, candidates in candidates_by_model.items()}
