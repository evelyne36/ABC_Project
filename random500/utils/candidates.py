from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PRED_FEATURE_FRACTIONS = [round(i * 0.1, 1) for i in range(1, 11)]
XGB_CONFIG_PREFIX = "XGBoost_r"
SHARED_PARAM_KEYS = [
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "n_estimators",
    "max_bin",
]


def load_search_space(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_one(rng: Any, spec: Dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "choice":
        values = list(spec["values"])
        value = rng.choice(values)
        if isinstance(value, np.generic):
            return value.item()
        return value
    if kind == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if kind == "loguniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if low <= 0.0 or high <= 0.0:
            raise ValueError(f"loguniform bounds must be positive: {spec}")
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    if kind == "logint":
        low = int(spec["low"])
        high = int(spec["high"])
        if low <= 0 or high <= 0:
            raise ValueError(f"logint bounds must be positive: {spec}")
        sampled = np.exp(rng.uniform(np.log(low), np.log(high)))
        return int(np.clip(np.round(sampled), low, high))
    if kind == "int":
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    raise ValueError(f"Unsupported search-space type: {kind}")


def normalize_candidate(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if "colsample" in out and "colsample_bytree" not in out:
        out["colsample_bytree"] = out.pop("colsample")
    if "min_data_in_leaf" in out and "min_child_weight" not in out:
        out["min_child_weight"] = out["min_data_in_leaf"]
    out["learning_rate"] = float(out.get("learning_rate", 0.05))
    out["max_depth"] = int(round(float(out.get("max_depth", 6))))
    out["min_child_weight"] = float(out.get("min_child_weight", 1.0))
    out["subsample"] = float(out.get("subsample", 0.8))
    out["colsample_bytree"] = float(out.get("colsample_bytree", 0.8))
    out["n_estimators"] = int(round(float(out.get("n_estimators", 400))))
    out["max_bin"] = int(round(float(out.get("max_bin", 255))))
    if "reg_lambda" in out:
        out["reg_lambda"] = float(out["reg_lambda"])
    return out


def _sample_shared_candidate(rng: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    params = {key: _sample_one(rng, spec) for key, spec in search_space["shared"].items()}
    return normalize_candidate(params)


def _sample_xgb_candidate(rng: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    params = _sample_shared_candidate(rng, search_space)
    for key, spec in search_space.get("xgb_cpu", {}).items():
        params[key] = _sample_one(rng, spec)
    return normalize_candidate(params)


def _extract_hyperparameters(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "hyperparameters" in raw and isinstance(raw["hyperparameters"], dict):
        raw = raw["hyperparameters"]
    return {key: value for key, value in raw.items() if key != "ag_args"}


def load_xgb_config_candidates(config_path: Path, limit: int) -> Tuple[List[Dict[str, Any]], str]:
    if not config_path.exists():
        return [], f"configs_xgboost.json not found at {config_path}; generated all candidates from search space."
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    candidates: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data[:limit]:
            candidates.append(normalize_candidate(_extract_hyperparameters(item)))
    elif isinstance(data, dict):
        numbered_keys = []
        for key in data:
            if key.startswith(XGB_CONFIG_PREFIX):
                try:
                    numbered_keys.append((int(key.replace(XGB_CONFIG_PREFIX, "")), key))
                except ValueError:
                    pass
        if numbered_keys:
            for _, key in sorted(numbered_keys)[:limit]:
                candidates.append(normalize_candidate(_extract_hyperparameters(data[key])))
        else:
            for key in sorted(data)[:limit]:
                value = data[key]
                if isinstance(value, dict):
                    candidates.append(normalize_candidate(_extract_hyperparameters(value)))
    else:
        raise ValueError(f"Unsupported config format: {config_path}")

    return candidates, f"loaded {len(candidates)} candidates from {config_path}"


def generate_xgb_candidates(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
    config_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rng = np.random.default_rng(seed + 5000)
    messages: List[str] = []
    candidates: List[Dict[str, Any]] = []
    if config_path is not None:
        loaded, message = load_xgb_config_candidates(config_path, min(200, max_candidates))
        candidates.extend(loaded)
        messages.append(message)
    loaded_count = len(candidates)

    seen = {json.dumps(candidate, sort_keys=True) for candidate in candidates}
    while len(candidates) < max_candidates:
        candidate = _sample_xgb_candidate(rng, search_space)
        marker = json.dumps(candidate, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append(candidate)

    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"xgb_{index:04d}"
        candidate["backbone_id"] = candidate["candidate_id"]
        candidate["source"] = "config" if index <= loaded_count else "generated"
    return candidates[:max_candidates], messages


def generate_metagb_candidates(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_candidates % len(PRED_FEATURE_FRACTIONS) != 0:
        raise ValueError("--max-candidates must be divisible by 10 for balanced MetaGB pred fractions")
    rng = np.random.default_rng(seed + 7000)
    per_fraction = max_candidates // len(PRED_FEATURE_FRACTIONS)
    candidates: List[Dict[str, Any]] = []
    for pred_fraction in PRED_FEATURE_FRACTIONS:
        for local_index in range(1, per_fraction + 1):
            params = _sample_shared_candidate(rng, search_space)
            params["pred_feature_fraction"] = float(pred_fraction)
            for key, spec in search_space.get("current_metagb", {}).items():
                params[key] = _sample_one(rng, spec)
            params["candidate_id"] = f"metagb_pf{int(pred_fraction * 10):02d}_{local_index:03d}"
            params["backbone_id"] = f"metagb_backbone_{len(candidates) + 1:04d}"
            params["source"] = "generated"
            candidates.append(normalize_candidate(params))
    return candidates


def generate_pred0_candidates(xgb_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, base in enumerate(xgb_candidates, start=1):
        params = {key: value for key, value in base.items() if key not in {"candidate_id", "source"}}
        params["candidate_id"] = f"pred0_{index:04d}"
        params["backbone_id"] = base["backbone_id"]
        params["pred_feature_fraction"] = 0.0
        params["source"] = "xgb_backbone"
        out.append(normalize_candidate(params))
    return out


def candidate_sets(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
    config_path: Optional[Path],
    models: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    selected = set(models or ["xgb_cpu", "current_metagb", "current_metagb_pred0"])
    needs_xgb_backbone = bool(selected & {"xgb_cpu", "current_metagb_pred0"})
    xgb_candidates: List[Dict[str, Any]] = []
    messages: List[str] = []
    if needs_xgb_backbone:
        xgb_candidates, messages = generate_xgb_candidates(search_space, max_candidates, seed, config_path)

    out: Dict[str, List[Dict[str, Any]]] = {}
    if "xgb_cpu" in selected:
        out["xgb_cpu"] = xgb_candidates
    if "current_metagb" in selected:
        out["current_metagb"] = generate_metagb_candidates(search_space, max_candidates, seed)
    if "current_metagb_pred0" in selected:
        out["current_metagb_pred0"] = generate_pred0_candidates(xgb_candidates)
    return out, messages


def candidate_config_payload(candidates_by_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for model, candidates in candidates_by_model.items():
        payload[model] = candidates
    return payload
