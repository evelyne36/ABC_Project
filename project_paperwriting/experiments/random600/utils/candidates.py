from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PRED_FEATURE_FRACTIONS = [round(i * 0.1, 1) for i in range(1, 11)]
SCORE_AGGS = [0, 1, 2]
METAGB_SEARCH_COMBINATIONS = [
    (pred_fraction, score_agg)
    for pred_fraction in PRED_FEATURE_FRACTIONS
    for score_agg in SCORE_AGGS
]
SHARED_PARAM_KEYS = [
    "learning_rate",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "n_estimators",
    "max_bin",
    "lambda_l2",
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
    out["learning_rate"] = float(out.get("learning_rate", 0.05))
    out["max_depth"] = int(round(float(out.get("max_depth", 6))))
    out["subsample"] = float(out.get("subsample", 0.8))
    out["colsample_bytree"] = float(out.get("colsample_bytree", 0.8))
    out["n_estimators"] = int(round(float(out.get("n_estimators", 400))))
    out["max_bin"] = int(round(float(out.get("max_bin", 255))))
    if "lambda_l2" in out:
        out["lambda_l2"] = float(out["lambda_l2"])
    if "reg_lambda" in out:
        out["reg_lambda"] = float(out["reg_lambda"])
    if "metagb_lambda_l2" in out:
        out["metagb_lambda_l2"] = float(out["metagb_lambda_l2"])
    if "l2_leaf_reg" in out:
        out["l2_leaf_reg"] = float(out["l2_leaf_reg"])
    return out


def _sample_shared_candidate(rng: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    params = {key: _sample_one(rng, spec) for key, spec in search_space["shared"].items()}
    return normalize_candidate(params)


def _generate_shared_backbones(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + 5000)
    candidates: List[Dict[str, Any]] = []
    seen = set()
    while len(candidates) < max_candidates:
        candidate = _sample_shared_candidate(rng, search_space)
        marker = json.dumps(candidate, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        candidate["backbone_id"] = f"shared_backbone_{len(candidates) + 1:04d}"
        candidates.append(candidate)
    return candidates


def generate_xgb_candidates(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
    config_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    messages = []
    if config_path is not None:
        messages.append("xgb config file is ignored; candidates are generated from the shared search space")
    candidates: List[Dict[str, Any]] = []
    for index, backbone in enumerate(_generate_shared_backbones(search_space, max_candidates, seed), start=1):
        params = {key: value for key, value in backbone.items() if key != "lambda_l2"}
        params["reg_lambda"] = float(backbone["lambda_l2"])
        params["candidate_id"] = f"xgb_{index:04d}"
        params["source"] = "generated_shared_backbone"
        candidates.append(normalize_candidate(params))
    return candidates, messages


def generate_lgbm_candidates(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index, backbone in enumerate(_generate_shared_backbones(search_space, max_candidates, seed), start=1):
        params = {key: value for key, value in backbone.items() if key != "lambda_l2"}
        params["reg_lambda"] = float(backbone["lambda_l2"])
        params["candidate_id"] = f"lgbm_{index:04d}"
        params["source"] = "generated_shared_backbone"
        candidates.append(normalize_candidate(params))
    return candidates


def generate_catboost_candidates(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index, backbone in enumerate(_generate_shared_backbones(search_space, max_candidates, seed), start=1):
        params = {key: value for key, value in backbone.items() if key != "lambda_l2"}
        params["l2_leaf_reg"] = float(backbone["lambda_l2"])
        params["candidate_id"] = f"catboost_{index:04d}"
        params["source"] = "generated_shared_backbone"
        candidates.append(normalize_candidate(params))
    return candidates


def generate_metagb_candidates(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_candidates % len(METAGB_SEARCH_COMBINATIONS) != 0:
        raise ValueError(
            "--max-candidates must be divisible by 30 for balanced MetaGB "
            "pred_feature_fraction and score_agg combinations"
        )
    candidates: List[Dict[str, Any]] = []
    bucket_counts = {combination: 0 for combination in METAGB_SEARCH_COMBINATIONS}
    for index, backbone in enumerate(_generate_shared_backbones(search_space, max_candidates, seed), start=1):
        combination = METAGB_SEARCH_COMBINATIONS[(index - 1) % len(METAGB_SEARCH_COMBINATIONS)]
        pred_fraction, score_agg = combination
        bucket_counts[combination] += 1
        params = {key: value for key, value in backbone.items() if key != "lambda_l2"}
        params["metagb_lambda_l2"] = float(backbone["lambda_l2"])
        params["pred_feature_fraction"] = float(pred_fraction)
        params["metagb_score_agg"] = int(score_agg)
        params["candidate_id"] = (
            f"metagb_pf{int(pred_fraction * 10):02d}_sa{score_agg}_"
            f"{bucket_counts[combination]:03d}"
        )
        params["source"] = "generated_shared_backbone"
        candidates.append(normalize_candidate(params))
    return candidates


def generate_pred0_candidates(xgb_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, base in enumerate(xgb_candidates, start=1):
        params = {
            key: value
            for key, value in base.items()
            if key not in {"candidate_id", "source", "reg_lambda"}
        }
        params["candidate_id"] = f"pred0_{index:04d}"
        params["metagb_lambda_l2"] = float(base["reg_lambda"])
        params["pred_feature_fraction"] = 0.0
        params["metagb_score_agg"] = 0
        params["source"] = "generated_shared_backbone"
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
    if "lgbm_cpu" in selected:
        out["lgbm_cpu"] = generate_lgbm_candidates(search_space, max_candidates, seed)
    if "catboost_cpu" in selected:
        out["catboost_cpu"] = generate_catboost_candidates(search_space, max_candidates, seed)
    if "current_metagb_pred0" in selected:
        out["current_metagb_pred0"] = generate_pred0_candidates(xgb_candidates)
    return out, messages


def candidate_config_payload(candidates_by_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for model, candidates in candidates_by_model.items():
        payload[model] = candidates
    return payload
