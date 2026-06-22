from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional


RANDOM_SEED = 42

DEFAULT_DATASETS = [
    "airfoil_self_noise.csv",
    "Another-Dataset-on-used-Fiat-500.csv",
    "concrete_compressive_strength.csv",
    "diamonds.csv",
    "Food_Delivery_Time.csv",
    "healthcare_insurance_expenses.csv",
    "houses.csv",
    "miami_housing.csv",
    "physiochemical_protein.csv",
    "QSAR_fish_toxicity.csv",
    "wine_quality.csv",
]

PRED_FEATURE_FRACTIONS = [round(i * 0.1, 1) for i in range(1, 11)]
N_ESTIMATORS_VALUES = [200, 400, 600, 800]

EXP2_SHARED_PARAMS_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 6, 8],
    "n_estimators": [200, 400, 600, 800],
    "max_bin": [255],
}

EXP2_COMMON_DATASET_GRIDS = {
    name.replace(".csv", ""): {
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    for name in DEFAULT_DATASETS
}

EXP2_XGB_SPECIFIC_GRID = {
    "min_child_weight": [10],
    "reg_lambda": [0.0, 1.0],
}

EXP2_METAGB_SPECIFIC_GRID = {
    "pred_feature_fraction": [0.2, 0.4, 0.6, 0.8, 1.0],
    "min_data_in_leaf": [10],
}


def sample_log_uniform(np: Any, rng: Any, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_log_uniform_int(np: Any, rng: Any, low: int, high: int) -> int:
    sampled = np.exp(rng.uniform(np.log(low), np.log(high)))
    return int(np.clip(np.round(sampled), low, high))


def attach_n_estimators(np: Any, candidates: List[Dict[str, Any]], rng: Any) -> None:
    assigned = np.tile(
        N_ESTIMATORS_VALUES,
        int(np.ceil(len(candidates) / len(N_ESTIMATORS_VALUES))),
    )[: len(candidates)]
    rng.shuffle(assigned)
    for params, n_estimators in zip(candidates, assigned):
        params["n_estimators"] = int(n_estimators)


def normalize_candidate(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if "colsample" in out and "colsample_bytree" not in out:
        out["colsample_bytree"] = out["colsample"]
    if "min_data_in_leaf" in out and "min_child_weight" not in out:
        out["min_child_weight"] = out["min_data_in_leaf"]
    out.setdefault("learning_rate", 0.05)
    out.setdefault("max_depth", 6)
    out.setdefault("n_estimators", 400)
    out.setdefault("subsample", 0.8)
    out.setdefault("colsample_bytree", 0.8)
    out.setdefault("min_child_weight", 10)
    out.setdefault("max_bin", 255)
    return out


def load_xgb_random_candidates(config_path: Path, np: Any, num_candidates: int, seed: int) -> List[Dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    candidates: List[Dict[str, Any]] = []
    for index in range(1, num_candidates + 1):
        raw = config_data[f"XGBoost_r{index}"].get("hyperparameters", {})
        params = {key: value for key, value in raw.items() if key != "ag_args"}
        candidates.append(normalize_candidate(params))
    attach_n_estimators(np, candidates, np.random.default_rng(seed + 2000))
    return candidates


def generate_exp1_metagb_random_candidates(np: Any, num_candidates: int, seed: int) -> List[Dict[str, Any]]:
    """Generate Exp1 MetaGB random candidates with balanced pred_feature_fraction classes."""
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    rng = np.random.default_rng(seed + 3000)
    candidates: List[Dict[str, Any]] = []
    per_class = max(1, num_candidates // len(PRED_FEATURE_FRACTIONS))

    for pred_feature_fraction in PRED_FEATURE_FRACTIONS:
        for _ in range(per_class):
            if len(candidates) >= num_candidates:
                break
            candidates.append(
                {
                    "pred_feature_fraction": float(pred_feature_fraction),
                    "learning_rate": sample_log_uniform(np, rng, 0.005, 0.1),
                    "max_depth": sample_log_uniform_int(np, rng, 4, 10),
                    "min_child_weight": sample_log_uniform(np, rng, 0.001, 5.0),
                    "subsample": float(rng.uniform(0.6, 1.0)),
                    "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                    "max_bin": 255,
                }
            )

    while len(candidates) < num_candidates:
        pred_feature_fraction = float(rng.choice(PRED_FEATURE_FRACTIONS))
        candidates.append(
            {
                "pred_feature_fraction": pred_feature_fraction,
                "learning_rate": sample_log_uniform(np, rng, 0.005, 0.1),
                "max_depth": sample_log_uniform_int(np, rng, 4, 10),
                "min_child_weight": sample_log_uniform(np, rng, 0.001, 5.0),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "max_bin": 255,
            }
        )

    attach_n_estimators(np, candidates, np.random.default_rng(seed + 4000))
    return [normalize_candidate(candidate) for candidate in candidates]


def generate_exp1_xgb_random_candidates(
    np: Any,
    num_candidates: int,
    seed: int,
    project_root: Path,
) -> List[Dict[str, Any]]:
    config_candidates = [
        project_root / "Exp1_54CV_ordinal" / "Exp_New" / "configs_xgboost.json",
        project_root / "Exp1_54CV_ordinal" / "configs_xgboost.json",
    ]
    for config_path in config_candidates:
        if config_path.exists():
            return load_xgb_random_candidates(config_path, np, num_candidates, seed)

    rng = np.random.default_rng(seed + 5000)
    candidates: List[Dict[str, Any]] = []
    for _ in range(num_candidates):
        candidates.append(
            normalize_candidate(
                {
                    "learning_rate": sample_log_uniform(np, rng, 0.005, 0.1),
                    "max_depth": sample_log_uniform_int(np, rng, 4, 10),
                    "min_child_weight": sample_log_uniform(np, rng, 0.001, 5.0),
                    "subsample": float(rng.uniform(0.6, 1.0)),
                    "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                    "max_bin": 255,
                    "reg_lambda": sample_log_uniform(np, rng, 0.0, 10.0),
                    #"min_child_weight": sample_log_uniform(np, rng, 0.001, 5.0),
                    
                }
            )
        )
    attach_n_estimators(np, candidates, np.random.default_rng(seed + 2000))
    return candidates


def grid_to_candidates(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [normalize_candidate(dict(zip(keys, combo))) for combo in product(*values)]


def exp2_param_grid(dataset_name: str, model_name: str) -> Dict[str, List[Any]]:
    dataset_key = dataset_name.replace(".csv", "")
    grid: Dict[str, List[Any]] = {}
    grid.update(EXP2_SHARED_PARAMS_GRID)
    grid.update(EXP2_COMMON_DATASET_GRIDS.get(dataset_key, {
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }))
    if model_name == "xgb_cpu":
        grid.update(EXP2_XGB_SPECIFIC_GRID)
    elif model_name == "current_metagb":
        grid.update(EXP2_METAGB_SPECIFIC_GRID)
    else:
        raise ValueError(f"Unknown model for Exp2 grid: {model_name}")
    return grid


def candidates_for_experiment(
    np: Any,
    experiment: str,
    model_name: str,
    dataset_name: str,
    max_candidates: int,
    seed: int,
    project_root: Path,
    limit_grid_candidates: bool = False,
) -> List[Dict[str, Any]]:
    if experiment == "exp1":
        if model_name == "xgb_cpu":
            return generate_exp1_xgb_random_candidates(np, max_candidates, seed, project_root)
        if model_name == "current_metagb":
            return generate_exp1_metagb_random_candidates(np, max_candidates, seed)
        raise ValueError(f"Unknown model for Exp1 random search: {model_name}")

    if experiment == "exp2":
        candidates = grid_to_candidates(exp2_param_grid(dataset_name, model_name))
        if limit_grid_candidates and max_candidates > 0:
            return candidates[:max_candidates]
        return candidates

    raise ValueError(f"Unknown experiment: {experiment}")
