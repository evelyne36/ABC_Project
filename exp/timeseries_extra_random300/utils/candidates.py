"""Candidate generation for the time-series tree-model benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crossgb_compat import canonicalize_models
from random300.utils.candidates import (
    candidate_config_payload,
    candidate_sets as tabular_candidate_sets,
    load_search_space,
)


def _baseline_candidates(
    xgb_candidates: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index, xgb_candidate in enumerate(xgb_candidates, start=1):
        params = dict(xgb_candidate)
        params["candidate_id"] = f"{model.removesuffix('_cpu')}_{index:04d}"
        params["source"] = "generated_shared_backbone"
        if model == "catboost_cpu":
            params["l2_leaf_reg"] = float(params.pop("reg_lambda"))
        candidates.append(params)
    return candidates


def candidate_sets(
    search_space: Dict[str, Any],
    max_candidates: int,
    seed: int,
    config_path: Optional[Path],
    models: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    selected = canonicalize_models(
        models or ["xgb_cpu", "crossgb", "lgbm_cpu", "catboost_cpu"]
    )
    baseline_models = {
        model for model in selected if model in {"lgbm_cpu", "catboost_cpu"}
    }
    tabular_models = [
        model
        for model in selected
        if model in {"xgb_cpu", "crossgb", "crossgb_pred0"}
    ]
    generator_models = list(tabular_models)
    if baseline_models and "xgb_cpu" not in generator_models:
        generator_models.append("xgb_cpu")

    generated, warnings = tabular_candidate_sets(
        search_space,
        max_candidates,
        seed,
        config_path,
        generator_models,
    )
    xgb_candidates = generated.get("xgb_cpu", [])
    output: Dict[str, List[Dict[str, Any]]] = {}
    for model in selected:
        if model in baseline_models:
            output[model] = _baseline_candidates(xgb_candidates, model)
        else:
            output[model] = generated[model]
    return output, warnings


__all__ = ["candidate_config_payload", "candidate_sets", "load_search_space"]
