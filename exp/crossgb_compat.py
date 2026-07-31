"""Canonical CrossGB model names shared by submitted experiment runners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


CROSSGB_MODEL = "crossgb"
CROSSGB_PRED0_MODEL = "crossgb_pred0"

CROSSGB_MODEL_CHOICES = (
    CROSSGB_MODEL,
    CROSSGB_PRED0_MODEL,
)


def canonical_model_name(name: str) -> str:
    """Return a validated public CrossGB model label."""

    return name


def canonicalize_models(models: Iterable[str]) -> list[str]:
    """Validate a model list while preserving order."""

    canonical = [canonical_model_name(model) for model in models]
    if len(canonical) != len(set(canonical)):
        raise ValueError("Model labels must be unique.")
    return canonical


def canonicalize_model_mapping(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated model-keyed mapping."""

    canonical: dict[str, Any] = {}
    for name, value in values.items():
        key = canonical_model_name(name)
        if key in canonical and canonical[key] != value:
            raise ValueError(f"Conflicting model entries resolve to {key!r}")
        canonical[key] = value
    return canonical


def search_space_for_model(search_space: Mapping[str, Any], model: str) -> Any:
    """Read a canonical model section."""

    canonical = canonical_model_name(model)
    return search_space.get(canonical, {})
