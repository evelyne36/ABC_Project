from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

from synthetic_crossgb.protocols import ScenarioInstance, main_experiment_instances


BASELINE_MODELS = [
    "lightgbm",
    "catboost",
    "random_forest",
]
BASELINE_DIMENSIONS = [10, 20]


@dataclass(frozen=True)
class Protocol:
    name: str
    scenarios: List[ScenarioInstance]
    data_seeds: List[int]
    outer_folds: int
    inner_folds: int
    max_candidates: int
    models: List[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scenario_count"] = len(self.scenarios)
        payload["expected_task_count"] = (
            len(self.scenarios) * len(self.data_seeds) * self.outer_folds * len(self.models)
        )
        payload["expected_candidate_row_count"] = (
            payload["expected_task_count"] * self.max_candidates
        )
        payload["planned_model_fit_count"] = (
            payload["expected_candidate_row_count"] * self.inner_folds
            + payload["expected_task_count"]
        )
        return payload


def protocol_by_name(name: str) -> Protocol:
    if name == "synthetic_baseline_random300_single_seed":
        scenarios = [
            item
            for item in main_experiment_instances(n_samples=3000)
            if item.dimension in BASELINE_DIMENSIONS
        ]
        return Protocol(
            name=name,
            scenarios=scenarios,
            data_seeds=[42],
            outer_folds=5,
            inner_folds=4,
            max_candidates=300,
            models=list(BASELINE_MODELS),
        )
    if name == "synthetic_baseline_framework_smoke":
        scenarios = [
            item
            for item in main_experiment_instances(n_samples=180)
            if item.dimension == BASELINE_DIMENSIONS[0]
        ]
        return Protocol(
            name=name,
            scenarios=scenarios,
            data_seeds=[42],
            outer_folds=2,
            inner_folds=2,
            max_candidates=3,
            models=list(BASELINE_MODELS),
        )
    raise ValueError(f"Unknown protocol: {name}")
