from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


CORE_SCENARIOS = [
    "global_linear",
    "regime_slope",
    "feature_switch",
    "multiplicative",
    "pure_step_control",
    "curved_control",
    "xor_classification",
    "regime_logit",
]

KEY_SCENARIOS = ["regime_slope", "multiplicative", "regime_logit"]
RANDOM600_SCENARIOS = [
    "regime_slope",
    "multiplicative",
    "regime_logit",
    "feature_switch",
    "pure_step_control",
    "curved_control",
]


@dataclass(frozen=True)
class ScenarioInstance:
    name: str
    base_scenario: str
    n_samples: int
    noise: float
    extra_nuisance_features: int = 0
    variant: str = "core"


@dataclass(frozen=True)
class Protocol:
    name: str
    scenarios: List[ScenarioInstance]
    data_seeds: List[int]
    outer_folds: int
    inner_folds: int
    max_candidates: int
    confirmatory: bool

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["scenario_count"] = len(self.scenarios)
        payload["expected_task_count"] = (
            len(self.scenarios) * len(self.data_seeds) * self.outer_folds * 3
        )
        payload["expected_candidate_row_count"] = (
            payload["expected_task_count"] * self.max_candidates
        )
        return payload


def core_instances(n_samples: int = 3000) -> List[ScenarioInstance]:
    return [
        ScenarioInstance(
            name=name,
            base_scenario=name,
            n_samples=n_samples,
            noise=0.15 if name not in {"xor_classification", "regime_logit"} else 0.05,
        )
        for name in CORE_SCENARIOS
    ]


def pressure_instances(n_samples: int = 3000) -> List[ScenarioInstance]:
    out: List[ScenarioInstance] = []
    for name in KEY_SCENARIOS:
        base_noise = 0.15 if name != "regime_logit" else 0.05
        out.extend(
            [
                ScenarioInstance(
                    name=f"{name}__small_n",
                    base_scenario=name,
                    n_samples=500,
                    noise=base_noise,
                    variant="small_n",
                ),
                ScenarioInstance(
                    name=f"{name}__high_noise",
                    base_scenario=name,
                    n_samples=n_samples,
                    noise=0.75 if name != "regime_logit" else 0.35,
                    variant="high_noise",
                ),
                ScenarioInstance(
                    name=f"{name}__high_dim",
                    base_scenario=name,
                    n_samples=n_samples,
                    noise=base_noise,
                    extra_nuisance_features=50,
                    variant="high_dim",
                ),
            ]
        )
    return out


def protocol_by_name(name: str) -> Protocol:
    if name == "smoke":
        smoke_pressure = [
            ScenarioInstance(
                name=item.name,
                base_scenario=item.base_scenario,
                n_samples=item.n_samples,
                noise=item.noise,
                extra_nuisance_features=(
                    10 if item.name == "regime_logit__high_dim" else item.extra_nuisance_features
                ),
                variant=item.variant,
            )
            for item in pressure_instances(n_samples=600)
            if item.name
            in {
                "regime_slope__small_n",
                "multiplicative__high_noise",
                "regime_logit__high_dim",
            }
        ]
        return Protocol(
            name=name,
            scenarios=core_instances(n_samples=600) + smoke_pressure,
            data_seeds=[101, 202],
            outer_folds=2,
            inner_folds=2,
            max_candidates=30,
            confirmatory=False,
        )
    if name == "random300_confirmatory":
        return Protocol(
            name=name,
            scenarios=core_instances() + pressure_instances(),
            data_seeds=list(range(1001, 1011)),
            outer_folds=5,
            inner_folds=4,
            max_candidates=300,
            confirmatory=True,
        )
    if name == "random600_confirmatory":
        keep = set(RANDOM600_SCENARIOS)
        return Protocol(
            name=name,
            scenarios=[item for item in core_instances() if item.name in keep],
            data_seeds=list(range(1001, 1011)),
            outer_folds=5,
            inner_folds=4,
            max_candidates=600,
            confirmatory=True,
        )
    raise ValueError(f"Unknown protocol: {name}")
