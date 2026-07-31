from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence


PRED_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
SCORE_AGG_LABELS = {0: "greedy", 1: "mean", 2: "median"}
MAIN_VARIANT_IDS = (
    "xgb",
    "bias_p00_sa0",
    "split_p10_sa0",
    "full_p10_sa0",
)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    family: str
    leaf_mode: str
    pred_fraction: Optional[float]
    score_agg: Optional[int]
    score_agg_label: str
    leaf_only_use_bias: Optional[bool]
    in_main_table: bool
    in_pred_sensitivity: bool
    in_scoreagg_ablation: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RealDatasetSpec:
    task_type: str
    filename: str

    @property
    def name(self) -> str:
        return self.filename.removesuffix(".csv")


@dataclass(frozen=True)
class SyntheticCaseSpec:
    scenario: str
    role: str


@dataclass(frozen=True)
class AblationProtocol:
    name: str
    version: str
    confirmatory: bool
    real_datasets: Sequence[RealDatasetSpec]
    synthetic_cases: Sequence[SyntheticCaseSpec]
    dimensions: Sequence[int]
    synthetic_data_seeds: Sequence[int]
    real_cv_repeats: int
    outer_folds: int
    inner_folds: int
    xgb_candidates: int
    synthetic_samples: int
    synthetic_backbone_seed: int
    regression_noise: float
    classification_noise: float
    cv_seed: int
    model_seed: int

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["real_dataset_count"] = len(self.real_datasets)
        payload["synthetic_case_count"] = len(self.synthetic_cases)
        payload["default_variants_per_task"] = len(build_variants())
        return payload


REAL_DIAGNOSTIC_PANEL = (
    RealDatasetSpec("regression", "airfoil_self_noise.csv"),
    RealDatasetSpec("regression", "miami_housing.csv"),
    RealDatasetSpec("regression", "physiochemical_protein.csv"),
    RealDatasetSpec("binary", "diabetes.csv"),
    RealDatasetSpec("binary", "credit-g.csv"),
    RealDatasetSpec("binary", "seismic-bumps.csv"),
    RealDatasetSpec("multiclass", "maternal_health_risk.csv"),
    RealDatasetSpec("multiclass", "anneal.csv"),
)

SYNTHETIC_DIAGNOSTIC_PANEL = (
    SyntheticCaseSpec("regime_slope", "mechanism_aligned"),
    SyntheticCaseSpec("feature_switch", "mechanism_aligned"),
    SyntheticCaseSpec("categorical_regime_slope", "mechanism_aligned"),
    SyntheticCaseSpec("sparse_multiplicative", "mechanism_aligned"),
    SyntheticCaseSpec("regime_logit", "mechanism_aligned_classification"),
    SyntheticCaseSpec("pure_step_control", "negative_control"),
    SyntheticCaseSpec("curved_control", "negative_control"),
)


def _crossgb_variant(
    variant_id: str,
    family: str,
    leaf_mode: str,
    pred_fraction: float,
    score_agg: int,
    leaf_only_use_bias: bool,
    *,
    in_main_table: bool = False,
    in_pred_sensitivity: bool = False,
    in_scoreagg_ablation: bool = False,
) -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id,
        family=family,
        leaf_mode=leaf_mode,
        pred_fraction=float(pred_fraction),
        score_agg=int(score_agg),
        score_agg_label=SCORE_AGG_LABELS[int(score_agg)],
        leaf_only_use_bias=bool(leaf_only_use_bias),
        in_main_table=in_main_table,
        in_pred_sensitivity=in_pred_sensitivity,
        in_scoreagg_ablation=in_scoreagg_ablation,
    )


def build_variants(include_split_scoreagg: bool = True) -> List[VariantSpec]:
    variants = [
        VariantSpec(
            variant_id="xgb",
            family="baseline",
            leaf_mode="constant",
            pred_fraction=None,
            score_agg=None,
            score_agg_label="not_applicable",
            leaf_only_use_bias=None,
            in_main_table=True,
            in_pred_sensitivity=False,
            in_scoreagg_ablation=False,
        ),
        _crossgb_variant(
            "bias_p00_sa0",
            "bias",
            "bias_only",
            0.0,
            0,
            True,
            in_main_table=True,
        ),
    ]
    for pred_fraction in PRED_FRACTIONS:
        suffix = int(round(pred_fraction * 10))
        variants.append(
            _crossgb_variant(
                f"split_p{suffix:02d}_sa0",
                "split_only",
                "split_only",
                pred_fraction,
                0,
                True,
                in_main_table=pred_fraction == 1.0,
                in_pred_sensitivity=True,
                in_scoreagg_ablation=(
                    include_split_scoreagg and pred_fraction == 1.0
                ),
            )
        )
        variants.append(
            _crossgb_variant(
                f"full_p{suffix:02d}_sa0",
                "full",
                "predictive",
                pred_fraction,
                0,
                False,
                in_main_table=pred_fraction == 1.0,
                in_pred_sensitivity=True,
                in_scoreagg_ablation=pred_fraction == 1.0,
            )
        )
    for score_agg in (1, 2):
        variants.append(
            _crossgb_variant(
                f"full_p10_sa{score_agg}",
                "full",
                "predictive",
                1.0,
                score_agg,
                False,
                in_scoreagg_ablation=True,
            )
        )
    if include_split_scoreagg:
        for score_agg in (1, 2):
            variants.append(
                _crossgb_variant(
                    f"split_p10_sa{score_agg}",
                    "split_only",
                    "split_only",
                    1.0,
                    score_agg,
                    True,
                    in_scoreagg_ablation=True,
                )
            )
    validate_variants(variants, expected_count=16 if include_split_scoreagg else 14)
    return variants


def validate_variants(
    variants: Sequence[VariantSpec],
    *,
    expected_count: Optional[int] = None,
) -> None:
    if expected_count is not None and len(variants) != expected_count:
        raise AssertionError(f"Expected {expected_count} variants, found {len(variants)}")
    identifiers = [variant.variant_id for variant in variants]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Variant identifiers are not unique")
    lookup = {variant.variant_id: variant for variant in variants}
    if tuple(identifier for identifier in MAIN_VARIANT_IDS if identifier in lookup) != MAIN_VARIANT_IDS:
        raise AssertionError("The four main-ablation variants are incomplete")
    main_ids = tuple(variant.variant_id for variant in variants if variant.in_main_table)
    if main_ids != MAIN_VARIANT_IDS:
        raise AssertionError(f"Unexpected main-ablation order: {main_ids}")
    for family in ("split_only", "full"):
        fractions = sorted(
            float(variant.pred_fraction)
            for variant in variants
            if (
                variant.family == family
                and variant.in_pred_sensitivity
                and variant.pred_fraction is not None
            )
        )
        if fractions != list(PRED_FRACTIONS):
            raise AssertionError(f"Incomplete pred_fraction grid for {family}: {fractions}")
    scoreaggs = sorted(
        int(variant.score_agg)
        for variant in variants
        if (
            variant.family == "full"
            and variant.in_scoreagg_ablation
            and variant.score_agg is not None
        )
    )
    if scoreaggs != [0, 1, 2]:
        raise AssertionError(f"Incomplete Full scoreagg grid: {scoreaggs}")
    for variant in variants:
        if variant.variant_id == "xgb":
            continue
        if variant.pred_fraction is None or not 0.0 <= variant.pred_fraction <= 1.0:
            raise AssertionError(f"Invalid pred_fraction: {variant}")
        if variant.score_agg not in SCORE_AGG_LABELS:
            raise AssertionError(f"Invalid score_agg: {variant}")
        if variant.family == "split_only" and not variant.leaf_only_use_bias:
            raise AssertionError(f"SplitOnly must use bias-only final leaves: {variant}")
        if variant.family == "full" and variant.leaf_only_use_bias:
            raise AssertionError(f"Full must allow predictive leaves: {variant}")


def protocol_by_name(name: str) -> AblationProtocol:
    if name == "smoke":
        return AblationProtocol(
            name="smoke",
            version="1.0",
            confirmatory=False,
            real_datasets=(
                RealDatasetSpec("regression", "airfoil_self_noise.csv"),
                RealDatasetSpec("binary", "diabetes.csv"),
            ),
            synthetic_cases=(
                SyntheticCaseSpec("regime_slope", "mechanism_aligned"),
                SyntheticCaseSpec("pure_step_control", "negative_control"),
            ),
            dimensions=(10,),
            synthetic_data_seeds=(101,),
            real_cv_repeats=1,
            outer_folds=2,
            inner_folds=2,
            xgb_candidates=1,
            synthetic_samples=400,
            synthetic_backbone_seed=9001,
            regression_noise=0.15,
            classification_noise=0.05,
            cv_seed=314159,
            model_seed=271828,
        )
    if name == "formal":
        return AblationProtocol(
            name="formal",
            version="1.3-seed3-grid5-scoreagg16",
            confirmatory=True,
            real_datasets=REAL_DIAGNOSTIC_PANEL,
            synthetic_cases=SYNTHETIC_DIAGNOSTIC_PANEL,
            dimensions=(10, 20),
            synthetic_data_seeds=(1001, 1002, 1003),
            real_cv_repeats=5,
            outer_folds=5,
            inner_folds=4,
            xgb_candidates=300,
            synthetic_samples=3000,
            synthetic_backbone_seed=9001,
            regression_noise=0.15,
            classification_noise=0.05,
            cv_seed=314159,
            model_seed=271828,
        )
    raise ValueError(f"Unknown ablation protocol: {name}")
