from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from crossgb_ablation.modeling import (
    assert_shared_backbone,
    crossgb_params_from_backbone,
)
from crossgb_ablation.protocols import (
    MAIN_VARIANT_IDS,
    PRED_FRACTIONS,
    build_variants,
    protocol_by_name,
)
from crossgb_ablation.run_ablation import build_parser, create_task_plan


ROOT = Path(__file__).resolve().parents[2]


def test_default_variant_contract_is_exactly_16() -> None:
    variants = build_variants()
    assert len(variants) == 16
    assert len({variant.variant_id for variant in variants}) == 16
    assert tuple(
        variant.variant_id for variant in variants if variant.in_main_table
    ) == MAIN_VARIANT_IDS
    assert {
        variant.pred_fraction
        for variant in variants
        if variant.family == "split_only" and variant.in_pred_sensitivity
    } == set(PRED_FRACTIONS)
    assert {
        variant.pred_fraction
        for variant in variants
        if variant.family == "full" and variant.in_pred_sensitivity
    } == set(PRED_FRACTIONS)


def test_variant_semantics_are_frozen() -> None:
    lookup = {variant.variant_id: variant for variant in build_variants()}
    bias = lookup["bias_p00_sa0"]
    split = lookup["split_p10_sa0"]
    full = lookup["full_p10_sa0"]
    assert (bias.pred_fraction, bias.score_agg, bias.leaf_only_use_bias) == (
        0.0,
        0,
        True,
    )
    assert (split.pred_fraction, split.score_agg, split.leaf_only_use_bias) == (
        1.0,
        0,
        True,
    )
    assert (full.pred_fraction, full.score_agg, full.leaf_only_use_bias) == (
        1.0,
        0,
        False,
    )
    assert {
        lookup[identifier].score_agg_label
        for identifier in ("full_p10_sa0", "full_p10_sa1", "full_p10_sa2")
    } == {"greedy", "mean", "median"}


def test_reduced_grid_removes_only_split_scoreagg_variants() -> None:
    formal = {variant.variant_id for variant in build_variants()}
    reduced = {variant.variant_id for variant in build_variants(False)}
    assert len(formal) == 16
    assert len(reduced) == 14
    assert formal - reduced == {"split_p10_sa1", "split_p10_sa2"}


def test_cli_defaults_to_formal_16_variant_grid() -> None:
    parser = build_parser()
    assert parser.parse_args([]).include_split_scoreagg is True
    assert parser.parse_args(["--no-split-scoreagg"]).include_split_scoreagg is False


def test_crossgb_params_reuse_public_backbone_and_lambda() -> None:
    backbone = {
        "candidate_id": "xgb_0001",
        "backbone_id": "shared_backbone_0001",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_estimators": 200,
        "max_bin": 255,
        "reg_lambda": 0.25,
    }
    split = next(
        variant
        for variant in build_variants()
        if variant.variant_id == "split_p10_sa0"
    )
    params = crossgb_params_from_backbone(backbone, split)
    assert_shared_backbone(backbone, params)
    assert params["crossgb_lambda_l2"] == backbone["reg_lambda"]
    assert params["crossgb_leaf_only_use_bias"] is True


def test_smoke_task_plan_has_expected_domain_units() -> None:
    protocol = protocol_by_name("smoke")
    dataset_root = ROOT.parent / "dataset"
    required_paths = [
        dataset_root / f"TabArena_{dataset.task_type}" / dataset.filename
        for dataset in protocol.real_datasets
    ]
    if any(not path.exists() for path in required_paths):
        pytest.skip("requires downloaded TabArena datasets")
    args = Namespace(
        domains=["real", "synthetic"],
        outer_fold_index=None,
        limit_tasks=None,
        regression_dir=str(dataset_root / "TabArena_regression"),
        binary_dir=str(dataset_root / "TabArena_binary"),
        multiclass_dir=str(dataset_root / "TabArena_multiclass"),
    )
    tasks = create_task_plan(args, protocol)
    expected_real = (
        len(protocol.real_datasets)
        * protocol.real_cv_repeats
        * protocol.outer_folds
    )
    expected_synthetic = (
        len(protocol.synthetic_cases)
        * len(protocol.dimensions)
        * len(protocol.synthetic_data_seeds)
        * protocol.outer_folds
    )
    assert len(tasks) == expected_real + expected_synthetic
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    assert {task["domain"] for task in tasks} == {"real", "synthetic"}


def test_formal_synthetic_protocol_uses_three_evaluation_seeds() -> None:
    protocol = protocol_by_name("formal")
    assert tuple(protocol.synthetic_data_seeds) == (1001, 1002, 1003)
    expected_tasks = (
        len(protocol.synthetic_cases)
        * len(protocol.dimensions)
        * len(protocol.synthetic_data_seeds)
        * protocol.outer_folds
    )
    assert expected_tasks == 210
