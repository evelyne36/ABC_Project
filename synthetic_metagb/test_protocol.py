from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from random300.utils.candidates import candidate_sets, load_search_space
from synthetic_metagb.protocols import MAIN_MODELS, protocol_by_name
from synthetic_metagb.run_benchmark import (
    MODELS,
    build_split_manifest,
    candidate_fairness_audit,
    canonical_hash,
    create_tasks,
    pending_tasks,
)
from synthetic_metagb.statistics import bootstrap_mean_ci, holm_adjust, paired_statistics, rank_biserial


ROOT = Path(__file__).resolve().parents[1]


def candidates(count: int):
    search = load_search_space(ROOT / "random300" / "configs" / "default_search_space.json")
    return candidate_sets(search, count, 42, None, MODELS)[0]


def candidates_for_models(count: int, models):
    search = load_search_space(ROOT / "random300" / "configs" / "default_search_space.json")
    return candidate_sets(search, count, 42, None, models)[0]


def test_protocol_sizes_are_frozen() -> None:
    smoke_protocol = protocol_by_name("smoke")
    random300_protocol = protocol_by_name("random300_confirmatory")
    smoke = smoke_protocol.to_dict()
    random300 = random300_protocol.to_dict()
    random600 = protocol_by_name("random600_confirmatory").to_dict()
    assert smoke["expected_task_count"] == 132
    assert smoke["expected_candidate_row_count"] == 3960
    assert random300["max_candidates"] == 300
    assert random600["max_candidates"] == 600
    assert next(
        item.extra_nuisance_features
        for item in smoke_protocol.scenarios
        if item.name == "regime_logit__high_dim"
    ) == 10
    assert {
        item.extra_nuisance_features
        for item in random300_protocol.scenarios
        if item.variant == "high_dim"
    } == {50}


def test_smoke_candidates_cover_complete_joint_cycle_and_are_fair() -> None:
    selected = candidates(30)
    audit = candidate_fairness_audit(selected)
    combinations = Counter(
        (item["pred_feature_fraction"], item["metagb_score_agg"])
        for item in selected["current_metagb"]
    )
    assert audit["passed"]
    assert audit["metagb_joint_combination_count"] == 30
    assert set(combinations.values()) == {1}


def test_main_random300_protocol_is_frozen_and_excludes_pred0() -> None:
    protocol = protocol_by_name("synthetic_main_random300_single_seed")
    payload = protocol.to_dict()
    family_counts = Counter(item.family for item in protocol.scenarios)
    assert protocol.models == MAIN_MODELS
    assert protocol.data_seeds == [42]
    assert len(protocol.scenarios) == 57
    assert payload["expected_task_count"] == 570
    assert payload["expected_candidate_row_count"] == 171_000
    assert payload["expected_candidate_row_count"] * protocol.inner_folds + payload["expected_task_count"] == 684_570
    assert family_counts == {
        "A_interaction": 12,
        "B_linear_in_leaves": 15,
        "C_conditional_predictive": 9,
        "D_sparse_interaction": 9,
        "E_negative_control": 9,
        "F_gaussian_noise": 3,
    }
    assert {item.dimension for item in protocol.scenarios} == {10, 20, 50}
    assert [item.dimension for item in protocol.scenarios] == [10] * 19 + [20] * 19 + [50] * 19


def test_main_framework_smoke_covers_every_base_case_at_dim10() -> None:
    main = protocol_by_name("synthetic_main_random300_single_seed")
    smoke = protocol_by_name("synthetic_main_framework_smoke")
    assert len(smoke.scenarios) == 19
    assert {item.case for item in smoke.scenarios} == {item.case for item in main.scenarios}
    assert {item.dimension for item in smoke.scenarios} == {10}
    assert smoke.to_dict()["expected_task_count"] == 76
    assert smoke.to_dict()["expected_candidate_row_count"] == 2_280


def test_main_models_share_candidate_backbones_without_pred0() -> None:
    selected = candidates_for_models(300, MAIN_MODELS)
    audit = candidate_fairness_audit(selected)
    assert audit["passed"]
    assert set(audit["candidate_counts"]) == set(MAIN_MODELS)
    assert "current_metagb_pred0" not in selected


def test_split_manifest_is_reproducible_and_has_no_model_dimension() -> None:
    protocol = protocol_by_name("smoke")
    first = build_split_manifest(protocol, 314159)
    second = build_split_manifest(protocol, 314159)
    assert first == second
    assert first["manifest_hash"] == canonical_hash(first["entries"])
    assert all("model" not in entry for entry in first["entries"])


def test_pending_tasks_skip_completed_shards(tmp_path: Path) -> None:
    protocol = protocol_by_name("smoke")
    args = type(
        "Args",
        (),
        {
            "cv_seed": 1,
            "model_seed": 2,
            "n_jobs": 1,
            "xgboost_python_package": str(ROOT / "beforereview/xgboost-master/python-package"),
        },
    )()
    tasks = create_tasks(args, protocol, candidates(30))
    completed = tasks[0]
    (tmp_path / f"{completed['task_id']}.json").write_text("{}", encoding="utf-8")
    remaining = pending_tasks(tasks, tmp_path)
    assert len(remaining) == len(tasks) - 1
    assert completed["task_id"] not in {item["task_id"] for item in remaining}


def test_statistics_helpers_and_seed_level_pairing() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert rank_biserial([1.0, 2.0, -0.5]) > 0.0
    low, high = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=7, resamples=1000)
    assert low <= 2.0 <= high
    rows = []
    for seed in range(10):
        rows.extend(
            [
                {"scenario": "s", "task_type": "regression", "primary_metric": "rmse", "data_seed": seed, "model": "xgb_cpu", "seed_primary_metric": 2.0},
                {"scenario": "s", "task_type": "regression", "primary_metric": "rmse", "data_seed": seed, "model": "current_metagb_pred0", "seed_primary_metric": 1.8},
                {"scenario": "s", "task_type": "regression", "primary_metric": "rmse", "data_seed": seed, "model": "current_metagb", "seed_primary_metric": 1.0},
            ]
        )
    stats = paired_statistics(pd.DataFrame(rows), bootstrap_seed=3)
    assert set(stats["reference_model"]) == {"xgb_cpu", "current_metagb_pred0"}
    assert (stats["mean_signed_advantage"] > 0.0).all()
    assert (stats["win_rate"] == 1.0).all()
