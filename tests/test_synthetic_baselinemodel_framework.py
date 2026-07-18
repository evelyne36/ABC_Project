from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from random300_baselinemodel.utils.candidates import candidate_sets, load_search_space
from synthetic_baselinemodel.protocols import (
    BASELINE_DIMENSIONS,
    BASELINE_MODELS,
    Protocol,
    protocol_by_name,
)
from synthetic_baselinemodel.run_benchmark import (
    build_split_manifest,
    canonical_hash,
    create_tasks,
    pending_tasks,
    run_task,
    search_space_audit,
)
from synthetic_metagb.protocols import main_experiment_instances


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_SPACE_PATH = (
    ROOT / "synthetic_baselinemodel" / "configs" / "default_search_space.json"
)
BASELINE_SPACE_PATH = (
    ROOT / "random300_baselinemodel" / "configs" / "default_search_space.json"
)


def test_full_protocol_has_frozen_random300_nested_cv_budget() -> None:
    protocol = protocol_by_name("synthetic_baseline_random300_single_seed")
    payload = protocol.to_dict()

    assert protocol.models == BASELINE_MODELS
    assert protocol.data_seeds == [42]
    assert protocol.outer_folds == 5
    assert protocol.inner_folds == 4
    assert protocol.max_candidates == 300
    assert BASELINE_DIMENSIONS == [10, 20]
    assert len(protocol.scenarios) == 38
    assert {item.dimension for item in protocol.scenarios} == {10, 20}
    assert payload["expected_task_count"] == 570
    assert payload["expected_candidate_row_count"] == 171_000
    assert payload["planned_model_fit_count"] == 684_570


def test_only_requested_tree_baselines_are_enabled() -> None:
    assert BASELINE_MODELS == ["lightgbm", "catboost", "random_forest"]
    assert "linear" not in BASELINE_MODELS
    assert "extra_trees" not in BASELINE_MODELS
    assert "torchmlp" not in BASELINE_MODELS
    assert "realmlp" not in BASELINE_MODELS
    assert "tabm" not in BASELINE_MODELS


def test_search_spaces_match_public_baseline_project_exactly() -> None:
    synthetic_space = json.loads(SYNTHETIC_SPACE_PATH.read_text(encoding="utf-8"))
    baseline_space = json.loads(BASELINE_SPACE_PATH.read_text(encoding="utf-8"))

    assert set(synthetic_space) == set(BASELINE_MODELS)
    for model in BASELINE_MODELS:
        assert synthetic_space[model] == baseline_space[model]

    candidates = candidate_sets(BASELINE_MODELS, 300, 42, synthetic_space)
    audit = search_space_audit(synthetic_space, candidates)
    assert audit["passed"]
    assert set(audit["candidate_counts"].values()) == {300}
    assert all(audit["shared_boosting_space_checks"].values())


def test_split_manifest_is_shared_across_models_and_reproducible() -> None:
    base = main_experiment_instances(n_samples=80)[0]
    protocol = Protocol(
        name="manifest_test",
        scenarios=[base],
        data_seeds=[42],
        outer_folds=2,
        inner_folds=2,
        max_candidates=2,
        models=list(BASELINE_MODELS),
    )
    first = build_split_manifest(protocol, 314159)
    second = build_split_manifest(protocol, 314159)

    assert first == second
    assert first["manifest_hash"] == canonical_hash(first["entries"])
    assert len(first["entries"]) == 2
    assert all("model" not in entry for entry in first["entries"])


def test_task_creation_and_resume_shards(tmp_path: Path) -> None:
    scenario = main_experiment_instances(n_samples=80)[0]
    protocol = Protocol(
        name="task_test",
        scenarios=[scenario],
        data_seeds=[42],
        outer_folds=2,
        inner_folds=2,
        max_candidates=2,
        models=["lightgbm", "random_forest"],
    )
    search_space = load_search_space(SYNTHETIC_SPACE_PATH)
    candidates = candidate_sets(protocol.models, 2, 42, search_space)
    args = SimpleNamespace(cv_seed=1, model_seed=2, n_jobs=1)
    tasks = create_tasks(args, protocol, candidates)

    assert len(tasks) == 4
    completed = tasks[0]
    shard = {
        "config_hash": "test-config",
        "candidate_rows": [{}, {}],
        "fold_row": {"task_id": completed["task_id"], "status": "ok"},
    }
    (tmp_path / f"{completed['task_id']}.json").write_text(
        json.dumps(shard), encoding="utf-8"
    )
    remaining = pending_tasks(tasks, tmp_path, "test-config")
    assert len(remaining) == 3
    assert completed["task_id"] not in {item["task_id"] for item in remaining}
    assert len(pending_tasks(tasks, tmp_path, "different-config")) == 4

    shard["fold_row"]["status"] = "no_valid_candidate"
    (tmp_path / f"{completed['task_id']}.json").write_text(
        json.dumps(shard), encoding="utf-8"
    )
    assert len(pending_tasks(tasks, tmp_path, "test-config")) == 4


def test_tasks_are_model_first_then_feature_10_before_20() -> None:
    scenarios = [
        item
        for item in main_experiment_instances(n_samples=80)
        if item.case == "regime_slope" and item.dimension in {10, 20}
    ]
    protocol = Protocol(
        name="ordering_test",
        scenarios=scenarios,
        data_seeds=[42],
        outer_folds=2,
        inner_folds=2,
        max_candidates=2,
        models=list(BASELINE_MODELS),
    )
    search_space = load_search_space(SYNTHETIC_SPACE_PATH)
    candidates = candidate_sets(protocol.models, 2, 42, search_space)
    args = SimpleNamespace(cv_seed=1, model_seed=2, n_jobs=1)

    tasks = create_tasks(args, protocol, candidates)
    observed = [
        (task["model"], task["scenario"]["dimension"], task["outer_index"])
        for task in tasks
    ]

    assert observed == [
        ("lightgbm", 10, 0),
        ("lightgbm", 10, 1),
        ("lightgbm", 20, 0),
        ("lightgbm", 20, 1),
        ("catboost", 10, 0),
        ("catboost", 10, 1),
        ("catboost", 20, 0),
        ("catboost", 20, 1),
        ("random_forest", 10, 0),
        ("random_forest", 10, 1),
        ("random_forest", 20, 0),
        ("random_forest", 20, 1),
    ]


def test_random_forest_task_runs_end_to_end() -> None:
    scenario = main_experiment_instances(n_samples=80)[0]
    protocol = Protocol(
        name="random_forest_test",
        scenarios=[scenario],
        data_seeds=[42],
        outer_folds=2,
        inner_folds=2,
        max_candidates=2,
        models=["random_forest"],
    )
    search_space = load_search_space(SYNTHETIC_SPACE_PATH)
    candidates = candidate_sets(protocol.models, 2, 42, search_space)
    args = SimpleNamespace(cv_seed=1, model_seed=2, n_jobs=1)
    task = create_tasks(args, protocol, candidates)[0]

    result = run_task(task)

    assert len(result["candidate_rows"]) == 2
    assert result["fold_row"]["status"] == "ok"
    assert result["fold_row"]["model"] == "random_forest"
    assert result["fold_row"]["test_primary_metric"] >= 0.0
