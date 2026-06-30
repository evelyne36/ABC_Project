from __future__ import annotations

from random600.utils.candidates import candidate_sets, load_search_space

from timeseries_extra_random300.scripts.run_timeseries_benchmark import (
    ADAPTER_ROOT,
    build_parser,
    create_tasks,
    horizon_gap,
    normalize_args,
)
from timeseries_extra_random300.utils.data import load_registry


def parse_args(*items: str):
    return normalize_args(build_parser().parse_args(list(items)))


def test_fixed_protocol_accepts_multiple_seeds_windows_and_horizons() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--seeds",
        "0",
        "1",
        "--seq-lens",
        "6",
        "12",
        "--pred-lens",
        "1",
        "24",
        "--models",
        "xgb_cpu",
        "--max-candidates",
        "1",
    )

    assert args.split_protocol == "fixed"
    assert args.seeds == [0, 1]
    assert args.seq_lens == [6, 12]
    assert args.pred_lens == [1, 24]
    assert args.seq_len == 6
    assert args.pred_len == 1


def test_ltsf_sequence_features_m_arguments_are_recorded() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--target-mode",
        "sequence",
        "--output-scope",
        "all",
        "--seq-lens",
        "12",
        "24",
        "--pred-lens",
        "96",
        "192",
        "--models",
        "xgb_cpu",
        "--max-candidates",
        "1",
    )

    assert args.target_mode == "sequence"
    assert args.output_scope == "all"
    assert args.pred_lens == [96, 192]


def test_tabpfn_ts_feature_mode_uses_sequence_horizon_gap() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--feature-mode",
        "tabpfn_ts",
        "--target-mode",
        "sequence",
        "--output-scope",
        "all",
        "--pred-len",
        "24",
        "--models",
        "xgb_cpu",
        "--max-candidates",
        "1",
    )

    assert args.feature_mode == "tabpfn_ts"
    assert horizon_gap(args) == 23


def test_fixed_protocol_expands_tasks_by_seed_seq_len_pred_len_dataset_and_model() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--datasets",
        "ETTh1",
        "--models",
        "xgb_cpu",
        "--seeds",
        "0",
        "1",
        "--seq-lens",
        "6",
        "12",
        "--pred-lens",
        "1",
        "24",
        "--max-raw-rows",
        "300",
        "--max-candidates",
        "1",
    )
    registry = load_registry(ADAPTER_ROOT / "configs" / "dataset_registry.json")
    search_space = load_search_space(ADAPTER_ROOT / "configs" / "timeseries_search_space.json")
    candidates_by_model, _warnings = candidate_sets(search_space, args.max_candidates, 0, None, args.models)

    tasks = create_tasks(args, registry, {0: candidates_by_model, 1: candidates_by_model})

    assert len(tasks) == 8
    assert {task["args"]["split_protocol"] for task in tasks} == {"fixed"}
    assert {task["args"]["seed"] for task in tasks} == {0, 1}
    assert {task["args"]["seq_len"] for task in tasks} == {6, 12}
    assert {task["args"]["pred_len"] for task in tasks} == {1, 24}
    assert all("__seed" in task["task_id"] and "__s" in task["task_id"] and "__p" in task["task_id"] for task in tasks)
    assert all("fixed_split" in task for task in tasks)


def test_fixed_protocol_orders_tasks_by_seed_outermost() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--datasets",
        "ETTh1",
        "ETTh2",
        "--models",
        "xgb_cpu",
        "current_metagb",
        "--seeds",
        "0",
        "1",
        "--seq-lens",
        "6",
        "12",
        "--pred-lens",
        "1",
        "24",
        "--max-raw-rows",
        "300",
        "--max-candidates",
        "30",
    )
    registry = load_registry(ADAPTER_ROOT / "configs" / "dataset_registry.json")
    search_space = load_search_space(ADAPTER_ROOT / "configs" / "timeseries_search_space.json")
    candidates_seed0, _warnings0 = candidate_sets(search_space, args.max_candidates, 0, None, args.models)
    candidates_seed1, _warnings1 = candidate_sets(search_space, args.max_candidates, 1, None, args.models)

    tasks = create_tasks(args, registry, {0: candidates_seed0, 1: candidates_seed1})

    assert len(tasks) == 32
    assert [task["args"]["seed"] for task in tasks[:16]] == [0] * 16
    assert [task["args"]["seed"] for task in tasks[16:]] == [1] * 16
    assert [task["args"]["seq_len"] for task in tasks[:8]] == [6] * 8
    assert [task["args"]["seq_len"] for task in tasks[8:16]] == [12] * 8


def test_metagb_fixed_score_agg_allows_ten_pred_fraction_smoke() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--models",
        "current_metagb",
        "--max-candidates",
        "10",
        "--metagb-fixed-score-agg",
        "0",
    )

    assert args.max_candidates == 10
    assert args.metagb_fixed_score_agg == 0
