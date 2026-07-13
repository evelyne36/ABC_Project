from __future__ import annotations

import numpy as np

from random600.utils.candidates import candidate_sets, load_search_space

from timeseries_extra_random300.scripts.run_timeseries_benchmark import (
    ADAPTER_ROOT,
    ColumnwiseMultiOutputRegressor,
    build_parser,
    create_tasks,
    evaluate_fixed_task,
    horizon_gap,
    normalize_args,
)
from timeseries_extra_random300.utils.data import DatasetSpec, RawTimeSeries, load_registry, supervised_row_count
from timeseries_extra_random300.utils.splits import fixed_chronological_split


def parse_args(*items: str):
    return normalize_args(build_parser().parse_args(list(items)))


def make_toy_series() -> RawTimeSeries:
    spec = DatasetSpec(
        name="toy",
        family="toy",
        source_repo="local",
        paper_repo="local",
        url="local",
        raw_path="toy.csv",
        format="plain_numeric_csv",
        target_column=-1,
        frequency="d",
    )
    values = np.arange(160, dtype=np.float32).reshape(80, 2)
    return RawTimeSeries(
        name="toy",
        values=values,
        target_index=1,
        columns=["var_0", "target"],
        timestamps=[str(idx) for idx in range(80)],
        spec=spec,
    )


class NoVerboseRegressor:
    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.value_, dtype=np.float32)


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


def test_columnwise_multioutput_regressor_accepts_no_verbose_estimators() -> None:
    estimator = ColumnwiseMultiOutputRegressor(NoVerboseRegressor)
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = np.arange(8, dtype=np.float32).reshape(4, 2)

    estimator.fit(X, y, verbose=False)
    pred = estimator.predict(X)

    assert pred.shape == (4, 2)
    np.testing.assert_allclose(pred[:, 0], np.full(4, 3.0, dtype=np.float32))
    np.testing.assert_allclose(pred[:, 1], np.full(4, 4.0, dtype=np.float32))


def test_tree_benchmark_models_are_accepted() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--models",
        "xgb_cpu",
        "current_metagb",
        "lgbm_cpu",
        "catboost_cpu",
        "--max-candidates",
        "30",
    )

    assert args.models == ["xgb_cpu", "current_metagb", "lgbm_cpu", "catboost_cpu"]


def test_tree_benchmark_candidate_sets_include_all_model_families() -> None:
    search_space = load_search_space(ADAPTER_ROOT / "configs" / "timeseries_search_space.json")
    candidates_by_model, warnings = candidate_sets(
        search_space,
        30,
        0,
        None,
        ["xgb_cpu", "current_metagb", "lgbm_cpu", "catboost_cpu"],
    )

    assert warnings == []
    assert set(candidates_by_model) == {"xgb_cpu", "current_metagb", "lgbm_cpu", "catboost_cpu"}
    assert {model: len(candidates) for model, candidates in candidates_by_model.items()} == {
        "xgb_cpu": 30,
        "current_metagb": 30,
        "lgbm_cpu": 30,
        "catboost_cpu": 30,
    }
    assert candidates_by_model["lgbm_cpu"][0]["candidate_id"] == "lgbm_0001"
    assert candidates_by_model["catboost_cpu"][0]["candidate_id"] == "catboost_0001"
    assert candidates_by_model["lgbm_cpu"][0]["backbone_id"] == candidates_by_model["catboost_cpu"][0]["backbone_id"]


def test_no_valid_fixed_task_summary_keeps_metric_schema() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--models",
        "xgb_cpu",
        "--max-candidates",
        "1",
    )
    series = make_toy_series()
    train_rows, val_rows, test_rows, fixed_split = fixed_chronological_split(
        supervised_row_count(series, args.seq_len, args.pred_len)
    )
    candidates = [
        {
            "learning_rate": 0.1,
            "max_depth": 2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_estimators": 1,
            "max_bin": 64,
            "reg_lambda": 1.0,
            "candidate_id": "bad_0001",
        }
    ]

    _candidate_rows, summary = evaluate_fixed_task(
        xgb=object(),
        args=args,
        series=series,
        model_name="unknown_model",
        candidates=candidates,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        fixed_split=fixed_split,
        fixed_splits_by_seq_len={args.seq_len: (train_rows, val_rows, test_rows, fixed_split)},
        split_records=[],
    )

    assert summary["status"] == "no_valid_candidate"
    assert np.isnan(summary["test_rmse"])
    assert np.isnan(summary["test_mae"])
    assert np.isnan(summary["test_mse"])
    assert np.isnan(summary["total_train_time_sec"])
    assert np.isnan(summary["total_predict_time_sec"])


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


def test_window_summary_feature_mode_is_accepted_for_gbdt_smoke() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--feature-mode",
        "window_summary",
        "--target-mode",
        "sequence",
        "--output-scope",
        "all",
        "--models",
        "xgb_cpu",
        "--max-candidates",
        "1",
    )

    assert args.feature_mode == "window_summary"
    assert args.target_mode == "sequence"


def test_origin_state_feature_mode_accepts_channel_budget() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--feature-mode",
        "origin_state",
        "--origin-state-channel-budget",
        "8",
        "--target-mode",
        "sequence",
        "--output-scope",
        "all",
        "--models",
        "xgb_cpu",
        "--max-candidates",
        "1",
    )

    assert args.feature_mode == "origin_state"
    assert args.origin_state_channel_budget == 8
    assert args.target_mode == "sequence"


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


def test_fixed_protocol_expands_tasks_for_four_tree_models() -> None:
    args = parse_args(
        "--split-protocol",
        "fixed",
        "--datasets",
        "ETTh1",
        "--models",
        "xgb_cpu",
        "current_metagb",
        "lgbm_cpu",
        "catboost_cpu",
        "--seeds",
        "0",
        "--seq-lens",
        "12",
        "--pred-lens",
        "1",
        "--feature-mode",
        "window_summary",
        "--target-mode",
        "sequence",
        "--output-scope",
        "target",
        "--max-raw-rows",
        "300",
        "--max-candidates",
        "30",
    )
    registry = load_registry(ADAPTER_ROOT / "configs" / "dataset_registry.json")
    search_space = load_search_space(ADAPTER_ROOT / "configs" / "timeseries_search_space.json")
    candidates_by_model, _warnings = candidate_sets(search_space, args.max_candidates, 0, None, args.models)

    tasks = create_tasks(args, registry, {0: candidates_by_model})

    assert len(tasks) == 4
    assert {task["model_name"] for task in tasks} == {"xgb_cpu", "current_metagb", "lgbm_cpu", "catboost_cpu"}
    assert all(task["args"]["target_mode"] == "sequence" for task in tasks)
    assert all(task["args"]["output_scope"] == "target" for task in tasks)
    assert all(task["args"]["feature_mode"] == "window_summary" for task in tasks)


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
