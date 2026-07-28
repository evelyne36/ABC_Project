from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from timeseries_extra_random300.utils.data import DatasetSpec, RawTimeSeries, make_fold_arrays


def make_series() -> RawTimeSeries:
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
    values = np.arange(80, dtype=np.float32).reshape(40, 2)
    return RawTimeSeries(
        name="toy",
        values=values,
        target_index=1,
        columns=["a", "target"],
        timestamps=[str(i) for i in range(40)],
        spec=spec,
    )


def make_wide_series() -> RawTimeSeries:
    spec = DatasetSpec(
        name="wide_toy",
        family="toy",
        source_repo="local",
        paper_repo="local",
        url="local",
        raw_path="wide_toy.csv",
        format="plain_numeric_csv",
        target_column=-1,
        frequency="d",
    )
    values = np.arange(120, dtype=np.float32).reshape(20, 6)
    return RawTimeSeries(
        name="wide_toy",
        values=values,
        target_index=5,
        columns=[f"var_{idx}" for idx in range(6)],
        timestamps=[str(i) for i in range(20)],
        spec=spec,
    )


def make_correlated_wide_series() -> RawTimeSeries:
    spec = DatasetSpec(
        name="corr_wide_toy",
        family="toy",
        source_repo="local",
        paper_repo="local",
        url="local",
        raw_path="corr_wide_toy.csv",
        format="plain_numeric_csv",
        target_column=-1,
        frequency="d",
    )
    values = np.zeros((20, 6), dtype=np.float32)
    values[:, 0] = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), 10)
    values[:, 1] = np.linspace(1.0, 0.0, 20)
    values[:, 2] = 0.5
    values[:, 3] = np.arange(20, dtype=np.float32)
    values[:, 4] = 0.0
    values[4:, 4] = np.arange(16, dtype=np.float32) * 10.0
    values[:, 5] = np.arange(20, dtype=np.float32)
    return RawTimeSeries(
        name="corr_wide_toy",
        values=values,
        target_index=5,
        columns=[f"var_{idx}" for idx in range(6)],
        timestamps=[str(i) for i in range(20)],
        spec=spec,
    )


def test_fold_scaler_fits_only_training_target_period() -> None:
    series = make_series()
    train_rows = np.arange(0, 10)
    eval_rows = np.arange(10, 15)
    fold = make_fold_arrays(series, train_rows, eval_rows, seq_len=4, pred_len=2, include_time_features=False)

    assert fold.train_target_end <= fold.eval_target_start
    assert fold.scaler_fit_end == fold.train_target_end
    assert fold.scaler_fit_end <= fold.eval_target_start
    assert fold.X_train.shape == (10, 8)
    assert fold.X_eval.shape == (5, 8)


def test_window_targets_follow_input_window() -> None:
    series = make_series()
    fold = make_fold_arrays(series, [0], [1], seq_len=3, pred_len=1, include_time_features=False)

    assert fold.train_target_start == 3
    assert fold.eval_target_start == 4
    assert fold.X_train.shape[1] == 6


def test_window_summary_mode_appends_leakage_safe_context_summaries() -> None:
    series = make_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=4,
        pred_len=2,
        include_time_features=False,
        feature_mode="window_summary",
    )
    scaler = StandardScaler().fit(series.values[:6])
    scaled = scaler.transform(series.values)
    window = scaled[0:4]
    recent = window[-2:]

    expected_summary = np.concatenate(
        [
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            window[-1],
            window[-1] - window[0],
            recent.mean(axis=0),
            recent.std(axis=0),
            recent[-1],
            np.asarray(
                [
                    window[-1].mean(),
                    window[-1].std(),
                    window[-1].min(),
                    window[-1].max(),
                    window.mean(),
                    window.std(),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)

    assert fold.X_train.shape == (1, 32)
    np.testing.assert_allclose(fold.X_train[0, :8], window.reshape(-1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fold.X_train[0, 8:], expected_summary, rtol=1e-6, atol=1e-6)


def test_origin_state_uses_all_channels_when_budget_allows() -> None:
    series = make_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=4,
        pred_len=2,
        include_time_features=False,
        feature_mode="origin_state",
        origin_state_channel_budget=8,
    )
    scaler = StandardScaler().fit(series.values[:6])
    scaled = scaler.transform(series.values)
    window = scaled[0:4]

    assert fold.X_train.shape == (1, 42)
    np.testing.assert_allclose(fold.X_train[0, :8], window.reshape(-1), rtol=1e-6, atol=1e-6)


def test_origin_state_respects_channel_budget_but_keeps_relational_all_channel_summary() -> None:
    series = make_wide_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=3,
        pred_len=1,
        include_time_features=False,
        feature_mode="origin_state",
        origin_state_channel_budget=3,
    )
    scaler = StandardScaler().fit(series.values[:4])
    scaled = scaler.transform(series.values)
    window = scaled[0:3]
    selected = window[:, [0, 1, 5]]
    relational = np.asarray(
        [
            window[-1].mean(),
            window[-1].std(),
            window[-1].min(),
            window[-1].max(),
            window.mean(),
            window.std(),
            window[-2:].mean(),
            window[-2:].std(),
        ],
        dtype=np.float32,
    )

    assert fold.X_train.shape == (1, 50)
    np.testing.assert_allclose(fold.X_train[0, :9], selected.reshape(-1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fold.X_train[0, -8:], relational, rtol=1e-6, atol=1e-6)


def test_origin_state_selects_channels_by_train_only_target_correlation() -> None:
    series = make_correlated_wide_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=3,
        pred_len=1,
        include_time_features=False,
        feature_mode="origin_state",
        origin_state_channel_budget=3,
    )
    scaler = StandardScaler().fit(series.values[:4])
    scaled = scaler.transform(series.values)
    window = scaled[0:3]
    expected_selected = window[:, [3, 1, 5]]

    np.testing.assert_allclose(fold.X_train[0, :9], expected_selected.reshape(-1), rtol=1e-6, atol=1e-6)


def test_sequence_target_uses_future_horizon_for_single_target() -> None:
    series = make_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=3,
        pred_len=2,
        include_time_features=False,
        target_mode="sequence",
        output_scope="target",
    )
    scaler = StandardScaler().fit(series.values[:5])
    scaled = scaler.transform(series.values)

    assert fold.y_train.shape == (1, 2)
    assert fold.train_target_start == 3
    assert fold.train_target_end == 5
    np.testing.assert_allclose(fold.y_train[0], scaled[3:5, 1])


def test_sequence_target_supports_multivariate_features_m_output() -> None:
    series = make_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=3,
        pred_len=2,
        include_time_features=False,
        target_mode="sequence",
        output_scope="all",
    )
    scaler = StandardScaler().fit(series.values[:5])
    scaled = scaler.transform(series.values)

    assert fold.y_train.shape == (1, 4)
    assert fold.train_target_start == 3
    assert fold.train_target_end == 5
    np.testing.assert_allclose(fold.y_train[0], scaled[3:5, :].reshape(-1))


def test_tabpfn_ts_mode_rolls_sequence_targets_into_scalar_rows() -> None:
    series = make_series()
    fold = make_fold_arrays(
        series,
        [0],
        [1],
        seq_len=3,
        pred_len=2,
        include_time_features=True,
        target_mode="sequence",
        output_scope="all",
        feature_mode="tabpfn_ts",
    )
    scaler = StandardScaler().fit(series.values[:5])
    scaled = scaler.transform(series.values)

    assert fold.X_train.shape == (4, 6)
    assert fold.X_eval.shape == (4, 6)
    assert fold.y_train.shape == (4,)
    np.testing.assert_allclose(fold.y_train, scaled[3:5, :].reshape(-1))


def test_sequence_target_with_horizon_gap_does_not_overlap_eval_targets() -> None:
    series = make_series()
    train_rows = np.arange(0, 10)
    eval_rows = np.arange(12, 15)
    fold = make_fold_arrays(
        series,
        train_rows,
        eval_rows,
        seq_len=4,
        pred_len=3,
        include_time_features=False,
        target_mode="sequence",
        output_scope="target",
    )

    assert fold.train_target_end <= fold.eval_target_start
