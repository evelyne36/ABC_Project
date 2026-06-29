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
