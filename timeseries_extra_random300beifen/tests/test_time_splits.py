from __future__ import annotations

from timeseries_extra_random300.utils.splits import (
    assert_time_ordered,
    fixed_chronological_split,
    expanding_window_splits,
)


def test_expanding_window_splits_are_ordered() -> None:
    splits = expanding_window_splits(100, 5, initial_train_fraction=0.5)
    assert len(splits) == 5
    previous_eval_end = 0
    for train_idx, eval_idx, split in splits:
        assert_time_ordered(train_idx, eval_idx)
        assert split.train_start == 0
        assert split.train_end == int(train_idx.max()) + 1
        assert split.eval_start == int(eval_idx.min())
        assert split.eval_end == int(eval_idx.max()) + 1
        assert split.eval_end > previous_eval_end
        previous_eval_end = split.eval_end


def test_fixed_chronological_split_uses_train_val_test_boundaries() -> None:
    train_idx, val_idx, test_idx, split = fixed_chronological_split(100)

    assert list(train_idx[[0, -1]]) == [0, 59]
    assert list(val_idx[[0, -1]]) == [60, 79]
    assert list(test_idx[[0, -1]]) == [80, 99]
    assert split.train_start == 0
    assert split.train_end == 60
    assert split.val_start == 60
    assert split.val_end == 80
    assert split.test_start == 80
    assert split.test_end == 100
    assert_time_ordered(train_idx, val_idx)
    assert_time_ordered(val_idx, test_idx)


def test_fixed_chronological_split_can_reserve_horizon_gap() -> None:
    train_idx, val_idx, test_idx, split = fixed_chronological_split(100, gap=3)

    assert list(train_idx[[0, -1]]) == [0, 59]
    assert list(val_idx[[0, -1]]) == [63, 79]
    assert list(test_idx[[0, -1]]) == [83, 99]
    assert split.train_end == 60
    assert split.val_start == 63
    assert split.val_end == 80
    assert split.test_start == 83
    assert split.test_end == 100
    assert_time_ordered(train_idx, val_idx)
    assert_time_ordered(val_idx, test_idx)
