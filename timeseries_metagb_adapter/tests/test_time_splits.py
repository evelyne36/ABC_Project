from __future__ import annotations

from timeseries_metagb_adapter.utils.splits import assert_time_ordered, expanding_window_splits


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

