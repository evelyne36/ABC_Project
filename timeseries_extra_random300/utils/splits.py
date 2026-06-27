from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class OrderedSplit:
    fold: int
    train_start: int
    train_end: int
    eval_start: int
    eval_end: int
    gap: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def expanding_window_splits(
    n_samples: int,
    n_splits: int,
    *,
    initial_train_fraction: float = 0.5,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, OrderedSplit]]:
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if n_samples < n_splits + 2:
        raise ValueError(f"Not enough samples for {n_splits} ordered splits: {n_samples}")
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be in (0, 1)")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    initial_train = max(1, int(n_samples * initial_train_fraction))
    remaining = n_samples - initial_train - gap
    if remaining < n_splits:
        initial_train = max(1, n_samples - n_splits - gap)
        remaining = n_samples - initial_train - gap
    if remaining < n_splits:
        raise ValueError(
            f"Not enough post-training samples for {n_splits} splits: "
            f"n_samples={n_samples}, initial_train={initial_train}, gap={gap}"
        )

    eval_size = max(1, remaining // n_splits)
    out: List[Tuple[np.ndarray, np.ndarray, OrderedSplit]] = []
    for fold in range(n_splits):
        train_end = initial_train + fold * eval_size
        eval_start = train_end + gap
        eval_end = eval_start + eval_size
        if fold == n_splits - 1:
            eval_end = n_samples
        if eval_start >= eval_end:
            break
        split = OrderedSplit(
            fold=fold + 1,
            train_start=0,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
            gap=gap,
        )
        train_idx = np.arange(split.train_start, split.train_end, dtype=np.int64)
        eval_idx = np.arange(split.eval_start, split.eval_end, dtype=np.int64)
        out.append((train_idx, eval_idx, split))
    if len(out) != n_splits:
        raise ValueError(f"Could only create {len(out)} splits out of requested {n_splits}")
    return out


def assert_time_ordered(train_idx: np.ndarray, eval_idx: np.ndarray) -> None:
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("train/eval indices must be non-empty")
    if int(train_idx.max()) >= int(eval_idx.min()):
        raise ValueError("Ordered split violation: train index reaches eval period")

