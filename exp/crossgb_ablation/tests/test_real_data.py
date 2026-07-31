from __future__ import annotations

import numpy as np
import pandas as pd

from crossgb_ablation.real_data import fit_transform_features


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": ["a", "b", "a"],
        }
    )
    evaluation = pd.DataFrame(
        {
            "numeric": [4.0, 5.0],
            "category": ["c", "a"],
        }
    )
    return train, evaluation


def test_fold_local_ordinal_marks_unseen_category() -> None:
    train, evaluation = _frames()
    transformed_train, transformed_evaluation = fit_transform_features(
        train,
        evaluation,
        "ordinal",
    )
    assert transformed_train.shape == (3, 2)
    assert transformed_evaluation.shape == (2, 2)
    assert transformed_evaluation[0, 1] == -1.0
    assert set(transformed_train[:, 1]) == {0.0, 1.0}


def test_fold_local_onehot_does_not_add_evaluation_vocabulary() -> None:
    train, evaluation = _frames()
    transformed_train, transformed_evaluation = fit_transform_features(
        train,
        evaluation,
        "onehot",
    )
    assert transformed_train.shape == (3, 3)
    assert transformed_evaluation.shape == (2, 3)
    np.testing.assert_array_equal(
        transformed_evaluation[0, 1:],
        np.zeros(2, dtype=np.float32),
    )


def test_drop_keeps_only_numeric_columns() -> None:
    train, evaluation = _frames()
    transformed_train, transformed_evaluation = fit_transform_features(
        train,
        evaluation,
        "drop",
    )
    np.testing.assert_array_equal(transformed_train[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(transformed_evaluation[:, 0], [4.0, 5.0])
