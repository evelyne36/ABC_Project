from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


@dataclass(frozen=True)
class RawRealDataset:
    features: pd.DataFrame
    target: np.ndarray
    task_type: str
    n_classes: int
    class_labels: List[str]


def load_raw_real_dataset(
    csv_path: Path,
    task_type: str,
) -> RawRealDataset:
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(
            f"Dataset must have at least one feature and one target: {csv_path}"
        )
    features = frame.iloc[:, :-1].copy()
    target = frame.iloc[:, -1].copy()
    valid = target.notna() & ~features.isna().any(axis=1)
    features = features.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)
    if features.empty:
        raise ValueError(f"No complete rows remain after missing-value filtering: {csv_path}")

    if task_type == "regression":
        numeric_target = pd.to_numeric(target, errors="coerce")
        numeric_valid = numeric_target.notna()
        features = features.loc[numeric_valid].reset_index(drop=True)
        values = numeric_target.loc[numeric_valid].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        return RawRealDataset(
            features=features,
            target=values,
            task_type=task_type,
            n_classes=0,
            class_labels=[],
        )

    labels = sorted(str(value) for value in pd.unique(target))
    mapping = {label: index for index, label in enumerate(labels)}
    values = target.astype(str).map(mapping).to_numpy(dtype=np.int64, copy=True)
    n_classes = len(labels)
    if task_type == "binary" and n_classes != 2:
        raise ValueError(
            f"Binary dataset must have exactly 2 classes, got {n_classes}: {csv_path}"
        )
    if task_type == "multiclass" and n_classes <= 2:
        raise ValueError(
            f"Multiclass dataset must have >2 classes, got {n_classes}: {csv_path}"
        )
    return RawRealDataset(
        features=features,
        target=values,
        task_type=task_type,
        n_classes=n_classes,
        class_labels=labels,
    )


def _column_groups(
    train: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    numeric = list(train.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in train.columns if column not in numeric]
    return numeric, categorical


def fit_transform_features(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    categorical_handling: str,
) -> Tuple[np.ndarray, np.ndarray]:
    numeric_columns, categorical_columns = _column_groups(train)
    numeric_train = (
        train[numeric_columns].to_numpy(dtype=np.float32, copy=True)
        if numeric_columns
        else np.empty((len(train), 0), dtype=np.float32)
    )
    numeric_evaluation = (
        evaluation[numeric_columns].to_numpy(dtype=np.float32, copy=True)
        if numeric_columns
        else np.empty((len(evaluation), 0), dtype=np.float32)
    )

    if categorical_handling == "drop":
        transformed_train = numeric_train
        transformed_evaluation = numeric_evaluation
    elif categorical_handling == "ordinal":
        if categorical_columns:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            categorical_train = encoder.fit_transform(
                train[categorical_columns].astype(str)
            )
            categorical_evaluation = encoder.transform(
                evaluation[categorical_columns].astype(str)
            )
            transformed_train = np.column_stack(
                [numeric_train, categorical_train]
            )
            transformed_evaluation = np.column_stack(
                [numeric_evaluation, categorical_evaluation]
            )
        else:
            transformed_train = numeric_train
            transformed_evaluation = numeric_evaluation
    elif categorical_handling == "onehot":
        if categorical_columns:
            try:
                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                )
            except TypeError:
                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                    dtype=np.float32,
                )
            categorical_train = encoder.fit_transform(
                train[categorical_columns].astype(str)
            )
            categorical_evaluation = encoder.transform(
                evaluation[categorical_columns].astype(str)
            )
            transformed_train = np.column_stack(
                [numeric_train, categorical_train]
            )
            transformed_evaluation = np.column_stack(
                [numeric_evaluation, categorical_evaluation]
            )
        else:
            transformed_train = numeric_train
            transformed_evaluation = numeric_evaluation
    else:
        raise ValueError(
            "categorical_handling must be one of: drop, onehot, ordinal"
        )

    if transformed_train.shape[1] == 0:
        raise ValueError(
            "No feature columns remain after fold-local preprocessing"
        )
    if transformed_train.shape[1] != transformed_evaluation.shape[1]:
        raise AssertionError("Train/evaluation feature dimensions differ")
    return (
        np.asarray(transformed_train, dtype=np.float32),
        np.asarray(transformed_evaluation, dtype=np.float32),
    )
