from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATASET_ALIASES = {
    "solar_AL": "solar",
    "solar_AL.txt": "solar",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    source_repo: str
    paper_repo: str
    url: str
    raw_path: str
    format: str
    target_column: str | int
    frequency: str
    date_column: str | None = None
    array_key: str = "data"
    notes: str = ""


@dataclass
class RawTimeSeries:
    name: str
    values: np.ndarray
    target_index: int
    columns: list[str]
    timestamps: list[str]
    spec: DatasetSpec

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass
class FoldArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    scaler_fit_start: int
    scaler_fit_end: int
    train_target_start: int
    train_target_end: int
    eval_target_start: int
    eval_target_end: int


def load_registry(path: Path) -> dict[str, DatasetSpec]:
    with path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Dict[str, Any]] = json.load(handle)
    return {name: DatasetSpec(name=name, **item) for name, item in payload.items()}


def resolve_dataset_name(registry: dict[str, DatasetSpec], name: str) -> str:
    if name in registry:
        return name
    alias = DATASET_ALIASES.get(name)
    if alias in registry:
        return alias
    available = sorted(set(registry) | set(DATASET_ALIASES))
    raise KeyError(f"Unknown dataset {name}. Available: {', '.join(available)}")


def raw_path_for(root: Path, spec: DatasetSpec) -> Path:
    return root / "data" / "raw" / spec.raw_path


def load_raw_series(root: Path, spec: DatasetSpec) -> RawTimeSeries:
    path = raw_path_for(root, spec)
    if not path.exists():
        raise FileNotFoundError(f"Dataset {spec.name} is missing: {path}")

    if spec.format == "csv_with_date":
        df = pd.read_csv(path)
        if spec.date_column not in df.columns:
            raise ValueError(f"{spec.name} missing date column {spec.date_column}")
        df[spec.date_column] = pd.to_datetime(df[spec.date_column])
        df = df.sort_values(spec.date_column).reset_index(drop=True)
        timestamps = df[spec.date_column].astype(str).tolist()
        numeric_df = df.drop(columns=[spec.date_column]).apply(pd.to_numeric, errors="coerce")
        target_index = list(numeric_df.columns).index(str(spec.target_column))
        columns = [str(col) for col in numeric_df.columns]
    elif spec.format == "plain_numeric_csv":
        numeric_df = pd.read_csv(path, header=None)
        numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
        columns = [f"var_{idx}" for idx in range(numeric_df.shape[1])]
        numeric_df.columns = columns
        target_index = int(spec.target_column)
        if target_index < 0:
            target_index = len(columns) + target_index
        timestamps = [str(idx) for idx in range(len(numeric_df))]
    elif spec.format == "npz_array":
        loaded = np.load(path)
        key = spec.array_key
        if key not in loaded.files:
            raise ValueError(f"{spec.name} missing npz array {key}; available={loaded.files}")
        values = np.asarray(loaded[key], dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim == 3:
            values = values.reshape(values.shape[0], values.shape[1] * values.shape[2])
        elif values.ndim != 2:
            raise ValueError(f"{spec.name} unsupported npz array shape {values.shape}")
        columns = [f"var_{idx}" for idx in range(values.shape[1])]
        numeric_df = pd.DataFrame(values, columns=columns)
        target_index = int(spec.target_column)
        if target_index < 0:
            target_index = len(columns) + target_index
        timestamps = [str(idx) for idx in range(len(numeric_df))]
    else:
        raise ValueError(f"Unsupported dataset format: {spec.format}")

    numeric_df = numeric_df.dropna(axis=0).reset_index(drop=True)
    if len(numeric_df) < 10:
        raise ValueError(f"{spec.name} has too few numeric rows after preprocessing")
    timestamps = timestamps[: len(numeric_df)]
    values = numeric_df.to_numpy(dtype=np.float32, copy=True)
    return RawTimeSeries(
        name=spec.name,
        values=values,
        target_index=target_index,
        columns=columns,
        timestamps=timestamps,
        spec=spec,
    )


def supervised_row_count(series: RawTimeSeries, seq_len: int, pred_len: int) -> int:
    count = series.n_rows - seq_len - pred_len + 1
    if count <= 0:
        raise ValueError(
            f"{series.name} too short for seq_len={seq_len}, pred_len={pred_len}: "
            f"rows={series.n_rows}"
        )
    return int(count)


def target_positions(row_indices: Sequence[int], seq_len: int, pred_len: int) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    return rows + seq_len + pred_len - 1


def target_bounds(
    row_indices: Sequence[int],
    seq_len: int,
    pred_len: int,
    target_mode: str = "direct",
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64)
    if target_mode == "direct":
        positions = target_positions(rows, seq_len, pred_len)
        return positions, positions + 1
    if target_mode == "sequence":
        starts = rows + seq_len
        return starts, starts + pred_len
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def _calendar_features(timestamps: list[str], positions: np.ndarray) -> np.ndarray:
    sample = [str(timestamps[int(pos)]) for pos in positions[: min(len(positions), 8)]]
    if sample and all(item.isdigit() for item in sample):
        return np.empty((len(positions), 0), dtype=np.float32)
    parsed = pd.to_datetime([timestamps[int(pos)] for pos in positions], errors="coerce")
    if parsed.isna().all():
        return np.empty((len(positions), 0), dtype=np.float32)
    frame = pd.DataFrame(
        {
            "month": parsed.month.to_numpy(dtype=np.float32) / 12.0,
            "day": parsed.day.to_numpy(dtype=np.float32) / 31.0,
            "weekday": parsed.weekday.to_numpy(dtype=np.float32) / 6.0,
            "hour": parsed.hour.to_numpy(dtype=np.float32) / 23.0,
        }
    )
    frame = frame.fillna(0.0)
    return frame.to_numpy(dtype=np.float32, copy=True)


def _make_xy(
    scaled_values: np.ndarray,
    timestamps: list[str],
    row_indices: Sequence[int],
    target_index: int,
    seq_len: int,
    pred_len: int,
    include_time_features: bool,
    target_mode: str,
    output_scope: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64)
    X = np.empty((len(rows), seq_len * scaled_values.shape[1]), dtype=np.float32)
    if output_scope == "target":
        output_indices = np.asarray([target_index], dtype=np.int64)
    elif output_scope == "all":
        output_indices = np.arange(scaled_values.shape[1], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported output_scope: {output_scope}")
    if target_mode == "direct":
        y = np.empty(len(rows), dtype=np.float32)
    elif target_mode == "sequence":
        y = np.empty((len(rows), pred_len * len(output_indices)), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    for out_idx, row in enumerate(rows):
        start = int(row)
        stop = start + seq_len
        target_pos = stop + pred_len - 1
        X[out_idx, :] = scaled_values[start:stop].reshape(-1)
        if target_mode == "direct":
            y[out_idx] = scaled_values[target_pos, target_index]
        else:
            y[out_idx, :] = scaled_values[stop : stop + pred_len, output_indices].reshape(-1)
    if include_time_features:
        cal = _calendar_features(timestamps, target_positions(rows, seq_len, pred_len))
        if cal.shape[1] > 0:
            X = np.concatenate([X, cal], axis=1)
    return X, y


def make_fold_arrays(
    series: RawTimeSeries,
    train_rows: Sequence[int],
    eval_rows: Sequence[int],
    *,
    seq_len: int,
    pred_len: int,
    include_time_features: bool = True,
    target_mode: str = "direct",
    output_scope: str = "target",
) -> FoldArrays:
    train_rows_arr = np.asarray(train_rows, dtype=np.int64)
    eval_rows_arr = np.asarray(eval_rows, dtype=np.int64)
    if len(train_rows_arr) == 0 or len(eval_rows_arr) == 0:
        raise ValueError("train_rows and eval_rows must be non-empty")
    train_target_start, train_target_end = target_bounds(train_rows_arr, seq_len, pred_len, target_mode)
    eval_target_start, eval_target_end = target_bounds(eval_rows_arr, seq_len, pred_len, target_mode)
    scaler_fit_start = 0
    scaler_fit_end = int(train_target_end.max())
    scaler = StandardScaler()
    scaler.fit(series.values[scaler_fit_start:scaler_fit_end])
    scaled_values = scaler.transform(series.values).astype(np.float32, copy=False)
    X_train, y_train = _make_xy(
        scaled_values,
        series.timestamps,
        train_rows_arr,
        series.target_index,
        seq_len,
        pred_len,
        include_time_features,
        target_mode,
        output_scope,
    )
    X_eval, y_eval = _make_xy(
        scaled_values,
        series.timestamps,
        eval_rows_arr,
        series.target_index,
        seq_len,
        pred_len,
        include_time_features,
        target_mode,
        output_scope,
    )
    return FoldArrays(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        scaler_fit_start=scaler_fit_start,
        scaler_fit_end=scaler_fit_end,
        train_target_start=int(train_target_start.min()),
        train_target_end=int(train_target_end.max()),
        eval_target_start=int(eval_target_start.min()),
        eval_target_end=int(eval_target_end.max()),
    )
