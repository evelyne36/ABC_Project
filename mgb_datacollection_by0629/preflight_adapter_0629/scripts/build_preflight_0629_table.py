from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_metagb.scenarios import generate_instance  # noqa: E402


PROJECT = ROOT / "mgb_datacollection_by0629" / "preflight_adapter_0629"
DEFAULT_RESULTS = ROOT / "mgb_datacollection_by0629" / "derived" / "dataset_level_mgb_vs_xgb.csv"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs"
TABARENA_DIRS = {
    "regression": ROOT / "dataset" / "TabArena_regression",
    "binary": ROOT / "dataset" / "TabArena_binary",
    "multiclass": ROOT / "dataset" / "TabArena_multiclass",
}
TABULAR_EXTRA_DIR = ROOT / "tabular_extra_random300" / "data"
CLASSIFICATION_SYNTHETIC = {"xor_classification", "xor_3way", "regime_logit"}


@dataclass
class PreparedDataset:
    X: np.ndarray
    y: np.ndarray
    raw_rows: int
    raw_features: int
    raw_numeric_features: int
    raw_categorical_features: int
    raw_missing_fraction: float
    processed_rows: int
    processed_features: int
    n_classes: int
    class_imbalance: float


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_divide(num: float, den: float, default: float = math.nan) -> float:
    if den == 0 or not math.isfinite(den):
        return default
    return num / den


def preprocess_features(feature_df: pd.DataFrame, target: pd.Series, handling: str) -> tuple[pd.DataFrame, pd.Series]:
    non_missing = target.notna()
    feature_df = feature_df.loc[non_missing].copy()
    target = target.loc[non_missing].copy()
    if handling == "drop":
        feature_df = feature_df.select_dtypes(include=[np.number]).dropna()
        target = target.loc[feature_df.index]
    elif handling == "onehot":
        feature_df = feature_df.dropna()
        target = target.loc[feature_df.index]
        categorical_cols = list(feature_df.select_dtypes(exclude=[np.number]).columns)
        if categorical_cols:
            feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)
    elif handling == "synthetic":
        pass
    else:
        raise ValueError(f"Unsupported handling: {handling}")
    if feature_df.empty:
        raise ValueError("No features remain after preprocessing")
    return feature_df, target


def resolve_csv_path(family: str, task_type: str, dataset: str) -> Path:
    if family == "tabular_otherdata_0629":
        path = TABULAR_EXTRA_DIR / f"{task_type}__{dataset}"
        if path.exists():
            return path
    path = TABARENA_DIRS[task_type] / dataset
    if path.exists():
        return path
    extra = TABULAR_EXTRA_DIR / f"{task_type}__{dataset}"
    if extra.exists():
        return extra
    raise FileNotFoundError(f"Could not resolve dataset path for {family}/{task_type}/{dataset}")


def load_csv_dataset(family: str, task_type: str, dataset: str, handling: str) -> PreparedDataset:
    path = resolve_csv_path(family, task_type, dataset)
    df = pd.read_csv(path)
    feature_raw = df.iloc[:, :-1]
    target_raw = df.iloc[:, -1]
    raw_numeric = int(feature_raw.select_dtypes(include=[np.number]).shape[1])
    raw_features = int(feature_raw.shape[1])
    raw_categorical = raw_features - raw_numeric
    raw_missing = float(df.isna().to_numpy().mean())
    feature_df, target = preprocess_features(feature_raw, target_raw, handling)
    X = feature_df.to_numpy(dtype=np.float64, copy=True)

    if task_type == "regression":
        y_series = pd.to_numeric(target, errors="coerce")
        valid = y_series.notna().to_numpy()
        X = X[valid]
        y = y_series.loc[y_series.notna()].to_numpy(dtype=np.float64, copy=True)
        n_classes = 0
        imbalance = math.nan
    else:
        y_codes, labels = pd.factorize(target, sort=True)
        y = y_codes.astype(np.float64, copy=False)
        n_classes = int(len(labels))
        counts = np.bincount(y_codes.astype(np.int64), minlength=max(n_classes, 1))
        imbalance = float(counts.max() / counts.sum()) if counts.sum() else math.nan
    return PreparedDataset(
        X=X,
        y=y,
        raw_rows=int(df.shape[0]),
        raw_features=raw_features,
        raw_numeric_features=raw_numeric,
        raw_categorical_features=raw_categorical,
        raw_missing_fraction=raw_missing,
        processed_rows=int(X.shape[0]),
        processed_features=int(X.shape[1]),
        n_classes=n_classes,
        class_imbalance=imbalance,
    )


def synthetic_base_and_dimension(name: str) -> tuple[str, int]:
    match = re.match(r"(?P<base>.+)__dim(?P<dim>\d+)$", name)
    if not match:
        raise ValueError(f"Could not parse synthetic dataset name: {name}")
    return match.group("base"), int(match.group("dim"))


def load_synthetic_dataset(dataset: str) -> PreparedDataset:
    base, dimension = synthetic_base_and_dimension(dataset)
    generator_base = "regime_slope" if base == "noise_regime_slope" else base
    noise = 0.75 if base == "noise_regime_slope" else (0.05 if base in CLASSIFICATION_SYNTHETIC else 0.15)
    X, y = generate_instance(
        generator_base,
        data_seed=42,
        n_samples=3000,
        noise=noise,
        extra_nuisance_features=0,
        target_features=dimension,
    )
    y_float = y.astype(np.float64, copy=False)
    if base in CLASSIFICATION_SYNTHETIC:
        counts = np.bincount(y.astype(np.int64))
        n_classes = int(len(counts))
        imbalance = float(counts.max() / counts.sum())
    else:
        n_classes = 0
        imbalance = math.nan
    return PreparedDataset(
        X=X.astype(np.float64, copy=False),
        y=y_float,
        raw_rows=int(X.shape[0]),
        raw_features=int(X.shape[1]),
        raw_numeric_features=int(X.shape[1]),
        raw_categorical_features=0,
        raw_missing_fraction=0.0,
        processed_rows=int(X.shape[0]),
        processed_features=int(X.shape[1]),
        n_classes=n_classes,
        class_imbalance=imbalance,
    )


def load_dataset(row: pd.Series) -> PreparedDataset:
    if row["family"] == "synthetic_0629":
        return load_synthetic_dataset(str(row["dataset"]))
    return load_csv_dataset(str(row["family"]), str(row["task_type"]), str(row["dataset"]), str(row["encoding"]))


def column_abs_correlations(X: np.ndarray, y: np.ndarray, max_features: int = 512) -> np.ndarray:
    if X.size == 0 or y.size == 0 or X.shape[0] < 3:
        return np.array([], dtype=np.float64)
    if X.shape[1] > max_features:
        variances = np.nanvar(X, axis=0)
        keep = np.argsort(variances)[-max_features:]
        X = X[:, keep]
    y_centered = y - np.nanmean(y)
    y_norm = float(np.sqrt(np.nansum(y_centered * y_centered)))
    if y_norm <= 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    X_centered = X - np.nanmean(X, axis=0)
    x_norm = np.sqrt(np.nansum(X_centered * X_centered, axis=0))
    denom = x_norm * y_norm
    corr = np.divide(
        np.nansum(X_centered * y_centered[:, None], axis=0),
        denom,
        out=np.zeros(X.shape[1], dtype=np.float64),
        where=denom > 0,
    )
    return np.abs(corr)


def metagb_leaf_score_proxy(X: np.ndarray, residual: np.ndarray, lambda_l2: float = 1.0) -> tuple[float, float, float]:
    if X.size == 0 or residual.size == 0 or X.shape[0] < 3:
        return math.nan, math.nan, math.nan
    x_mean = np.nanmean(X, axis=0)
    x_std = np.nanstd(X, axis=0)
    valid = x_std > 0
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    Z = (X[:, valid] - x_mean[valid]) / x_std[valid]
    const_score = float(np.nansum(residual) ** 2 / (float(len(residual)) + lambda_l2))
    numerators = np.nansum(Z * residual[:, None], axis=0) ** 2
    denominators = np.nansum(Z * Z, axis=0) + lambda_l2
    linear_scores = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators > 0)
    best_linear = float(np.nanmax(linear_scores)) if linear_scores.size else 0.0
    positive_sum = float(np.nansum(linear_scores[linear_scores > 0]))
    concentration = safe_divide(best_linear, positive_sum, default=0.0)
    return best_linear - const_score, safe_divide(best_linear, const_score + 1e-12, default=math.nan), concentration


def shallow_tree_probe(task_type: str, data: PreparedDataset, seed: int, sample_cap: int) -> dict[str, float]:
    X, y = data.X, data.y
    if X.shape[0] < 20 or X.shape[1] == 0:
        return {
            "global_abs_corr_max": math.nan,
            "global_abs_corr_top5_mean": math.nan,
            "local_abs_corr_max_mean": math.nan,
            "local_linearity_lift": math.nan,
            "decoupled_leaf_fraction": math.nan,
            "projected_gradient_excess_mean": math.nan,
            "projected_gradient_lift_mean": math.nan,
            "predictive_score_concentration_mean": math.nan,
            "tree_leaf_count": math.nan,
        }
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_cap:
        idx = rng.choice(X.shape[0], size=sample_cap, replace=False)
        X_fit, y_fit = X[idx], y[idx]
    else:
        X_fit, y_fit = X, y

    global_corr = column_abs_correlations(X_fit, y_fit)
    global_max = float(np.nanmax(global_corr)) if global_corr.size else math.nan
    top5 = float(np.nanmean(np.sort(global_corr)[-min(5, global_corr.size) :])) if global_corr.size else math.nan

    min_leaf = max(10, int(0.03 * X_fit.shape[0]))
    if task_type == "regression":
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=min_leaf, random_state=seed)
        tree.fit(X_fit, y_fit)
    else:
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_leaf, random_state=seed)
        tree.fit(X_fit, y_fit.astype(np.int64))

    leaf_ids = tree.apply(X_fit)
    split_features = {int(feature) for feature in tree.tree_.feature if int(feature) >= 0}
    residual = y_fit - float(np.nanmean(y_fit))
    local_max_values: list[float] = []
    projected_excess_values: list[float] = []
    projected_lift_values: list[float] = []
    concentration_values: list[float] = []
    decoupled = 0
    considered = 0
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        if int(mask.sum()) < 10:
            continue
        corr = column_abs_correlations(X_fit[mask], y_fit[mask])
        if corr.size == 0:
            continue
        best_idx = int(np.nanargmax(corr))
        local_max_values.append(float(corr[best_idx]))
        considered += 1
        if best_idx not in split_features:
            decoupled += 1
        excess, lift, concentration = metagb_leaf_score_proxy(X_fit[mask], residual[mask])
        if math.isfinite(excess):
            projected_excess_values.append(excess)
        if math.isfinite(lift):
            projected_lift_values.append(lift)
        if math.isfinite(concentration):
            concentration_values.append(concentration)

    local_mean = float(np.nanmean(local_max_values)) if local_max_values else math.nan
    return {
        "global_abs_corr_max": global_max,
        "global_abs_corr_top5_mean": top5,
        "local_abs_corr_max_mean": local_mean,
        "local_linearity_lift": safe_divide(local_mean, global_max, default=math.nan),
        "decoupled_leaf_fraction": safe_divide(float(decoupled), float(considered), default=math.nan),
        "projected_gradient_excess_mean": float(np.nanmean(projected_excess_values)) if projected_excess_values else math.nan,
        "projected_gradient_lift_mean": float(np.nanmean(projected_lift_values)) if projected_lift_values else math.nan,
        "predictive_score_concentration_mean": float(np.nanmean(concentration_values)) if concentration_values else math.nan,
        "tree_leaf_count": float(considered),
    }


def static_profile(row: pd.Series, seed: int, sample_cap: int) -> dict[str, Any]:
    data = load_dataset(row)
    X = data.X
    stds = np.nanstd(X, axis=0) if X.size else np.array([], dtype=np.float64)
    positive_stds = stds[stds > 0]
    scale_cv = safe_divide(float(np.nanstd(positive_stds)), float(np.nanmean(positive_stds)), default=math.nan) if positive_stds.size else math.nan
    pressure = float(data.processed_rows * (data.processed_features ** 2))
    payload: dict[str, Any] = {
        "raw_rows": data.raw_rows,
        "raw_features": data.raw_features,
        "raw_numeric_features": data.raw_numeric_features,
        "raw_categorical_features": data.raw_categorical_features,
        "raw_numeric_fraction": safe_divide(data.raw_numeric_features, data.raw_features),
        "raw_categorical_fraction": safe_divide(data.raw_categorical_features, data.raw_features),
        "raw_missing_fraction": data.raw_missing_fraction,
        "processed_rows": data.processed_rows,
        "processed_features": data.processed_features,
        "onehot_expansion_ratio": safe_divide(data.processed_features, max(data.raw_features, 1)),
        "feature_row_ratio": safe_divide(data.processed_features, max(data.processed_rows, 1)),
        "zero_fraction_processed": float(np.mean(X == 0.0)) if X.size else math.nan,
        "feature_scale_cv": scale_cv,
        "n_classes": data.n_classes,
        "class_imbalance": data.class_imbalance,
        "runtime_pressure_proxy": pressure,
        "runtime_pressure_log": math.log1p(pressure),
    }
    payload.update(shallow_tree_probe(str(row["task_type"]), data, seed, sample_cap))
    return payload


def observed_label(row: pd.Series, strong_threshold: float) -> dict[str, Any]:
    winner = row["dataset_winner"]
    improvement = finite_float(row["test_improvement"])
    return {
        "observed_winner": winner,
        "observed_mgb_win": winner == "mgb_win",
        "observed_xgb_win": winner == "xgb_win",
        "strong_positive": winner == "mgb_win" and improvement >= strong_threshold,
        "strong_negative": winner == "xgb_win" and improvement <= -strong_threshold,
    }


def build_table(results: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.include_synthetic:
        results = results[results["family"] != "synthetic_0629"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        if args.min_paired_folds and int(row["paired_folds"]) < args.min_paired_folds:
            continue
        base = row.to_dict()
        base["dataset_key"] = "|".join(
            str(base.get(col, "")) for col in ["family", "protocol", "encoding", "task_type", "dataset"]
        )
        try:
            profile = static_profile(row, args.seed, args.sample_cap)
            error = ""
        except Exception as exc:  # keep the audit table complete
            profile = {}
            error = f"{type(exc).__name__}: {exc}"
        base.update(profile)
        base.update(observed_label(row, args.strong_threshold))
        base["profile_error"] = error
        rows.append(base)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 0629 preflight feature table.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-cap", type=int, default=5000)
    parser.add_argument("--min-paired-folds", type=int, default=1)
    parser.add_argument("--strong-threshold", type=float, default=0.01)
    parser.add_argument("--include-synthetic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    table = build_table(results, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "preflight_0629_dataset_table.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    print(f"rows={len(table)} errors={int(table['profile_error'].astype(bool).sum())}")


if __name__ == "__main__":
    main()
