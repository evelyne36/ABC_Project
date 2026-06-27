from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "metagb_preflight_adapter" / "outputs"
TASK_DIRS = {
    "regression": ROOT / "dataset" / "TabArena_regression",
    "binary": ROOT / "dataset" / "TabArena_binary",
    "multiclass": ROOT / "dataset" / "TabArena_multiclass",
}


@dataclass
class PreparedDataset:
    X: np.ndarray
    y_numeric: np.ndarray
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


def infer_task_type(dataset: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for task_type, dataset_dir in TASK_DIRS.items():
        if (dataset_dir / dataset).exists():
            return task_type
    raise FileNotFoundError(f"Could not infer task type for dataset: {dataset}")


def dataset_path(task_type: str, dataset: str) -> Path:
    path = TASK_DIRS[task_type] / dataset
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def preprocess_features(feature_df: pd.DataFrame, target: pd.Series, categorical_handling: str) -> tuple[pd.DataFrame, pd.Series]:
    non_missing = target.notna()
    feature_df = feature_df.loc[non_missing].copy()
    target = target.loc[non_missing].copy()

    if categorical_handling == "drop":
        feature_df = feature_df.select_dtypes(include=[np.number])
        feature_df = feature_df.dropna()
        target = target.loc[feature_df.index]
    elif categorical_handling == "onehot":
        feature_df = feature_df.dropna()
        target = target.loc[feature_df.index]
        categorical_cols = list(feature_df.select_dtypes(exclude=[np.number]).columns)
        if categorical_cols:
            feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)
    elif categorical_handling == "ordinal":
        categorical_cols = list(feature_df.select_dtypes(exclude=[np.number]).columns)
        for column in categorical_cols:
            feature_df[column], _ = pd.factorize(feature_df[column], sort=True)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
        combined = feature_df.copy()
        combined["__target__"] = target
        combined = combined.dropna()
        target = combined["__target__"]
        feature_df = combined.drop(columns=["__target__"])
    else:
        raise ValueError("categorical_handling must be one of: drop, onehot, ordinal")

    if feature_df.empty:
        raise ValueError("No feature data remains after preprocessing")
    return feature_df, target


def load_prepared_dataset(task_type: str, dataset: str, categorical_handling: str) -> PreparedDataset:
    df = pd.read_csv(dataset_path(task_type, dataset))
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must contain at least one feature and one target: {dataset}")

    feature_raw = df.iloc[:, :-1]
    target_raw = df.iloc[:, -1]
    raw_numeric = int(feature_raw.select_dtypes(include=[np.number]).shape[1])
    raw_categorical = int(feature_raw.shape[1] - raw_numeric)
    raw_missing = float(df.isna().to_numpy().mean())

    feature_df, target = preprocess_features(feature_raw, target_raw, categorical_handling)
    X = feature_df.to_numpy(dtype=np.float64, copy=True)

    if task_type == "regression":
        y_series = pd.to_numeric(target, errors="coerce")
        valid = y_series.notna().to_numpy()
        X = X[valid]
        y_numeric = y_series.loc[y_series.notna()].to_numpy(dtype=np.float64, copy=True)
        n_classes = 0
        class_imbalance = math.nan
    else:
        y_codes, labels = pd.factorize(target, sort=True)
        y_numeric = y_codes.astype(np.float64, copy=False)
        n_classes = int(len(labels))
        counts = np.bincount(y_codes.astype(np.int64), minlength=max(n_classes, 1))
        class_imbalance = float(counts.max() / counts.sum()) if counts.sum() else math.nan

    return PreparedDataset(
        X=X,
        y_numeric=y_numeric,
        raw_rows=int(df.shape[0]),
        raw_features=int(feature_raw.shape[1]),
        raw_numeric_features=raw_numeric,
        raw_categorical_features=raw_categorical,
        raw_missing_fraction=raw_missing,
        processed_rows=int(X.shape[0]),
        processed_features=int(X.shape[1]),
        n_classes=n_classes,
        class_imbalance=class_imbalance,
    )


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
    """Approximate MetaGB's one-feature leaf score against a constant leaf score.

    This uses a squared-error first-step proxy where the centered target acts like
    a negative gradient. Features are standardized before scoring so the proxy
    measures alignment rather than raw scale.
    """
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
    linear_scores = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float64),
        where=denominators > 0,
    )
    best_linear = float(np.nanmax(linear_scores)) if linear_scores.size else 0.0
    positive_sum = float(np.nansum(linear_scores[linear_scores > 0]))
    concentration = safe_divide(best_linear, positive_sum, default=0.0)
    excess = best_linear - const_score
    lift = safe_divide(best_linear, const_score + 1e-12, default=math.nan)
    return excess, lift, concentration


def shallow_tree_probe(task_type: str, data: PreparedDataset, seed: int) -> dict[str, float]:
    X = data.X
    y = data.y_numeric
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

    sample_cap = min(5000, X.shape[0])
    rng = np.random.default_rng(seed)
    if X.shape[0] > sample_cap:
        rows = rng.choice(X.shape[0], size=sample_cap, replace=False)
        X_fit = X[rows]
        y_fit = y[rows]
    else:
        X_fit = X
        y_fit = y

    global_corr = column_abs_correlations(X_fit, y_fit)
    global_max = float(np.nanmax(global_corr)) if global_corr.size else math.nan
    top5 = float(np.nanmean(np.sort(global_corr)[-min(5, global_corr.size) :])) if global_corr.size else math.nan

    if task_type == "regression":
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(10, int(0.03 * X_fit.shape[0])), random_state=seed)
        tree.fit(X_fit, y_fit)
    else:
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, int(0.03 * X_fit.shape[0])), random_state=seed)
        tree.fit(X_fit, y_fit.astype(np.int64))

    leaf_ids = tree.apply(X_fit)
    split_features = {int(feature) for feature in tree.tree_.feature if int(feature) >= 0}
    local_max_values: list[float] = []
    projected_excess_values: list[float] = []
    projected_lift_values: list[float] = []
    concentration_values: list[float] = []
    decoupled = 0
    considered = 0
    residual = y_fit - float(np.nanmean(y_fit))

    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        if int(mask.sum()) < 10:
            continue
        corr = column_abs_correlations(X_fit[mask], y_fit[mask])
        if corr.size == 0:
            continue
        best_idx = int(np.nanargmax(corr))
        best_corr = float(corr[best_idx])
        local_max_values.append(best_corr)
        considered += 1
        if best_idx not in split_features:
            decoupled += 1
        excess, projected_lift, concentration = metagb_leaf_score_proxy(X_fit[mask], residual[mask])
        if math.isfinite(excess):
            projected_excess_values.append(excess)
        if math.isfinite(projected_lift):
            projected_lift_values.append(projected_lift)
        if math.isfinite(concentration):
            concentration_values.append(concentration)

    local_mean = float(np.nanmean(local_max_values)) if local_max_values else math.nan
    lift = safe_divide(local_mean, global_max, default=math.nan)
    return {
        "global_abs_corr_max": global_max,
        "global_abs_corr_top5_mean": top5,
        "local_abs_corr_max_mean": local_mean,
        "local_linearity_lift": lift,
        "decoupled_leaf_fraction": safe_divide(float(decoupled), float(considered), default=math.nan),
        "projected_gradient_excess_mean": float(np.nanmean(projected_excess_values)) if projected_excess_values else math.nan,
        "projected_gradient_lift_mean": float(np.nanmean(projected_lift_values)) if projected_lift_values else math.nan,
        "predictive_score_concentration_mean": float(np.nanmean(concentration_values)) if concentration_values else math.nan,
        "tree_leaf_count": float(considered),
    }


def static_profile(task_type: str, dataset: str, categorical_handling: str, seed: int) -> dict[str, Any]:
    data = load_prepared_dataset(task_type, dataset, categorical_handling)
    X = data.X
    stds = np.nanstd(X, axis=0) if X.size else np.array([], dtype=np.float64)
    positive_stds = stds[stds > 0]
    zero_fraction = float(np.mean(X == 0.0)) if X.size else math.nan
    scale_cv = safe_divide(float(np.nanstd(positive_stds)), float(np.nanmean(positive_stds)), default=math.nan) if positive_stds.size else math.nan
    probe = shallow_tree_probe(task_type, data, seed)

    row: dict[str, Any] = {
        "task_type": task_type,
        "dataset": dataset,
        "categorical_handling": categorical_handling,
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
        "zero_fraction_processed": zero_fraction,
        "feature_scale_cv": scale_cv,
        "n_classes": data.n_classes,
        "class_imbalance": data.class_imbalance,
    }
    row.update(probe)
    return row


def summarize_labels(summary: pd.DataFrame, alpha_runtime_penalty: float, utility_threshold: float, win_threshold: float) -> pd.DataFrame:
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["task_type", "dataset", "primary_metric", "primary_metric_direction"]
    value_cols = [
        "val_primary_metric_mean",
        "test_primary_metric",
        "final_train_time_sec",
        "total_train_time_sec",
        "selected_pred_feature_fraction",
        "selected_metagb_score_agg",
    ]
    for col in value_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    grouped = ok.groupby(group_cols + ["model"], as_index=False).agg(
        val_primary_metric_mean=("val_primary_metric_mean", "mean"),
        test_primary_metric=("test_primary_metric", "mean"),
        final_train_time_sec=("final_train_time_sec", "mean"),
        total_train_time_sec=("total_train_time_sec", "mean"),
        selected_pred_feature_fraction=("selected_pred_feature_fraction", "mean"),
        selected_metagb_score_agg=("selected_metagb_score_agg", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else math.nan),
    )
    pivot = grouped.pivot_table(
        index=group_cols,
        columns="model",
        values=[
            "val_primary_metric_mean",
            "test_primary_metric",
            "final_train_time_sec",
            "total_train_time_sec",
            "selected_pred_feature_fraction",
            "selected_metagb_score_agg",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    required = {"test_primary_metric__xgb_cpu", "test_primary_metric__current_metagb"}
    if not required.issubset(set(pivot.columns)):
        return pd.DataFrame()

    maximize = pivot["primary_metric_direction"].eq("maximize")
    xgb = pivot["test_primary_metric__xgb_cpu"].astype(float)
    mgb = pivot["test_primary_metric__current_metagb"].astype(float)
    pivot["mgb_absolute_delta"] = np.where(maximize, mgb - xgb, xgb - mgb)
    pivot["mgb_relative_improvement"] = np.where(maximize, (mgb - xgb) / xgb.abs(), (xgb - mgb) / xgb.abs())
    pivot["mgb_win"] = pivot["mgb_relative_improvement"] > win_threshold

    xgb_time = pivot.get("final_train_time_sec__xgb_cpu", pd.Series(np.nan, index=pivot.index)).astype(float)
    mgb_time = pivot.get("final_train_time_sec__current_metagb", pd.Series(np.nan, index=pivot.index)).astype(float)
    pivot["runtime_ratio"] = mgb_time / xgb_time.replace(0, np.nan)
    runtime_penalty = alpha_runtime_penalty * np.log(pivot["runtime_ratio"].replace(0, np.nan))
    pivot["mgb_utility"] = pivot["mgb_relative_improvement"] - runtime_penalty
    pivot["mgb_worth_it"] = pivot["mgb_utility"] > utility_threshold
    return pivot


def recommendation_score(row: pd.Series) -> tuple[float, str]:
    score = 0.0
    lift = finite_float(row.get("local_linearity_lift"))
    local_corr = finite_float(row.get("local_abs_corr_max_mean"))
    decoupled = finite_float(row.get("decoupled_leaf_fraction"))
    feature_row_ratio = finite_float(row.get("feature_row_ratio"), 0.0)
    scale_cv = finite_float(row.get("feature_scale_cv"), 0.0)
    categorical_fraction = finite_float(row.get("raw_categorical_fraction"), 0.0)
    task_type = str(row.get("task_type", ""))

    if math.isfinite(lift):
        score += 1.4 * min(max(lift - 1.0, 0.0), 1.5)
    if math.isfinite(local_corr):
        score += 1.2 * min(local_corr, 0.8)
    if math.isfinite(decoupled):
        score += 1.0 * decoupled
    projected_lift = finite_float(row.get("projected_gradient_lift_mean"))
    concentration = finite_float(row.get("predictive_score_concentration_mean"))
    if math.isfinite(projected_lift):
        score += 0.25 * min(math.log1p(max(projected_lift, 0.0)), 2.0)
    if math.isfinite(concentration):
        score += 0.5 * concentration
    if math.isfinite(lift) and math.isfinite(decoupled) and lift > 1.2 and decoupled < 0.33:
        score -= 0.7
    if task_type == "multiclass":
        score += 0.6
    elif task_type == "binary":
        score -= 0.2

    if feature_row_ratio > 0.2:
        score -= min(1.0, feature_row_ratio)
    if scale_cv > 2.0:
        score -= 0.4
    if categorical_fraction > 0.5 and str(row.get("categorical_handling")) == "drop":
        score -= 0.3

    if score >= 2.1:
        decision = "TRY_MGB_FULL"
    elif score >= 1.1:
        decision = "TRY_MGB_LIGHT"
    else:
        decision = "USE_XGB"
    return score, decision


def build_report(table: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    lines = [
        "# MetaGB Preflight Report",
        "",
        f"Rows: {len(table)}",
        "",
        "## Recommendation Counts",
        "",
    ]
    if "recommendation" in recommendations:
        counts = recommendations["recommendation"].value_counts().sort_index()
        for name, count in counts.items():
            lines.append(f"- {name}: {int(count)}")
    labeled = table[table["mgb_relative_improvement"].notna()].copy()
    lines.extend(["", "## Observed Labels", ""])
    if labeled.empty:
        lines.append("No paired XGB/MetaGB labels were available.")
    else:
        lines.append(f"- Labeled datasets: {len(labeled)}")
        lines.append(f"- Observed wins: {int(labeled['mgb_win'].sum())}/{len(labeled)}")
        lines.append(f"- Mean relative improvement: {labeled['mgb_relative_improvement'].mean():.4f}")
        by_rec = labeled.groupby("recommendation")["mgb_relative_improvement"].agg(["count", "mean", "median"]).reset_index()
        lines.extend(["", "## Relative Improvement By Recommendation", ""])
        lines.append(by_rec.to_markdown(index=False))
        labeled["try_mgb"] = labeled["recommendation"].str.startswith("TRY_MGB")
        confusion = pd.crosstab(labeled["try_mgb"], labeled["mgb_win"], margins=True)
        lines.extend(["", "## Try-vs-Win Confusion", ""])
        lines.append(confusion.to_markdown())
        misses = labeled[labeled["mgb_win"] & ~labeled["try_mgb"]]
        false_tries = labeled[~labeled["mgb_win"] & labeled["try_mgb"]]
        if not misses.empty:
            lines.extend(["", "## Missed Wins", ""])
            lines.append(
                misses[
                    [
                        "dataset",
                        "recommendation",
                        "preflight_score",
                        "mgb_relative_improvement",
                        "local_linearity_lift",
                        "decoupled_leaf_fraction",
                    ]
                ].to_markdown(index=False)
            )
        if not false_tries.empty:
            lines.extend(["", "## False Try Recommendations", ""])
            lines.append(
                false_tries[
                    [
                        "dataset",
                        "recommendation",
                        "preflight_score",
                        "mgb_relative_improvement",
                        "local_linearity_lift",
                        "decoupled_leaf_fraction",
                    ]
                ].to_markdown(index=False)
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a first-pass descriptive scorer, not a validated meta-model.",
            "- The shallow-tree probe is a mechanism proxy for local one-feature responses.",
            "- More completed binary/multiclass/drop/onehot result groups are needed before strong thresholds are credible.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MetaGB preflight dataset table.")
    parser.add_argument("--result-dir", action="append", type=Path, required=True, help="Directory containing summary.csv.")
    parser.add_argument("--categorical-handling", choices=["drop", "onehot", "ordinal"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-runtime-penalty", type=float, default=0.002)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--win-threshold", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[pd.DataFrame] = []
    for result_dir in args.result_dir:
        summary_path = result_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.csv: {summary_path}")
        summaries.append(pd.read_csv(summary_path))
    summary = pd.concat(summaries, ignore_index=True)
    labels = summarize_labels(summary, args.alpha_runtime_penalty, args.utility_threshold, args.win_threshold)

    profile_rows: list[dict[str, Any]] = []
    for item in labels[["task_type", "dataset"]].drop_duplicates().itertuples(index=False):
        profile_rows.append(static_profile(item.task_type, item.dataset, args.categorical_handling, args.seed))
    profiles = pd.DataFrame(profile_rows)
    table = profiles.merge(labels, on=["task_type", "dataset"], how="left")

    scores = table.apply(recommendation_score, axis=1, result_type="expand")
    table["preflight_score"] = scores[0].astype(float)
    table["recommendation"] = scores[1].astype(str)

    rec_cols = [
        "task_type",
        "dataset",
        "categorical_handling",
        "recommendation",
        "preflight_score",
        "mgb_relative_improvement",
        "mgb_win",
        "mgb_worth_it",
        "runtime_ratio",
        "selected_pred_feature_fraction__current_metagb",
        "selected_metagb_score_agg__current_metagb",
        "local_linearity_lift",
        "local_abs_corr_max_mean",
        "decoupled_leaf_fraction",
        "projected_gradient_excess_mean",
        "projected_gradient_lift_mean",
        "predictive_score_concentration_mean",
        "processed_rows",
        "processed_features",
    ]
    recommendations = table[[col for col in rec_cols if col in table.columns]].sort_values(
        ["recommendation", "preflight_score"], ascending=[True, False]
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "preflight_dataset_table.csv", index=False)
    recommendations.to_csv(output_dir / "preflight_recommendations.csv", index=False)
    (output_dir / "preflight_report.md").write_text(build_report(table, recommendations), encoding="utf-8")

    print(f"wrote {output_dir / 'preflight_dataset_table.csv'}")
    print(f"wrote {output_dir / 'preflight_recommendations.csv'}")
    print(f"wrote {output_dir / 'preflight_report.md'}")


if __name__ == "__main__":
    main()
