from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timeseries_extra_random300.utils.data import (  # noqa: E402
    load_raw_series,
    load_registry,
    make_fold_arrays,
    supervised_row_count,
)
from timeseries_extra_random300.utils.splits import expanding_window_splits  # noqa: E402


def safe_divide(num: float, den: float, default: float = math.nan) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return default
    return float(num / den)


def clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def bounded_positive_log_lift(value: float, neutral: float, high: float) -> float:
    if not math.isfinite(value) or value <= neutral:
        return 0.0
    return clip01(math.log1p(value - neutral) / math.log1p(high - neutral))


def bounded_fraction(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value) or high <= low:
        return 0.0
    return clip01((value - low) / (high - low))


def column_abs_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.size == 0 or y.size == 0:
        return np.array([], dtype=np.float64)
    X64 = X.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    X_center = X64 - np.nanmean(X64, axis=0)
    y_center = y64 - float(np.nanmean(y64))
    numerator = np.nansum(X_center * y_center[:, None], axis=0)
    denominator = np.sqrt(np.nansum(X_center * X_center, axis=0) * np.nansum(y_center * y_center))
    return np.abs(
        np.divide(
            numerator,
            denominator,
            out=np.zeros(X64.shape[1], dtype=np.float64),
            where=denominator > 0,
        )
    )


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


def shallow_tree_probe(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
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
    tree = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=max(10, int(0.03 * X_fit.shape[0])),
        random_state=seed,
    )
    tree.fit(X_fit, y_fit)
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
        local_max_values.append(float(corr[best_idx]))
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


def mechanism_components(row: pd.Series) -> dict[str, float]:
    projected_lift = float(row.get("projected_gradient_lift_mean", math.nan))
    local_lift = float(row.get("local_linearity_lift", math.nan))
    decoupled = float(row.get("decoupled_leaf_fraction", math.nan))
    concentration = float(row.get("predictive_score_concentration_mean", math.nan))
    local_corr = float(row.get("local_abs_corr_max_mean", math.nan))
    global_corr = float(row.get("global_abs_corr_max", math.nan))
    return {
        "component_linear_leaf_advantage": bounded_positive_log_lift(projected_lift, neutral=1.0, high=8.0),
        "component_cond_pred_decoupling": bounded_fraction(decoupled, low=0.2, high=0.8),
        "component_local_heterogeneity": bounded_positive_log_lift(local_lift, neutral=1.0, high=1.8),
        "component_predictive_concentration": bounded_fraction(concentration, low=0.25, high=0.65),
        "component_signal_support": bounded_fraction(local_corr, low=0.10, high=0.60),
        "component_global_single_feature_penalty": 0.15 * bounded_fraction(global_corr, low=0.75, high=0.95),
    }


def mechanism_score(row: pd.Series) -> tuple[float, dict[str, float]]:
    components = mechanism_components(row)
    weights = {
        "component_linear_leaf_advantage": 0.35,
        "component_cond_pred_decoupling": 0.25,
        "component_local_heterogeneity": 0.20,
        "component_predictive_concentration": 0.10,
        "component_signal_support": 0.10,
    }
    weighted = sum(weights[key] * components[key] for key in weights)
    weighted -= components["component_global_single_feature_penalty"]
    return clip01(weighted / sum(weights.values())), components


def band(score: float) -> str:
    if score >= 0.65:
        return "STRONG_MECHANISM_MATCH"
    if score >= 0.40:
        return "MODERATE_MECHANISM_MATCH"
    return "WEAK_MECHANISM_MATCH"


def outcome_table(summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    pivot = summary.pivot_table(index="dataset", columns="model", values="test_rmse", aggfunc="mean")
    pivot["mgb_relative_rmse_improvement"] = (pivot["xgb_cpu"] - pivot["current_metagb"]) / pivot["xgb_cpu"]
    pivot["winner"] = np.where(pivot["mgb_relative_rmse_improvement"] > 0, "current_metagb", "xgb_cpu")
    selected = (
        summary.groupby(["dataset", "model"])["selected_candidate_index"]
        .agg(lambda values: ",".join(str(int(v)) for v in values))
        .reset_index()
        .pivot(index="dataset", columns="model", values="selected_candidate_index")
    )
    selected.columns = [f"selected_candidates_{col}" for col in selected.columns]
    return pivot.reset_index().merge(selected.reset_index(), on="dataset", how="left")


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    registry = load_registry(Path(args.registry))
    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        spec = registry[dataset_name]
        series = load_raw_series(ADAPTER_ROOT, spec)
        if args.max_raw_rows and series.n_rows > args.max_raw_rows:
            series.values = series.values[: args.max_raw_rows]
            series.timestamps = series.timestamps[: args.max_raw_rows]
        n_rows = supervised_row_count(series, args.seq_len, args.pred_len)
        outer_splits = expanding_window_splits(n_rows, args.outer_folds, initial_train_fraction=0.5)
        train_rows, eval_rows, _ = outer_splits[0]
        fold = make_fold_arrays(
            series,
            train_rows,
            eval_rows,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            include_time_features=args.include_time_features,
        )
        probe = shallow_tree_probe(fold.X_train, fold.y_train, args.seed)
        row = {
            "dataset": dataset_name,
            "family": spec.family,
            "rows_raw_used": series.n_rows,
            "rows_supervised_used": n_rows,
            "analysis_rows": int(fold.X_train.shape[0]),
            "analysis_features": int(fold.X_train.shape[1]),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            **probe,
        }
        score, components = mechanism_score(pd.Series(row))
        row.update(components)
        row["mechanism_alignment_score"] = score
        row["mechanism_band"] = band(score)
        rows.append(row)
    scores = pd.DataFrame(rows)
    outcomes = outcome_table(Path(args.summary))
    return scores.merge(outcomes, on="dataset", how="left")


def build_report(df: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# Time-Series MetaGB Mechanism Audit",
        "",
        "## Contract",
        "",
        "- Mechanism scores are computed from the first outer training block only.",
        "- Smoke outcomes are joined afterwards as retrospective audit evidence.",
        "- This report does not modify or import outputs into `metagb_preflight_adapter/`.",
        "",
        "## Result Table",
        "",
    ]
    table_cols = [
        "dataset",
        "mechanism_band",
        "mechanism_alignment_score",
        "component_linear_leaf_advantage",
        "component_cond_pred_decoupling",
        "component_local_heterogeneity",
        "component_predictive_concentration",
        "component_signal_support",
        "mgb_relative_rmse_improvement",
        "winner",
    ]
    lines.append(df[table_cols].to_markdown(index=False))
    lines.extend(["", "## Interpretation", ""])
    for _, row in df.sort_values("mechanism_alignment_score", ascending=False).iterrows():
        rel = float(row["mgb_relative_rmse_improvement"])
        lines.append(
            f"- {row['dataset']}: score={row['mechanism_alignment_score']:.3f}, "
            f"band={row['mechanism_band']}, smoke winner={row['winner']}, "
            f"relative RMSE improvement={rel:.3f}."
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            f"- summary input: `{args.summary}`",
            f"- max_raw_rows: {args.max_raw_rows}",
            "- The completed smoke used tiny trees and two outer folds, so outcome evidence is weak for final ranking.",
            "- Candidate test metrics are available only for selected candidates, so this cannot estimate an oracle search gap.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    default_results = ADAPTER_ROOT / "results" / "smoke_2datasets_2x2_30_head300_tinytrees_20260624" / "summary.csv"
    parser = argparse.ArgumentParser(description="Analyze time-series mechanism alignment for MetaGB.")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "exchange_rate"])
    parser.add_argument("--registry", default=str(ADAPTER_ROOT / "configs" / "dataset_registry.json"))
    parser.add_argument("--summary", default=str(default_results))
    parser.add_argument("--output-dir", default=str(ADAPTER_ROOT / "outputs" / "mechanism_audit_smoke_head300"))
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--outer-folds", type=int, default=2)
    parser.add_argument("--max-raw-rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-time-features", action="store_true", default=True)
    parser.add_argument("--no-time-features", dest="include_time_features", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = build_rows(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "timeseries_mechanism_alignment_scores.csv"
    report_path = out_dir / "timeseries_mechanism_alignment_report.md"
    df.to_csv(scores_path, index=False)
    report_path.write_text(build_report(df, args), encoding="utf-8")
    print(f"wrote {scores_path}")
    print(f"wrote {report_path}")
    print(df[["dataset", "mechanism_alignment_score", "mechanism_band", "mgb_relative_rmse_improvement", "winner"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

