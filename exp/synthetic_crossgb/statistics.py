from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from random300.utils.metrics import primary_metric_direction


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    ordered = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, resamples: int = 10_000
) -> Tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def rank_biserial(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data != 0.0)]
    if data.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(data)).rank(method="average").to_numpy()
    positive = float(ranks[data > 0.0].sum())
    negative = float(ranks[data < 0.0].sum())
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def paired_statistics(seed_rows: pd.DataFrame, bootstrap_seed: int = 42) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario, scenario_df in seed_rows.groupby("scenario"):
        task_type = str(scenario_df["task_type"].iloc[0])
        metric = str(scenario_df["primary_metric"].iloc[0])
        family = str(scenario_df["family"].iloc[0]) if "family" in scenario_df else "legacy"
        case = str(scenario_df["case"].iloc[0]) if "case" in scenario_df else scenario
        dimension = int(scenario_df["dimension"].iloc[0]) if "dimension" in scenario_df else 0
        direction = primary_metric_direction(task_type)
        pivot = scenario_df.pivot(index="data_seed", columns="model", values="seed_primary_metric")
        for reference in ["xgb_cpu", "crossgb_pred0"]:
            required = {"crossgb", reference}
            if not required.issubset(pivot.columns):
                continue
            paired = pivot[list(required)].dropna()
            crossgb = paired["crossgb"].to_numpy(dtype=float)
            baseline = paired[reference].to_numpy(dtype=float)
            advantage = baseline - crossgb if direction == "minimize" else crossgb - baseline
            if direction == "minimize":
                relative = (baseline - crossgb) / np.maximum(np.abs(baseline), 1e-12)
            else:
                relative = advantage.copy()
            ci_low, ci_high = bootstrap_mean_ci(
                advantage, seed=bootstrap_seed + len(rows)
            )
            try:
                p_value = (
                    float(wilcoxon(advantage, alternative="two-sided").pvalue)
                    if len(advantage) >= 2 and np.any(advantage != 0.0)
                    else float("nan")
                )
            except ValueError:
                p_value = float("nan")
            rows.append(
                {
                    "scenario": scenario,
                    "family": family,
                    "case": case,
                    "dimension": dimension,
                    "task_type": task_type,
                    "primary_metric": metric,
                    "reference_model": reference,
                    "n_data_seeds": len(advantage),
                    "mean_crossgb": float(np.mean(crossgb)),
                    "mean_reference": float(np.mean(baseline)),
                    "mean_signed_advantage": float(np.mean(advantage)),
                    "median_signed_advantage": float(np.median(advantage)),
                    "mean_relative_improvement": float(np.mean(relative)),
                    "win_rate": float(np.mean(advantage > 0.0)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_p_value": p_value,
                    "rank_biserial": rank_biserial(advantage),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_adjusted_p_value"] = holm_adjust(out["wilcoxon_p_value"].tolist())
    return out


def seed_level_results(fold_rows: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in ["family", "case", "dimension"]
        if column in fold_rows.columns
    ]
    return (
        fold_rows.groupby(
            [
                "scenario",
                "base_scenario",
                "variant",
                *metadata_columns,
                "task_type",
                "primary_metric",
                "data_seed",
                "model",
            ],
            as_index=False,
        )
        .agg(
            seed_primary_metric=("test_primary_metric", "mean"),
            outer_fold_std=("test_primary_metric", "std"),
            mean_fit_sec=("fit_sec", "mean"),
            mean_predict_sec=("predict_sec", "mean"),
        )
    )


def mark_confirmatory(stats: pd.DataFrame, confirmatory: bool) -> pd.DataFrame:
    out = stats.copy()
    out["confirmatory"] = bool(confirmatory)
    out["interpretation"] = (
        "confirmatory"
        if confirmatory
        else "smoke_only_non_confirmatory_do_not_claim_significance"
    )
    return out


def case_type_summary(stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if stats.empty or "family" not in stats.columns:
        return pd.DataFrame(), pd.DataFrame()
    xgb = stats[stats["reference_model"] == "xgb_cpu"].copy()
    aggregations = {
        "cell_count": ("scenario", "count"),
        "positive_cell_rate": ("mean_signed_advantage", lambda values: float(np.mean(values > 0.0))),
        "mean_signed_advantage": ("mean_signed_advantage", "mean"),
        "mean_relative_improvement": ("mean_relative_improvement", "mean"),
        "mean_win_rate": ("win_rate", "mean"),
    }
    case_summary = (
        xgb.groupby(["family", "case", "task_type", "primary_metric"], as_index=False)
        .agg(**aggregations)
    )
    type_summary = (
        xgb.groupby(["family", "task_type", "primary_metric"], as_index=False)
        .agg(**aggregations)
    )
    return case_summary, type_summary
