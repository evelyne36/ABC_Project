from __future__ import annotations

import json
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

from .protocols import MAIN_VARIANT_IDS


SETTING_COLUMNS = [
    "run_id",
    "config_hash",
    "execution_hash",
    "variant_hash",
    "domain",
    "dataset",
    "scenario",
    "dimension",
    "task_type",
    "primary_metric",
]

UNIT_RESULT_COLUMNS = [
    *SETTING_COLUMNS,
    "unit_id",
    "variant_id",
    "family",
    "leaf_mode",
    "pred_fraction",
    "score_agg",
    "score_agg_label",
    "unit_primary_metric",
    "outer_fold_std",
    "mean_fit_sec",
    "mean_predict_sec",
    "outer_fold_count",
    "oof_sample_count",
]

RAW_SUMMARY_COLUMNS = [
    *SETTING_COLUMNS,
    "variant_id",
    "family",
    "leaf_mode",
    "pred_fraction",
    "score_agg",
    "score_agg_label",
    "n_units",
    "metric_mean",
    "metric_std",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "mean_fit_sec",
    "mean_predict_sec",
]

EFFECT_COLUMNS = [
    *SETTING_COLUMNS,
    "effect",
    "target_variant",
    "baseline_variant",
    "n_paired_units",
    "target_mean",
    "baseline_mean",
    "mean_signed_improvement",
    "signed_ci_low",
    "signed_ci_high",
    "mean_relative_improvement",
    "relative_ci_low",
    "relative_ci_high",
    "win_rate",
]


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> Tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        data,
        size=(int(resamples), data.size),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _classification_logloss(frame: pd.DataFrame) -> float:
    probabilities = []
    for value in frame["probabilities_json"]:
        if not isinstance(value, str) or not value:
            return float("nan")
        probabilities.append(json.loads(value))
    proba = np.asarray(probabilities, dtype=float)
    y_true = frame["y_true"].to_numpy(dtype=int)
    if proba.ndim != 2 or proba.shape[0] != len(y_true):
        return float("nan")
    return float(log_loss(y_true, proba, labels=list(range(proba.shape[1]))))


def unit_level_results(
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if fold_results.empty or predictions.empty:
        return _empty_frame(UNIT_RESULT_COLUMNS)
    successful = fold_results[fold_results["status"] == "ok"].copy()
    if successful.empty:
        return _empty_frame(UNIT_RESULT_COLUMNS)
    metadata_columns = [
        *SETTING_COLUMNS,
        "unit_id",
        "variant_id",
        "family",
        "leaf_mode",
        "pred_fraction",
        "score_agg",
        "score_agg_label",
    ]
    identity_counts = successful.groupby(
        ["unit_id", "variant_id"],
        dropna=False,
    )[metadata_columns].nunique(dropna=False)
    inconsistent = identity_counts.gt(1).any(axis=1)
    if inconsistent.any():
        bad_groups = [
            f"{unit_id}::{variant_id}"
            for unit_id, variant_id in identity_counts.index[inconsistent]
        ]
        raise ValueError(
            "Inconsistent metadata within unit/variant groups: "
            + ", ".join(bad_groups)
        )
    metadata = successful[metadata_columns].drop_duplicates(
        subset=["unit_id", "variant_id"]
    )
    timing = (
        successful.groupby(["unit_id", "variant_id"], as_index=False)
        .agg(
            outer_fold_std=("test_primary_metric", "std"),
            mean_fit_sec=("fit_sec", "mean"),
            mean_predict_sec=("predict_sec", "mean"),
            outer_fold_count=("outer_fold", "nunique"),
        )
    )
    rows: List[Dict[str, object]] = []
    grouped_predictions = predictions.groupby(
        ["unit_id", "variant_id"],
        dropna=False,
    )
    for (unit_id, variant_id), frame in grouped_predictions:
        meta = metadata[
            (metadata["unit_id"] == unit_id)
            & (metadata["variant_id"] == variant_id)
        ]
        if meta.empty:
            continue
        if frame["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate OOF sample for unit={unit_id}, variant={variant_id}"
            )
        task_type = str(meta.iloc[0]["task_type"])
        if task_type == "regression":
            metric = float(
                np.sqrt(
                    mean_squared_error(
                        frame["y_true"].to_numpy(dtype=float),
                        frame["y_pred"].to_numpy(dtype=float),
                    )
                )
            )
        else:
            metric = _classification_logloss(frame)
        time_row = timing[
            (timing["unit_id"] == unit_id)
            & (timing["variant_id"] == variant_id)
        ].iloc[0]
        row = meta.iloc[0].to_dict()
        row.update(
            {
                "unit_primary_metric": metric,
                "outer_fold_std": float(time_row["outer_fold_std"]),
                "mean_fit_sec": float(time_row["mean_fit_sec"]),
                "mean_predict_sec": float(time_row["mean_predict_sec"]),
                "outer_fold_count": int(time_row["outer_fold_count"]),
                "oof_sample_count": int(len(frame)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=UNIT_RESULT_COLUMNS)


def _raw_summary(
    unit_results: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    if unit_results.empty:
        return _empty_frame(RAW_SUMMARY_COLUMNS)
    rows: List[Dict[str, object]] = []
    group_columns = [
        *SETTING_COLUMNS,
        "variant_id",
        "family",
        "leaf_mode",
        "pred_fraction",
        "score_agg",
        "score_agg_label",
    ]
    for group_key, frame in unit_results.groupby(group_columns, dropna=False):
        values = frame["unit_primary_metric"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, seed=seed + len(rows))
        row = dict(zip(group_columns, group_key))
        row.update(
            {
                "n_units": int(len(values)),
                "metric_mean": float(np.mean(values)),
                "metric_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "mean_fit_sec": float(frame["mean_fit_sec"].mean()),
                "mean_predict_sec": float(frame["mean_predict_sec"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=RAW_SUMMARY_COLUMNS)


def paired_effects(
    unit_results: pd.DataFrame,
    comparisons: Mapping[str, Tuple[str, str]],
    *,
    seed: int,
) -> pd.DataFrame:
    if unit_results.empty:
        return _empty_frame(EFFECT_COLUMNS)
    rows: List[Dict[str, object]] = []
    for setting_key, setting in unit_results.groupby(SETTING_COLUMNS, dropna=False):
        pivot = setting.pivot_table(
            index="unit_id",
            columns="variant_id",
            values="unit_primary_metric",
            aggfunc="first",
        )
        for effect_name, (target_id, baseline_id) in comparisons.items():
            if target_id not in pivot or baseline_id not in pivot:
                continue
            paired = pivot[[target_id, baseline_id]].dropna()
            if paired.empty:
                continue
            target = paired[target_id].to_numpy(dtype=float)
            baseline = paired[baseline_id].to_numpy(dtype=float)
            signed = baseline - target
            relative = signed / np.maximum(np.abs(baseline), 1e-12)
            signed_low, signed_high = bootstrap_mean_ci(
                signed,
                seed=seed + len(rows) * 2,
            )
            relative_low, relative_high = bootstrap_mean_ci(
                relative,
                seed=seed + len(rows) * 2 + 1,
            )
            row = dict(zip(SETTING_COLUMNS, setting_key))
            row.update(
                {
                    "effect": effect_name,
                    "target_variant": target_id,
                    "baseline_variant": baseline_id,
                    "n_paired_units": int(len(paired)),
                    "target_mean": float(np.mean(target)),
                    "baseline_mean": float(np.mean(baseline)),
                    "mean_signed_improvement": float(np.mean(signed)),
                    "signed_ci_low": signed_low,
                    "signed_ci_high": signed_high,
                    "mean_relative_improvement": float(np.mean(relative)),
                    "relative_ci_low": relative_low,
                    "relative_ci_high": relative_high,
                    "win_rate": float(np.mean(signed > 0.0)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=EFFECT_COLUMNS)


def main_ablation_tables(
    unit_results: pd.DataFrame,
    *,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if unit_results.empty:
        return (
            _empty_frame(RAW_SUMMARY_COLUMNS),
            _empty_frame(EFFECT_COLUMNS),
        )
    main = unit_results[unit_results["variant_id"].isin(MAIN_VARIANT_IDS)].copy()
    raw = _raw_summary(main, seed=seed)
    effects = paired_effects(
        main,
        {
            "split_vs_bias": ("split_p10_sa0", "bias_p00_sa0"),
            "leaf_vs_split": ("full_p10_sa0", "split_p10_sa0"),
            "total_vs_bias": ("full_p10_sa0", "bias_p00_sa0"),
            "full_vs_xgb": ("full_p10_sa0", "xgb"),
        },
        seed=seed + 10_000,
    )
    return raw, effects


def pred_sensitivity_tables(
    unit_results: pd.DataFrame,
    *,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if unit_results.empty:
        return (
            _empty_frame(RAW_SUMMARY_COLUMNS),
            _empty_frame([*EFFECT_COLUMNS, "pred_fraction"]),
        )
    sensitivity = unit_results[
        (unit_results["score_agg"] == 0)
        & (unit_results["family"].isin(["split_only", "full"]))
    ].copy()
    raw = _raw_summary(sensitivity, seed=seed)
    effect_rows: List[pd.DataFrame] = []
    for pred_fraction in sorted(sensitivity["pred_fraction"].dropna().unique()):
        suffix = int(round(float(pred_fraction) * 10))
        subset_ids = {
            "bias_p00_sa0",
            f"split_p{suffix:02d}_sa0",
            f"full_p{suffix:02d}_sa0",
        }
        subset = unit_results[unit_results["variant_id"].isin(subset_ids)]
        effects = paired_effects(
            subset,
            {
                "split": (f"split_p{suffix:02d}_sa0", "bias_p00_sa0"),
                "leaf": (
                    f"full_p{suffix:02d}_sa0",
                    f"split_p{suffix:02d}_sa0",
                ),
                "total": (f"full_p{suffix:02d}_sa0", "bias_p00_sa0"),
            },
            seed=seed + suffix * 1_000,
        )
        if not effects.empty:
            effects["pred_fraction"] = float(pred_fraction)
            effect_rows.append(effects)
    combined = (
        pd.concat(effect_rows, ignore_index=True)
        if effect_rows
        else _empty_frame([*EFFECT_COLUMNS, "pred_fraction"])
    )
    return raw, combined


def scoreagg_tables(
    unit_results: pd.DataFrame,
    *,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if unit_results.empty:
        return (
            _empty_frame(RAW_SUMMARY_COLUMNS),
            _empty_frame(EFFECT_COLUMNS),
        )
    identifiers = {"full_p10_sa0", "full_p10_sa1", "full_p10_sa2"}
    if {"split_p10_sa1", "split_p10_sa2"}.issubset(
        set(unit_results["variant_id"])
    ):
        identifiers.update(
            {"split_p10_sa0", "split_p10_sa1", "split_p10_sa2"}
        )
    scoreagg = unit_results[unit_results["variant_id"].isin(identifiers)].copy()
    raw = _raw_summary(scoreagg, seed=seed)
    comparisons = {
        "mean_vs_greedy": ("full_p10_sa1", "full_p10_sa0"),
        "median_vs_greedy": ("full_p10_sa2", "full_p10_sa0"),
    }
    if "split_p10_sa1" in identifiers:
        comparisons.update(
            {
                "split_mean_vs_greedy": (
                    "split_p10_sa1",
                    "split_p10_sa0",
                ),
                "split_median_vs_greedy": (
                    "split_p10_sa2",
                    "split_p10_sa0",
                ),
            }
        )
    effects = paired_effects(
        scoreagg,
        comparisons,
        seed=seed + 20_000,
    )
    return raw, effects
