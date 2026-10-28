from __future__ import annotations

import pandas as pd
import pytest

from metagb_ablation.statistics import (
    main_ablation_tables,
    pred_sensitivity_tables,
    scoreagg_tables,
    unit_level_results,
)


def _fold_rows() -> pd.DataFrame:
    rows = []
    variant_losses = {
        "xgb": 2.0,
        "bias_p00_sa0": 1.8,
        "split_p10_sa0": 1.5,
        "full_p10_sa0": 1.0,
        "full_p10_sa1": 1.1,
        "full_p10_sa2": 1.2,
    }
    for suffix in (2, 4, 6, 8, 10):
        variant_losses.setdefault(
            f"split_p{suffix:02d}_sa0",
            1.8 - 0.03 * suffix,
        )
        variant_losses.setdefault(
            f"full_p{suffix:02d}_sa0",
            1.8 - 0.08 * suffix,
        )
    for seed in (101, 202, 303):
        for outer_fold in (1, 2):
            for variant_id, loss in variant_losses.items():
                family = (
                    "baseline"
                    if variant_id == "xgb"
                    else "bias"
                    if variant_id.startswith("bias")
                    else "split_only"
                    if variant_id.startswith("split")
                    else "full"
                )
                pred_fraction = (
                    None
                    if variant_id == "xgb"
                    else 0.0
                    if variant_id.startswith("bias")
                    else int(variant_id.split("_p", 1)[1][:2]) / 10.0
                )
                score_agg = (
                    None
                    if variant_id == "xgb"
                    else int(variant_id.rsplit("sa", 1)[1])
                )
                rows.append(
                    {
                        "run_id": "r",
                        "config_hash": "c",
                        "execution_hash": "e",
                        "variant_hash": "v",
                        "domain": "synthetic",
                        "dataset": "",
                        "scenario": "regime_slope",
                        "dimension": 10,
                        "task_type": "regression",
                        "primary_metric": "rmse",
                        "unit_id": f"seed{seed}",
                        "outer_fold": outer_fold,
                        "variant_id": variant_id,
                        "family": family,
                        "leaf_mode": family,
                        "pred_fraction": pred_fraction,
                        "score_agg": score_agg,
                        "score_agg_label": {
                            None: "not_applicable",
                            0: "greedy",
                            1: "mean",
                            2: "median",
                        }[score_agg],
                        "test_primary_metric": loss + seed * 1e-6,
                        "fit_sec": 1.0,
                        "predict_sec": 0.1,
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def _predictions(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fold in folds.iterrows():
        for offset in range(2):
            rows.append(
                {
                    "task_id": f"{fold['unit_id']}__out{fold['outer_fold']}",
                    "unit_id": fold["unit_id"],
                    "variant_id": fold["variant_id"],
                    "sample_id": (int(fold["outer_fold"]) - 1) * 2 + offset,
                    "y_true": 0.0,
                    "y_pred": float(fold["test_primary_metric"]),
                    "probabilities_json": "",
                }
            )
    return pd.DataFrame(rows)


def test_main_effect_decomposition_uses_paired_units() -> None:
    folds = _fold_rows()
    units = unit_level_results(folds, _predictions(folds))
    raw, effects = main_ablation_tables(units, seed=3)
    assert set(raw["variant_id"]) == {
        "xgb",
        "bias_p00_sa0",
        "split_p10_sa0",
        "full_p10_sa0",
    }
    lookup = effects.set_index("effect")
    assert lookup.loc["split_vs_bias", "mean_signed_improvement"] > 0.0
    assert lookup.loc["leaf_vs_split", "mean_signed_improvement"] > 0.0
    assert lookup.loc["total_vs_bias", "mean_signed_improvement"] > 0.0
    assert (effects["n_paired_units"] == 3).all()


def test_pred_sensitivity_has_two_curves_and_three_effects_per_p() -> None:
    folds = _fold_rows()
    units = unit_level_results(folds, _predictions(folds))
    raw, effects = pred_sensitivity_tables(units, seed=5)
    assert set(raw["family"]) == {"split_only", "full"}
    assert raw["pred_fraction"].nunique() == 5
    assert set(effects["effect"]) == {"split", "leaf", "total"}
    assert effects["pred_fraction"].nunique() == 5


def test_scoreagg_table_reuses_full_p1_greedy() -> None:
    folds = _fold_rows()
    units = unit_level_results(folds, _predictions(folds))
    raw, effects = scoreagg_tables(units, seed=7)
    assert set(raw["variant_id"]) == {
        "full_p10_sa0",
        "full_p10_sa1",
        "full_p10_sa2",
    }
    assert set(effects["effect"]) == {
        "mean_vs_greedy",
        "median_vs_greedy",
    }
    assert (effects["mean_signed_improvement"] < 0.0).all()


def test_extended_scoreagg_reports_full_two_by_three_design() -> None:
    folds = _fold_rows()
    predictions = _predictions(folds)
    units = unit_level_results(folds, predictions)
    split_greedy = units[units["variant_id"] == "split_p10_sa0"]
    additions = []
    for score_agg, suffix in ((1, "mean"), (2, "median")):
        frame = split_greedy.copy()
        frame["variant_id"] = f"split_p10_sa{score_agg}"
        frame["score_agg"] = score_agg
        frame["score_agg_label"] = suffix
        frame["unit_primary_metric"] += 0.1 * score_agg
        additions.append(frame)
    extended = pd.concat([units, *additions], ignore_index=True)
    raw, effects = scoreagg_tables(extended, seed=9)
    assert raw["variant_id"].nunique() == 6
    assert {
        "mean_vs_greedy",
        "median_vs_greedy",
        "split_mean_vs_greedy",
        "split_median_vs_greedy",
    } == set(effects["effect"])


def test_unit_metric_is_recomputed_from_oof_predictions_not_fold_mean() -> None:
    folds = pd.DataFrame(
        [
            {
                "run_id": "r",
                "config_hash": "c",
                "execution_hash": "e",
                "variant_hash": "v",
                "domain": "real",
                "dataset": "d",
                "scenario": "",
                "dimension": 0,
                "task_type": "regression",
                "primary_metric": "rmse",
                "unit_id": "repeat1",
                "outer_fold": 1,
                "variant_id": "xgb",
                "family": "baseline",
                "leaf_mode": "constant",
                "pred_fraction": None,
                "score_agg": None,
                "score_agg_label": "not_applicable",
                "test_primary_metric": 1.0,
                "fit_sec": 1.0,
                "predict_sec": 0.1,
                "status": "ok",
            },
            {
                "run_id": "r",
                "config_hash": "c",
                "execution_hash": "e",
                "variant_hash": "v",
                "domain": "real",
                "dataset": "d",
                "scenario": "",
                "dimension": 0,
                "task_type": "regression",
                "primary_metric": "rmse",
                "unit_id": "repeat1",
                "outer_fold": 2,
                "variant_id": "xgb",
                "family": "baseline",
                "leaf_mode": "constant",
                "pred_fraction": None,
                "score_agg": None,
                "score_agg_label": "not_applicable",
                "test_primary_metric": 3.0,
                "fit_sec": 1.0,
                "predict_sec": 0.1,
                "status": "ok",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "unit_id": "repeat1",
                "variant_id": "xgb",
                "sample_id": 0,
                "y_true": 0.0,
                "y_pred": 1.0,
                "probabilities_json": "",
            },
            {
                "unit_id": "repeat1",
                "variant_id": "xgb",
                "sample_id": 1,
                "y_true": 0.0,
                "y_pred": 3.0,
                "probabilities_json": "",
            },
        ]
    )
    units = unit_level_results(folds, predictions)
    assert len(units) == 1
    assert units.iloc[0]["unit_primary_metric"] == pytest.approx(
        (5.0) ** 0.5
    )
    assert units.iloc[0]["unit_primary_metric"] != 2.0


def test_unit_aggregation_rejects_mixed_execution_identity() -> None:
    folds = _fold_rows()
    predictions = _predictions(folds)
    target = (
        (folds["unit_id"] == "seed101")
        & (folds["variant_id"] == "xgb")
        & (folds["outer_fold"] == 2)
    )
    folds.loc[target, "execution_hash"] = "CORRUPT"
    with pytest.raises(ValueError, match="Inconsistent metadata"):
        unit_level_results(folds, predictions)
