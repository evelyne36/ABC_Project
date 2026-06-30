from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "mgb_datacollection_by0629" / "preflight_adapter_0629"
DEFAULT_INPUT = PROJECT / "outputs" / "preflight_0629_dataset_table.csv"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs"


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def bounded_fraction(value: float, low: float, high: float) -> float:
    if not math.isfinite(value) or high <= low:
        return 0.0
    return clip01((value - low) / (high - low))


def bounded_log_lift(value: float, neutral: float, high: float) -> float:
    if not math.isfinite(value) or value <= neutral:
        return 0.0
    return clip01(math.log1p(value - neutral) / math.log1p(high - neutral))


def data_group(family: Any) -> str:
    text = str(family or "")
    if text.startswith("tabarena"):
        return "tabarena"
    if text.startswith("synthetic"):
        return "synthetic"
    if text.startswith("tabular_otherdata"):
        return "tabular_otherdata"
    return "other"


def bounded_log_value(value: float, low: float, high: float) -> float:
    if not math.isfinite(value) or value < 0:
        return 0.0
    return bounded_fraction(math.log1p(value), math.log1p(low), math.log1p(high))


def mechanism_components(row: pd.Series) -> dict[str, float]:
    projected_lift = finite_float(row.get("projected_gradient_lift_mean"))
    local_lift = finite_float(row.get("local_linearity_lift"))
    decoupled = finite_float(row.get("decoupled_leaf_fraction"))
    concentration = finite_float(row.get("predictive_score_concentration_mean"))
    local_corr = finite_float(row.get("local_abs_corr_max_mean"))
    global_corr = finite_float(row.get("global_abs_corr_max"))
    pressure = finite_float(row.get("runtime_pressure_log"), 0.0)
    raw_rows = finite_float(row.get("raw_rows"), 0.0)
    processed_features = finite_float(row.get("processed_features"), 0.0)
    class_imbalance = finite_float(row.get("class_imbalance"), 0.0)

    linear_leaf_advantage = bounded_log_lift(projected_lift, neutral=1.0, high=8.0)
    conditional_predictive_decoupling = bounded_fraction(decoupled, low=0.2, high=0.8)
    local_heterogeneity = bounded_log_lift(local_lift, neutral=1.0, high=1.8)
    predictive_concentration = bounded_fraction(concentration, low=0.25, high=0.65)
    signal_support = bounded_fraction(local_corr, low=0.10, high=0.60)
    global_corr_support = bounded_fraction(global_corr, low=0.30, high=0.80)
    feature_support = bounded_log_value(processed_features, low=6.0, high=30.0)
    small_data_support = 1.0 - bounded_log_value(raw_rows, low=5000.0, high=50000.0)
    medium_data_support = bounded_log_value(raw_rows, low=500.0, high=5000.0) * (
        1.0 - bounded_log_value(raw_rows, low=5000.0, high=50000.0)
    )
    pressure_support = bounded_fraction(pressure, low=10.0, high=15.0)
    low_concentration_support = 1.0 - predictive_concentration
    imbalance_penalty = bounded_fraction(class_imbalance, low=0.75, high=0.95)
    global_single_feature_penalty = 0.15 * bounded_fraction(global_corr, low=0.75, high=0.95)
    runtime_pressure_penalty = 0.08 * bounded_fraction(pressure, low=15.0, high=22.0)

    return {
        "component_linear_leaf_advantage": linear_leaf_advantage,
        "component_cond_pred_decoupling": conditional_predictive_decoupling,
        "component_local_heterogeneity": local_heterogeneity,
        "component_predictive_concentration": predictive_concentration,
        "component_signal_support": signal_support,
        "component_global_corr_support": global_corr_support,
        "component_feature_support": feature_support,
        "component_small_data_support": small_data_support,
        "component_medium_data_support": medium_data_support,
        "component_pressure_support": pressure_support,
        "component_low_concentration_support": low_concentration_support,
        "component_imbalance_penalty": imbalance_penalty,
        "component_global_single_feature_penalty": global_single_feature_penalty,
        "component_runtime_pressure_penalty": runtime_pressure_penalty,
    }


def intrinsic_mechanism_score(row: pd.Series) -> tuple[float, dict[str, float]]:
    components = mechanism_components(row)
    synergy = components["component_linear_leaf_advantage"] * max(
        components["component_cond_pred_decoupling"],
        components["component_local_heterogeneity"],
    )
    raw = (
        0.45 * synergy
        + 0.20 * components["component_local_heterogeneity"]
        + 0.15 * components["component_cond_pred_decoupling"]
        + 0.10 * components["component_signal_support"]
        + 0.10 * components["component_predictive_concentration"]
        - components["component_global_single_feature_penalty"]
    )
    mechanism = clip01(raw)
    practical = clip01(mechanism - components["component_runtime_pressure_penalty"])
    return practical, {**components, "component_linear_decoupling_synergy": synergy, "intrinsic_mechanism_score": mechanism}


def calibrated_adapter_score(row: pd.Series) -> tuple[float, dict[str, float]]:
    intrinsic, components = intrinsic_mechanism_score(row)
    group = data_group(row.get("family"))
    if group == "synthetic":
        calibrated = intrinsic
    elif group == "tabarena":
        calibrated = (
            0.30 * components["component_linear_leaf_advantage"]
            + 0.25 * components["component_global_corr_support"]
            + 0.20 * components["component_signal_support"]
            + 0.15 * components["component_feature_support"]
            + 0.10 * components["component_small_data_support"]
            - 0.20 * components["component_cond_pred_decoupling"]
            - 0.20 * components["component_local_heterogeneity"]
            - 0.15 * components["component_imbalance_penalty"]
        )
    elif group == "tabular_otherdata":
        calibrated = (
            0.25 * components["component_linear_leaf_advantage"]
            + 0.30 * components["component_feature_support"]
            + 0.25 * components["component_pressure_support"]
            + 0.10 * components["component_medium_data_support"]
            + 0.10 * components["component_low_concentration_support"]
            - 0.20 * components["component_cond_pred_decoupling"]
        )
    else:
        calibrated = intrinsic
    return clip01(calibrated), {**components, "mechanism_alignment_score_raw": intrinsic}


def score_thresholds(group: str) -> tuple[float, float]:
    if group in {"tabarena", "tabular_otherdata"}:
        return 0.30, 0.50
    return 0.35, 0.60


def score_band(score: float, group: str = "synthetic") -> str:
    medium_threshold, high_threshold = score_thresholds(group)
    if score >= high_threshold:
        return "HIGH_MGB_FIT"
    if score >= medium_threshold:
        return "MEDIUM_MGB_FIT"
    return "LOW_MGB_FIT"


def score_interval(score: float, group: str = "synthetic") -> str:
    medium_threshold, high_threshold = score_thresholds(group)
    if score >= high_threshold:
        return f"{high_threshold:.2f}-1.00"
    if score >= medium_threshold:
        return f"{medium_threshold:.2f}-{high_threshold:.2f}"
    return f"0.00-{medium_threshold:.2f}"


def score_interpretation(score: float, group: str = "synthetic") -> str:
    band = score_band(score, group)
    if band == "HIGH_MGB_FIT":
        return "MGB适配度较高，值得优先尝试或保留MGB结果"
    if band == "MEDIUM_MGB_FIT":
        return "MGB适配度中等，可用轻量MGB或TabArena30式预算验证"
    return "MGB适配度一般，若资源紧张可优先使用XGB"


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        group = data_group(row.get("family"))
        score, components = calibrated_adapter_score(row)
        payload = row.to_dict()
        payload.update(components)
        payload["data_group"] = group
        payload["mechanism_alignment_score"] = score
        payload["score_interval"] = score_interval(score, group)
        payload["mechanism_band"] = score_band(score, group)
        payload["score_interpretation"] = score_interpretation(score, group)
        rows.append(payload)
    out = pd.DataFrame(rows)
    return out.sort_values("mechanism_alignment_score", ascending=False).reset_index(drop=True)


def build_report(scores: pd.DataFrame) -> str:
    band_counts = scores["mechanism_band"].value_counts().reindex(
        ["HIGH_MGB_FIT", "MEDIUM_MGB_FIT", "LOW_MGB_FIT"], fill_value=0
    )
    lines = [
        "# 0629 MetaGB 机制适配度评分",
        "",
        "## 分数区间",
        "",
        "| 分数区间 | 解释 |",
        "|---|---|",
        "| synthetic: 0.00-0.35; TabArena/otherdata: 0.00-0.30 | MGB适配度一般，不急于跑完整MGB；资源紧张时优先XGB |",
        "| synthetic: 0.35-0.60; TabArena/otherdata: 0.30-0.50 | MGB适配度中等，可用轻量MGB/TabArena30式预算验证 |",
        "| synthetic: 0.60-1.00; TabArena/otherdata: 0.50-1.00 | MGB适配度较高，值得优先尝试MGB |",
        "",
        "## Band Counts",
        "",
    ]
    for band, count in band_counts.items():
        lines.append(f"- {band}: {int(count)}")
    top_cols = [
        "dataset_key",
        "mechanism_alignment_score",
        "score_interval",
        "mechanism_band",
        "task_type",
        "encoding",
        "data_group",
        "observed_winner",
        "test_improvement",
        "mechanism_alignment_score_raw",
        "component_linear_leaf_advantage",
        "component_cond_pred_decoupling",
        "component_local_heterogeneity",
        "component_predictive_concentration",
        "component_signal_support",
    ]
    keep = [c for c in top_cols if c in scores.columns]
    lines.extend(["", "## Top 20 MGB Fit", "", scores[keep].head(20).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "该分数只使用数据本身和浅层预分析特征，不使用最终MGB/XGB胜负标签。"
            "本版规则和阈值经过0629已有结果的回测校准，可用于当前结果解释和下一轮实验优先级排序；"
            "在新数据上仍应视作事前筛选分，而不是已经外部验证的胜负分类器。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score 0629 MetaGB mechanism alignment.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    scores = build_scores(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "preflight_0629_scores.csv", index=False)
    (args.output_dir / "score_report_zh.md").write_text(build_report(scores), encoding="utf-8")
    print(f"wrote {args.output_dir / 'preflight_0629_scores.csv'}")
    print(f"wrote {args.output_dir / 'score_report_zh.md'}")


if __name__ == "__main__":
    main()
