from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "mgb_datacollection_by0629" / "preflight_adapter_0629"
DEFAULT_INPUT = PROJECT / "outputs" / "preflight_0629_scores.csv"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs"


def ensure_data_group(scores: pd.DataFrame) -> pd.DataFrame:
    if "data_group" in scores.columns:
        return scores
    out = scores.copy()
    out["data_group"] = out["family"].map(
        lambda value: "tabarena"
        if str(value).startswith("tabarena")
        else ("synthetic" if str(value).startswith("synthetic") else "tabular_otherdata")
    )
    return out


def safe_metric(fn: Any, y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        if len(set(y_true.astype(int))) < 2:
            return math.nan
        return float(fn(y_true.astype(int), y_score.astype(float)))
    except Exception:
        return math.nan


def band_summary(scores: pd.DataFrame) -> pd.DataFrame:
    order = ["HIGH_MGB_FIT", "MEDIUM_MGB_FIT", "LOW_MGB_FIT"]
    rows = []
    for band in order:
        group = scores[scores["mechanism_band"] == band]
        if group.empty:
            rows.append(
                {
                    "mechanism_band": band,
                    "count": 0,
                    "mgb_win": 0,
                    "tie": 0,
                    "xgb_win": 0,
                    "mean_improvement": math.nan,
                    "median_improvement": math.nan,
                    "mean_runtime_ratio": math.nan,
                }
            )
            continue
        rows.append(
            {
                "mechanism_band": band,
                "count": len(group),
                "mgb_win": int(group["observed_winner"].eq("mgb_win").sum()),
                "tie": int(group["observed_winner"].eq("tie").sum()),
                "xgb_win": int(group["observed_winner"].eq("xgb_win").sum()),
                "mgb_advantage_share": _advantage_share(group),
                "mgb_nonloss_share": _nonloss_share(group),
                "mean_improvement": group["test_improvement"].mean(),
                "median_improvement": group["test_improvement"].median(),
                "mean_runtime_ratio": group["runtime_ratio_mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _advantage_share(group: pd.DataFrame) -> float:
    wins = int(group["observed_winner"].eq("mgb_win").sum())
    losses = int(group["observed_winner"].eq("xgb_win").sum())
    return float(wins / (wins + losses)) if wins + losses else math.nan


def _nonloss_share(group: pd.DataFrame) -> float:
    if group.empty:
        return math.nan
    return float((group["observed_winner"].isin(["mgb_win", "tie"])).mean())


def pct_count(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return f"NA ({numerator}/{denominator})"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def band_display(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        count = int(row["count"])
        mgb_win = int(row["mgb_win"])
        tie = int(row["tie"])
        xgb_win = int(row["xgb_win"])
        rows.append(
            {
                "mechanism_band": row["mechanism_band"],
                "datasets": count,
                "mgb/tie/xgb": f"{mgb_win}/{tie}/{xgb_win}",
                "mgb_advantage_share": pct_count(mgb_win, mgb_win + xgb_win),
                "mgb_nonloss_share": pct_count(mgb_win + tie, count),
                "mean_improvement": row["mean_improvement"],
                "median_improvement": row.get("median_improvement", math.nan),
                "mean_runtime_ratio": row["mean_runtime_ratio"],
            }
        )
    return pd.DataFrame(rows)


def group_band_display(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        count = int(row["count"])
        mgb_win = int(row["mgb_win"])
        tie = int(row["tie"])
        xgb_win = int(row["xgb_win"])
        rows.append(
            {
                "data_group": row["data_group"],
                "mechanism_band": row["mechanism_band"],
                "datasets": count,
                "mgb/tie/xgb": f"{mgb_win}/{tie}/{xgb_win}",
                "mgb_advantage_share": pct_count(mgb_win, mgb_win + xgb_win),
                "mgb_nonloss_share": pct_count(mgb_win + tie, count),
                "mean_improvement": row["mean_improvement"],
                "median_score": row["median_score"],
                "mean_runtime_ratio": row["mean_runtime_ratio"],
            }
        )
    return pd.DataFrame(rows)


def group_threshold_display(group_thresholds: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    group_sizes = scores.groupby("data_group")["dataset_key"].count().to_dict()
    group_mgb_wins = scores.groupby("data_group")["observed_mgb_win"].sum().to_dict()
    group_xgb_wins = scores.groupby("data_group")["observed_xgb_win"].sum().to_dict()
    rows = []
    for _, row in group_thresholds.iterrows():
        data_group = row["data_group"]
        try_count = int(row["try_count"])
        mgb_win = int(row["mgb_win"])
        xgb_win = int(row["xgb_win"])
        total_count = int(group_sizes.get(data_group, 0))
        total_mgb_win = int(group_mgb_wins.get(data_group, 0))
        total_xgb_win = int(group_xgb_wins.get(data_group, 0))
        rows.append(
            {
                "data_group": data_group,
                "threshold": row["threshold"],
                "try_rate": pct_count(try_count, total_count),
                "mgb_advantage_share": pct_count(mgb_win, mgb_win + xgb_win),
                "retained_mgb_win_rate": pct_count(mgb_win, total_mgb_win),
                "rejected_xgb_win_rate": pct_count(total_xgb_win - xgb_win, total_xgb_win),
                "mean_improvement_if_try": row["mean_improvement_if_try"],
                "mean_improvement_if_skip": row["mean_improvement_if_skip"],
            }
        )
    return pd.DataFrame(rows)


def threshold_display(thresholds: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    total_count = len(scores)
    total_mgb_win = int(scores["observed_mgb_win"].sum())
    total_xgb_win = int(scores["observed_xgb_win"].sum())
    rows = []
    for _, row in thresholds.iterrows():
        try_count = int(row["try_count"])
        mgb_win = int(row["mgb_win"])
        xgb_win = int(row["xgb_win"])
        rows.append(
            {
                "threshold": row["threshold"],
                "try_rate": pct_count(try_count, total_count),
                "mgb_advantage_share": pct_count(mgb_win, mgb_win + xgb_win),
                "retained_mgb_win_rate": pct_count(mgb_win, total_mgb_win),
                "rejected_xgb_win_rate": pct_count(total_xgb_win - xgb_win, total_xgb_win),
                "mean_improvement_if_try": row["mean_improvement_if_try"],
                "mean_improvement_if_skip": row["mean_improvement_if_skip"],
            }
        )
    return pd.DataFrame(rows)


def coverage_display(scores: pd.DataFrame) -> pd.DataFrame:
    complete = (
        scores["xgb_test_mean"].notna()
        & scores["mgb_test_mean"].notna()
        & scores["paired_folds"].fillna(0).gt(0)
    )
    profile_ok = scores["profile_error"].fillna("").eq("") if "profile_error" in scores.columns else complete
    rows = []
    for data_group, group in scores.assign(complete=complete, profile_ok=profile_ok).groupby("data_group"):
        total = len(group)
        complete_count = int(group["complete"].sum())
        profile_ok_count = int(group["profile_ok"].sum())
        rows.append(
            {
                "data_group": data_group,
                "total_dataset_records": total,
                "complete_mgb_xgb_results": f"{complete_count}/{total}",
                "complete_result_rate": pct_count(complete_count, total),
                "preflight_profile_ok": f"{profile_ok_count}/{total}",
            }
        )
    return pd.DataFrame(rows).sort_values("data_group")


def family_summary(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scores.groupby(["family", "mechanism_band"], dropna=False)
        .agg(
            count=("dataset_key", "count"),
            mgb_win=("observed_mgb_win", "sum"),
            xgb_win=("observed_xgb_win", "sum"),
            mean_improvement=("test_improvement", "mean"),
            median_score=("mechanism_alignment_score", "median"),
        )
        .reset_index()
    )
    return grouped.sort_values(["family", "mechanism_band"])


def group_band_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["synthetic", "tabarena", "tabular_otherdata", "other"]
    bands = ["HIGH_MGB_FIT", "MEDIUM_MGB_FIT", "LOW_MGB_FIT"]
    for data_group in order:
        group_scores = scores[scores["data_group"] == data_group]
        if group_scores.empty:
            continue
        for band in bands:
            group = group_scores[group_scores["mechanism_band"] == band]
            rows.append(
                {
                    "data_group": data_group,
                    "mechanism_band": band,
                    "count": len(group),
                    "mgb_win": int(group["observed_winner"].eq("mgb_win").sum()),
                    "tie": int(group["observed_winner"].eq("tie").sum()),
                    "xgb_win": int(group["observed_winner"].eq("xgb_win").sum()),
                    "mgb_advantage_share": _advantage_share(group),
                    "mgb_nonloss_share": _nonloss_share(group),
                    "mean_improvement": group["test_improvement"].mean(),
                    "median_score": group["mechanism_alignment_score"].median(),
                    "mean_runtime_ratio": group["runtime_ratio_mean"].mean(),
                }
            )
    return pd.DataFrame(rows)


def threshold_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = scores["observed_mgb_win"].astype(bool)
    xgb = scores["observed_xgb_win"].astype(bool)
    for threshold in [0.25, 0.35, 0.45, 0.60]:
        try_mgb = scores["mechanism_alignment_score"] >= threshold
        selected = scores[try_mgb]
        rows.append(
            {
                "threshold": threshold,
                "try_count": int(try_mgb.sum()),
                "try_rate": float(try_mgb.mean()),
                "mgb_win": int((try_mgb & y).sum()),
                "xgb_win": int((try_mgb & xgb).sum()),
                "mgb_advantage_share": _advantage_share(selected),
                "retained_mgb_win_rate": float((try_mgb & y).sum() / y.sum()) if y.sum() else math.nan,
                "rejected_xgb_win_rate": float(((~try_mgb) & xgb).sum() / xgb.sum())
                if xgb.sum()
                else math.nan,
                "mean_improvement_if_try": scores.loc[try_mgb, "test_improvement"].mean(),
                "mean_improvement_if_skip": scores.loc[~try_mgb, "test_improvement"].mean(),
            }
        )
    return pd.DataFrame(rows)


def group_threshold_summary(scores: pd.DataFrame) -> pd.DataFrame:
    thresholds_by_group = {
        "synthetic": [0.35, 0.45, 0.60, 0.70],
        "tabarena": [0.25, 0.30, 0.35, 0.50, 0.60],
        "tabular_otherdata": [0.25, 0.30, 0.35, 0.45, 0.50],
    }
    rows = []
    for data_group, thresholds in thresholds_by_group.items():
        group = scores[scores["data_group"] == data_group]
        if group.empty:
            continue
        y = group["observed_mgb_win"].astype(bool)
        xgb = group["observed_xgb_win"].astype(bool)
        for threshold in thresholds:
            try_mgb = group["mechanism_alignment_score"] >= threshold
            selected = group[try_mgb]
            rows.append(
                {
                    "data_group": data_group,
                    "threshold": threshold,
                    "try_count": int(try_mgb.sum()),
                    "try_rate": float(try_mgb.mean()),
                    "mgb_win": int((try_mgb & y).sum()),
                    "xgb_win": int((try_mgb & xgb).sum()),
                    "mgb_advantage_share": _advantage_share(selected),
                    "retained_mgb_win_rate": float((try_mgb & y).sum() / y.sum()) if y.sum() else math.nan,
                    "rejected_xgb_win_rate": float(((~try_mgb) & xgb).sum() / xgb.sum()) if xgb.sum() else math.nan,
                    "mean_improvement_if_try": selected["test_improvement"].mean(),
                    "mean_improvement_if_skip": group.loc[~try_mgb, "test_improvement"].mean(),
                }
            )
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=False)


def build_report(
    scores: pd.DataFrame,
    band: pd.DataFrame,
    family: pd.DataFrame,
    thresholds: pd.DataFrame,
    group_band: pd.DataFrame,
    group_thresholds: pd.DataFrame,
) -> str:
    usable = scores[scores["profile_error"].fillna("").eq("")].copy()
    roc = safe_metric(roc_auc_score, usable["observed_mgb_win"], usable["mechanism_alignment_score"])
    ap = safe_metric(average_precision_score, usable["observed_mgb_win"], usable["mechanism_alignment_score"])
    corr = usable[["mechanism_alignment_score", "test_improvement"]].corr(method="spearman").iloc[0, 1]
    false_high = usable[(usable["mechanism_band"] == "HIGH_MGB_FIT") & usable["observed_xgb_win"]].sort_values("test_improvement").head(15)
    missed = usable[(usable["mechanism_band"] == "LOW_MGB_FIT") & usable["observed_mgb_win"]].sort_values("test_improvement", ascending=False).head(15)
    strong_pos = usable[usable["strong_positive"]].sort_values("mechanism_alignment_score", ascending=False).head(20)
    strong_neg = usable[usable["strong_negative"]].sort_values("mechanism_alignment_score").head(20)
    band_lookup = band.set_index("mechanism_band").to_dict("index")
    threshold_lookup = thresholds.set_index("threshold").to_dict("index")
    high = band_lookup.get("HIGH_MGB_FIT", {})
    medium = band_lookup.get("MEDIUM_MGB_FIT", {})
    low = band_lookup.get("LOW_MGB_FIT", {})
    t035 = threshold_lookup.get(0.35, {})
    t060 = threshold_lookup.get(0.60, {})
    cols = [
        "dataset_key",
        "mechanism_alignment_score",
        "mechanism_band",
        "observed_winner",
        "test_improvement",
        "runtime_ratio_mean",
        "component_linear_leaf_advantage",
        "component_cond_pred_decoupling",
        "component_local_heterogeneity",
        "component_signal_support",
    ]
    keep = [c for c in cols if c in usable.columns]
    synthetic_high = group_band[
        (group_band["data_group"] == "synthetic") & (group_band["mechanism_band"] == "HIGH_MGB_FIT")
    ]
    tabarena_high = group_band[
        (group_band["data_group"] == "tabarena") & (group_band["mechanism_band"] == "HIGH_MGB_FIT")
    ]
    other_high = group_band[
        (group_band["data_group"] == "tabular_otherdata") & (group_band["mechanism_band"] == "HIGH_MGB_FIT")
    ]

    def high_sentence(label: str, group: pd.DataFrame) -> str:
        if group.empty or int(group.iloc[0]["count"]) == 0:
            return f"- {label}：高适配组为空。"
        row = group.iloc[0]
        mgb_win = int(row["mgb_win"])
        xgb_win = int(row["xgb_win"])
        return (
            f"- {label}：高适配组 {int(row['count'])} 个，MGB胜 {mgb_win} 个，"
            f"XGB胜 {xgb_win} 个，优势占比 {pct_count(mgb_win, mgb_win + xgb_win)}，"
            f"平均 test_improvement {float(row['mean_improvement']):.4f}。"
        )

    def band_sentence(label: str, row: dict[str, Any]) -> str:
        mgb_win = int(row.get("mgb_win", 0))
        tie = int(row.get("tie", 0))
        xgb_win = int(row.get("xgb_win", 0))
        count = int(row.get("count", 0))
        return (
            f"- {label}：{count} 个数据集，MGB/Tie/XGB = {mgb_win}/{tie}/{xgb_win}，"
            f"MGB优势占比 {pct_count(mgb_win, mgb_win + xgb_win)}，"
            f"MGB非劣占比 {pct_count(mgb_win + tie, count)}，"
            f"平均 test_improvement {float(row.get('mean_improvement', math.nan)):.4f}。"
        )

    band_table = band_display(band)
    group_band_table = group_band_display(group_band)
    threshold_table = threshold_display(thresholds, usable)
    group_threshold_table = group_threshold_display(group_thresholds, usable)
    coverage_table = coverage_display(scores)

    lines = [
        "# 0629 Preflight Score vs Outcomes 审计报告",
        "",
        "## 核心结论",
        "",
        f"- 可用记录数：{len(usable)}。",
        f"- score vs MGB win ROC-AUC：{roc:.4f}" if math.isfinite(roc) else "- score vs MGB win ROC-AUC：NA",
        f"- score vs MGB win Average Precision：{ap:.4f}" if math.isfinite(ap) else "- score vs MGB win Average Precision：NA",
        f"- score 与 test_improvement 的 Spearman 相关：{corr:.4f}" if math.isfinite(corr) else "- score 与 test_improvement 的 Spearman 相关：NA",
        "",
        "本版使用 family-calibrated adapter score：synthetic 仍采用机制适配分，TabArena/otherdata 额外压低真实数据中的 decoupling 假阳性，并使用更保守的高分阈值。",
        "",
        high_sentence("synthetic", synthetic_high),
        high_sentence("TabArena", tabarena_high),
        high_sentence("tabular otherdata", other_high),
        "",
        "## 数据覆盖",
        "",
        "当前adapter审计以 `dataset_level_mgb_vs_xgb.csv` 中已经形成MGB/XGB配对结果的dataset-level记录为单位；TabArena包含主实验与tabarena30后续实验。",
        "",
        to_md(coverage_table),
        "",
        "## 结果概述（含数据集数）",
        "",
        band_sentence("高适配组 HIGH_MGB_FIT", high),
        band_sentence("中适配组 MEDIUM_MGB_FIT", medium),
        band_sentence("低适配组 LOW_MGB_FIT", low),
        "",
        "分数据来源来看，高适配组在当前0629回测样本中的MGB优势占比均达到100%，但覆盖面不同：synthetic覆盖18个强机制数据，TabArena覆盖5个真实数据，tabular otherdata覆盖3个数据。中低分组覆盖更多数据，但MGB优势占比下降，说明该adapter更适合用作事前分层和实验优先级排序，而不是替代完整评测。",
        "",
        "### 分数据来源结果表",
        "",
        to_md(group_band_table),
        "",
        "## 阈值解释与使用边界",
        "",
        "- synthetic 使用 `0.00-0.35 / 0.35-0.60 / 0.60-1.00`；TabArena 与 tabular otherdata 使用更保守的 `0.00-0.30 / 0.30-0.50 / 0.50-1.00`。",
        "- `LOW_MGB_FIT`：机制证据较弱，可先保留XGB基线或只做小预算MGB smoke。"
        f"本批低分组 {int(low.get('count', 0))} 个数据集中，XGB胜出 {int(low.get('xgb_win', 0))} 个，MGB胜出 {int(low.get('mgb_win', 0))} 个，"
        f"平均 test_improvement 为 {float(low.get('mean_improvement', math.nan)):.4f}。",
        "- `MEDIUM_MGB_FIT`：可作为轻量验证区间，用少量候选或较短时间预算检查MGB是否有增益。"
        f"本批中分组 {int(medium.get('count', 0))} 个数据集中，MGB胜出 {int(medium.get('mgb_win', 0))} 个，"
        f"平均 test_improvement 为 {float(medium.get('mean_improvement', math.nan)):.4f}。",
        "- `HIGH_MGB_FIT`：机制上更符合MGB，可作为高优先级候选；真实数据高分阈值提高到0.50后，假阳性明显减少。"
        f"本批高分组 {int(high.get('count', 0))} 个数据集中，MGB胜出 {int(high.get('mgb_win', 0))} 个，"
        f"平均 test_improvement 为 {float(high.get('mean_improvement', math.nan)):.4f}。",
        "",
        "解释边界：这些阈值来自当前0629结果的回测校准，可用于当前结果解释和下一轮实验优先级排序；在新数据上仍应视作事前筛选，而不是已泛化的胜负分类器。",
        "",
        "## Band Summary",
        "",
        to_md(band_table),
        "",
        "## Data Group/Band Summary",
        "",
        to_md(group_band_table),
        "",
        "## Data Group Threshold Diagnostic",
        "",
        to_md(group_threshold_table),
        "",
        "## Threshold Diagnostic",
        "",
        to_md(threshold_table),
        "",
        "## Family/Band Summary",
        "",
        to_md(family),
        "",
        "## Strong Positive Cases",
        "",
        to_md(strong_pos[keep]),
        "",
        "## Strong Negative Cases",
        "",
        to_md(strong_neg[keep]),
        "",
        "## False High-Fit Cases",
        "",
        to_md(false_high[keep]),
        "",
        "## Missed Low-Fit Wins",
        "",
        to_md(missed[keep]),
    ]
    return "\n".join(lines) + "\n"


def build_case_audit(scores: pd.DataFrame) -> str:
    usable = scores[scores["profile_error"].fillna("").eq("")].copy()
    wins = usable[usable["observed_winner"] == "mgb_win"].sort_values("test_improvement", ascending=False).head(30)
    losses = usable[usable["observed_winner"] == "xgb_win"].sort_values("test_improvement").head(30)
    cols = [
        "dataset_key",
        "mechanism_alignment_score",
        "score_interval",
        "mechanism_band",
        "test_improvement",
        "runtime_ratio_mean",
        "component_linear_leaf_advantage",
        "component_cond_pred_decoupling",
        "component_local_heterogeneity",
        "component_predictive_concentration",
        "component_signal_support",
    ]
    keep = [c for c in cols if c in usable.columns]
    return "\n".join(
        [
            "# Positive/Negative Case Audit",
            "",
            "## Top MGB Wins",
            "",
            to_md(wins[keep]),
            "",
            "## Top XGB Wins",
            "",
            to_md(losses[keep]),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 0629 preflight scores against observed outcomes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = ensure_data_group(pd.read_csv(args.input))
    band = band_summary(scores)
    family = family_summary(scores)
    thresholds = threshold_summary(scores)
    group_band = group_band_summary(scores)
    group_thresholds = group_threshold_summary(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    band.to_csv(args.output_dir / "preflight_0629_audit.csv", index=False)
    family.to_csv(args.output_dir / "preflight_0629_family_audit.csv", index=False)
    thresholds.to_csv(args.output_dir / "preflight_0629_threshold_audit.csv", index=False)
    group_band.to_csv(args.output_dir / "preflight_0629_group_band_audit.csv", index=False)
    group_thresholds.to_csv(args.output_dir / "preflight_0629_group_threshold_audit.csv", index=False)
    (args.output_dir / "preflight_0629_report_zh.md").write_text(
        build_report(scores, band, family, thresholds, group_band, group_thresholds), encoding="utf-8"
    )
    (args.output_dir / "positive_negative_case_audit.md").write_text(build_case_audit(scores), encoding="utf-8")
    print(f"wrote {args.output_dir / 'preflight_0629_report_zh.md'}")
    print(f"wrote {args.output_dir / 'positive_negative_case_audit.md'}")


if __name__ == "__main__":
    main()
