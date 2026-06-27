from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "metagb_preflight_adapter" / "outputs" / "preflight_dataset_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "metagb_preflight_adapter" / "outputs"


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


def bounded_positive_log_lift(value: float, neutral: float, high: float) -> float:
    if not math.isfinite(value) or value <= neutral:
        return 0.0
    return clip01(math.log1p(value - neutral) / math.log1p(high - neutral))


def bounded_fraction(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return 0.0
    if high <= low:
        return 0.0
    return clip01((value - low) / (high - low))


def mechanism_components(row: pd.Series) -> dict[str, float]:
    projected_lift = finite_float(row.get("projected_gradient_lift_mean"))
    local_lift = finite_float(row.get("local_linearity_lift"))
    decoupled = finite_float(row.get("decoupled_leaf_fraction"))
    concentration = finite_float(row.get("predictive_score_concentration_mean"))
    local_corr = finite_float(row.get("local_abs_corr_max_mean"))
    global_corr = finite_float(row.get("global_abs_corr_max"))

    # Direct proxy for MetaGB's leaf-score extension. This is intentionally
    # conservative: projected-gradient lift must exceed constant-leaf lift.
    linear_leaf_advantage = bounded_positive_log_lift(projected_lift, neutral=1.0, high=8.0)

    # MetaGB's distinctive split/leaf separation.
    conditional_predictive_decoupling = bounded_fraction(decoupled, low=0.2, high=0.8)

    # Local response heterogeneity: local one-feature association gets stronger
    # after a cheap conditional partition.
    local_heterogeneity = bounded_positive_log_lift(local_lift, neutral=1.0, high=1.8)

    # Feature-selected leaves are better matched when signal is sparse/concentrated.
    predictive_sparse_concentration = bounded_fraction(concentration, low=0.25, high=0.65)

    # Guard against high ratios caused by weak denominators.
    local_signal_support = bounded_fraction(local_corr, low=0.10, high=0.60)

    # A strong global single-feature relationship is not itself a MetaGB-specific
    # reason; it may be handled by ordinary boosted trees. Penalize only mildly.
    global_single_feature_penalty = 0.15 * bounded_fraction(global_corr, low=0.75, high=0.95)

    return {
        "component_linear_leaf_advantage": linear_leaf_advantage,
        "component_cond_pred_decoupling": conditional_predictive_decoupling,
        "component_local_heterogeneity": local_heterogeneity,
        "component_predictive_concentration": predictive_sparse_concentration,
        "component_signal_support": local_signal_support,
        "component_global_single_feature_penalty": global_single_feature_penalty,
    }


def mechanism_score(row: pd.Series, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    components = mechanism_components(row)
    score = (
        args.weight_linear * components["component_linear_leaf_advantage"]
        + args.weight_decoupling * components["component_cond_pred_decoupling"]
        + args.weight_heterogeneity * components["component_local_heterogeneity"]
        + args.weight_concentration * components["component_predictive_concentration"]
        + args.weight_support * components["component_signal_support"]
        - components["component_global_single_feature_penalty"]
    )
    weight_sum = args.weight_linear + args.weight_decoupling + args.weight_heterogeneity + args.weight_concentration + args.weight_support
    score = clip01(score / weight_sum if weight_sum > 0 else 0.0)
    return score, components


def score_band(score: float, args: argparse.Namespace) -> str:
    if score >= args.strong_threshold:
        return "STRONG_MECHANISM_MATCH"
    if score >= args.moderate_threshold:
        return "MODERATE_MECHANISM_MATCH"
    return "WEAK_MECHANISM_MATCH"


def build_scores(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        score, components = mechanism_score(row, args)
        payload = {
            "task_type": row.get("task_type"),
            "dataset": row.get("dataset"),
            "categorical_handling": row.get("categorical_handling"),
            "mechanism_alignment_score": score,
            "mechanism_band": score_band(score, args),
            **components,
            "projected_gradient_lift_mean": row.get("projected_gradient_lift_mean"),
            "local_linearity_lift": row.get("local_linearity_lift"),
            "decoupled_leaf_fraction": row.get("decoupled_leaf_fraction"),
            "predictive_score_concentration_mean": row.get("predictive_score_concentration_mean"),
            "local_abs_corr_max_mean": row.get("local_abs_corr_max_mean"),
        }
        for optional in ["mgb_relative_improvement", "mgb_win", "mgb_worth_it", "runtime_ratio"]:
            if optional in row:
                payload[f"audit_{optional}"] = row.get(optional)
        rows.append(payload)
    return pd.DataFrame(rows).sort_values("mechanism_alignment_score", ascending=False)


def build_report(scores: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# MetaGB Mechanism-Alignment Preflight Report",
        "",
        "## Scoring Contract",
        "",
        "- The score is intrinsic: it is computed from dataset/mechanism proxy features.",
        "- Full MetaGB-vs-XGBoost outcomes are not used to compute the score.",
        "- Outcome columns, when present, are used only for retrospective audit.",
        "- Runtime penalty is intentionally not part of the score.",
        "",
        "## Configuration",
        "",
        f"- input: `{args.input}`",
        f"- rows: {len(scores)}",
        f"- weights: linear={args.weight_linear}, decoupling={args.weight_decoupling}, heterogeneity={args.weight_heterogeneity}, concentration={args.weight_concentration}, support={args.weight_support}",
        f"- strong_threshold: {args.strong_threshold}",
        f"- moderate_threshold: {args.moderate_threshold}",
        "",
        "## Band Counts",
        "",
    ]
    for band, count in scores["mechanism_band"].value_counts().sort_index().items():
        lines.append(f"- {band}: {int(count)}")

    top_cols = [
        "dataset",
        "mechanism_band",
        "mechanism_alignment_score",
        "component_linear_leaf_advantage",
        "component_cond_pred_decoupling",
        "component_local_heterogeneity",
        "component_predictive_concentration",
        "component_signal_support",
    ]
    lines.extend(["", "## Top Mechanism Matches", "", scores[top_cols].head(12).to_markdown(index=False)])

    if "audit_mgb_relative_improvement" in scores.columns:
        audit = scores.copy()
        audit["audit_mgb_relative_improvement"] = pd.to_numeric(audit["audit_mgb_relative_improvement"], errors="coerce")
        lines.extend(["", "## Retrospective Audit Only", ""])
        grouped = audit.groupby("mechanism_band")["audit_mgb_relative_improvement"].agg(["count", "mean", "median"]).reset_index()
        lines.append(grouped.to_markdown(index=False))
        lines.extend(
            [
                "",
                "This audit is not training evidence. It only checks whether the intrinsic mechanism score is aligned with existing completed experiments.",
            ]
        )

    lines.extend(
        [
            "",
            "## Academic Interpretation",
            "",
            "A high score means the dataset appears structurally compatible with MetaGB's feature-selected local linear leaves and conditional/predictive feature decoupling. It does not guarantee a benchmark win; it says the dataset deserves a MetaGB trial on mechanism grounds.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score intrinsic mechanism alignment with MetaGB.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-linear", type=float, default=0.35)
    parser.add_argument("--weight-decoupling", type=float, default=0.25)
    parser.add_argument("--weight-heterogeneity", type=float, default=0.20)
    parser.add_argument("--weight-concentration", type=float, default=0.10)
    parser.add_argument("--weight-support", type=float, default=0.10)
    parser.add_argument("--strong-threshold", type=float, default=0.65)
    parser.add_argument("--moderate-threshold", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    scores = build_scores(df, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "mechanism_alignment_scores.csv", index=False)
    (args.output_dir / "mechanism_alignment_report.md").write_text(build_report(scores, args), encoding="utf-8")
    print(f"wrote {args.output_dir / 'mechanism_alignment_scores.csv'}")
    print(f"wrote {args.output_dir / 'mechanism_alignment_report.md'}")


if __name__ == "__main__":
    main()
