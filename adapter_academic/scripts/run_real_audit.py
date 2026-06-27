from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mechanism_score import ScoreConfig, score_proxy_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "metagb_preflight_adapter" / "outputs" / "preflight_dataset_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "adapter_academic" / "outputs"


def build_scores(df: pd.DataFrame, config: ScoreConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        scored = score_proxy_row(row, config)
        payload = {
            "task_type": row.get("task_type"),
            "dataset": row.get("dataset"),
            "categorical_handling": row.get("categorical_handling"),
            **scored,
        }
        for col in [
            "mgb_relative_improvement",
            "mgb_win",
            "mgb_worth_it",
            "runtime_ratio",
            "selected_pred_feature_fraction__current_metagb",
            "selected_metagb_score_agg__current_metagb",
        ]:
            if col in row:
                payload[f"audit_{col}"] = row.get(col)
        rows.append(payload)
    return pd.DataFrame(rows).sort_values("mechanism_alignment_score", ascending=False)


def build_report(scores: pd.DataFrame) -> str:
    lines = [
        "# Real Benchmark Retrospective Audit",
        "",
        "## Contract",
        "",
        "- Real benchmark outcomes are not used to compute mechanism scores.",
        "- The audit only checks whether intrinsic mechanism scores align with completed experiments.",
        "- High score but no benchmark win should trigger mechanism-to-optimization case analysis, not immediate scorer rejection.",
        "",
        "## Band Counts",
        "",
    ]
    for band, count in scores["mechanism_band"].value_counts().sort_index().items():
        lines.append(f"- {band}: {int(count)}")

    lines.extend(["", "## Ranked Scores", ""])
    display_cols = [
        "dataset",
        "mechanism_band",
        "mechanism_alignment_score",
        "component_linear_leaf_advantage",
        "component_cond_pred_decoupling",
        "component_local_heterogeneity",
        "component_predictive_concentration",
        "component_signal_support",
        "audit_mgb_relative_improvement",
        "audit_mgb_win",
    ]
    available = [col for col in display_cols if col in scores.columns]
    lines.append(scores[available].to_markdown(index=False))

    if "audit_mgb_relative_improvement" in scores.columns:
        grouped = scores.groupby("mechanism_band")["audit_mgb_relative_improvement"].agg(["count", "mean", "median"]).reset_index()
        lines.extend(["", "## Outcome Audit By Mechanism Band", "", grouped.to_markdown(index=False)])
        high_failures = scores[
            scores["mechanism_band"].isin(["STRONG_MECHANISM_MATCH", "MODERATE_MECHANISM_MATCH"])
            & (pd.to_numeric(scores["audit_mgb_relative_improvement"], errors="coerce") <= 0)
        ]
        if not high_failures.empty:
            lines.extend(
                [
                    "",
                    "## High-Score Non-Wins For Case Analysis",
                    "",
                    high_failures[available].to_markdown(index=False),
                    "",
                    "These cases may contain MetaGB-compatible structure that the current training protocol failed to convert into final benchmark gain.",
                ]
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real benchmark audit for intrinsic mechanism scores.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    scores = build_scores(df, ScoreConfig())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "real_audit_scores.csv", index=False)
    (args.output_dir / "real_audit_report.md").write_text(build_report(scores), encoding="utf-8")
    print(f"wrote {args.output_dir / 'real_audit_scores.csv'}")
    print(f"wrote {args.output_dir / 'real_audit_report.md'}")


if __name__ == "__main__":
    main()
