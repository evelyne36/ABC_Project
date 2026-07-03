from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "adapter_academic" / "outputs" / "synthetic_scores.csv"
DEFAULT_OUTPUT_DIR = ROOT / "adapter_academic" / "outputs"

COMPONENTS = [
    "component_linear_leaf_advantage",
    "component_cond_pred_decoupling",
    "component_local_heterogeneity",
    "component_predictive_concentration",
    "component_signal_support",
]

DEFAULT_WEIGHTS = {
    "component_linear_leaf_advantage": 0.35,
    "component_cond_pred_decoupling": 0.25,
    "component_local_heterogeneity": 0.20,
    "component_predictive_concentration": 0.10,
    "component_signal_support": 0.10,
}


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    denom = sum(max(value, 0.0) for value in weights.values())
    if denom <= 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    total = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for component, weight in weights.items():
        total += float(weight) * pd.to_numeric(df[component], errors="coerce").fillna(0.0)
    return (total / denom).clip(0.0, 1.0)


def auc_or_nan(y: pd.Series, score: pd.Series) -> float:
    try:
        return float(roc_auc_score(y.astype(int), score.astype(float)))
    except ValueError:
        return np.nan


def build_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = df["expected_mechanism_match"].astype(int)
    base = weighted_score(df, DEFAULT_WEIGHTS)
    rows.append({"variant": "full", "removed_component": "", "auc": auc_or_nan(y, base), "mean_score": float(base.mean())})
    for removed in COMPONENTS:
        weights = dict(DEFAULT_WEIGHTS)
        weights[removed] = 0.0
        score = weighted_score(df, weights)
        rows.append(
            {
                "variant": f"remove_{removed.replace('component_', '')}",
                "removed_component": removed,
                "auc": auc_or_nan(y, score),
                "mean_score": float(score.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = df["expected_mechanism_match"].astype(int)
    for linear in [0.20, 0.35, 0.50]:
        for decoupling in [0.15, 0.25, 0.35]:
            weights = dict(DEFAULT_WEIGHTS)
            weights["component_linear_leaf_advantage"] = linear
            weights["component_cond_pred_decoupling"] = decoupling
            score = weighted_score(df, weights)
            rows.append(
                {
                    "weight_linear": linear,
                    "weight_decoupling": decoupling,
                    "auc": auc_or_nan(y, score),
                    "mean_expected_positive_score": float(score[y == 1].mean()),
                    "mean_expected_negative_score": float(score[y == 0].mean()),
                    "score_gap": float(score[y == 1].mean() - score[y == 0].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_report(ablation: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    lines = [
        "# Ablation And Sensitivity Report",
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False),
        "",
        "## Sensitivity",
        "",
        sensitivity.sort_values("auc", ascending=False).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Ablation tests whether each mechanism component is necessary for synthetic mechanism recovery.",
        "- Sensitivity checks whether the score is robust to plausible theory-weight changes.",
        "- These checks address the NeurIPS-style concern that the score is an arbitrary heuristic.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation and sensitivity checks over synthetic mechanism scores.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    ablation = build_ablation(df)
    sensitivity = build_sensitivity(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(args.output_dir / "ablation_results.csv", index=False)
    sensitivity.to_csv(args.output_dir / "sensitivity_results.csv", index=False)
    (args.output_dir / "ablation_sensitivity_report.md").write_text(build_report(ablation, sensitivity), encoding="utf-8")
    print(f"wrote {args.output_dir / 'ablation_results.csv'}")
    print(f"wrote {args.output_dir / 'sensitivity_results.csv'}")
    print(f"wrote {args.output_dir / 'ablation_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
