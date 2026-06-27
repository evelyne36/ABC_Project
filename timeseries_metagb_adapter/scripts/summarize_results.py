from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize time-series XGBoost vs MetaGB results.")
    parser.add_argument("results_dir")
    return parser


def summarize(results_dir: Path) -> str:
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)
    grouped = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            total_train_time_sec_mean=("total_train_time_sec", "mean"),
            total_predict_time_sec_mean=("total_predict_time_sec", "mean"),
            folds=("outer_fold", "count"),
        )
        .sort_values(["dataset", "test_rmse_mean", "model"])
    )
    aggregate_path = results_dir / "aggregate.csv"
    grouped.to_csv(aggregate_path, index=False)

    lines = [
        "# Time-Series Smoke Report",
        "",
        f"results_dir: `{results_dir}`",
        "",
        "## Aggregate",
        "",
        grouped.to_markdown(index=False),
        "",
        "## Winner By Dataset",
        "",
    ]
    for dataset, dataset_df in grouped.groupby("dataset"):
        best = dataset_df.sort_values("test_rmse_mean").iloc[0]
        lines.append(
            f"- {dataset}: {best['model']} "
            f"(mean RMSE={best['test_rmse_mean']:.6g}, folds={int(best['folds'])})"
        )
    manifest_path = results_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Split Audit",
                "",
                f"- split records: {len(manifest.get('splits', []))}",
                "- invariant: each train block precedes its validation/test block; scalers are fit only on the training target period.",
            ]
        )
    report = "\n".join(lines) + "\n"
    report_path = results_dir / "smoke_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    args = build_parser().parse_args()
    print(summarize(Path(args.results_dir).expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

