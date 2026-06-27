#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {
    "regression": ROOT / "dataset" / "TabArena_regression",
    "binary": ROOT / "dataset" / "TabArena_binary",
    "multiclass": ROOT / "dataset" / "TabArena_multiclass",
}

STATUS_GROUPS = [
    ("binary", "drop", "binary_drop_0624"),
    ("binary", "onehot", "binary_onehot_0624"),
    ("multiclass", "drop", "multiclass_drop_0624"),
    ("multiclass", "onehot", "multiclass_onehot_0624"),
]

EXCLUDED_DATASETS = {"in_vehicle_coupon_recommendation.csv"}

SLOW_NOTES = {
    ("binary", "drop", "Bioresponse.csv"): "USER_SLOW: 1epoch/132h",
    ("binary", "drop", "NATICUSdroid.csv"): "USER_SLOW: 50epoch/34h",
    ("binary", "drop", "taiwanese_bankruptcy_prediction.csv"): "USER_SLOW: 50epoch/39h",
    ("binary", "onehot", "Marketing_Campaign.csv"): "USER_SLOW: 1epoch/2h",
    ("binary", "onehot", "polish_companies_bankruptcy.csv"): "USER_SLOW: 50epoch/12h",
    ("multiclass", "drop", "hiva_agnostic.csv"): "USER_VERY_SLOW: 1epoch/7d",
    ("multiclass", "onehot", "anneal.csv"): "USER_SLOW: 50epoch/46h",
    ("multiclass", "onehot", "splice.csv"): "USER_SLOW: 1epoch/6h",
    ("multiclass", "onehot", "hiva_agnostic.csv"): "USER_SLOW: unknown, likely very high",
    ("multiclass", "onehot", "students_dropout_and_academic_success.csv"): "USER_RUNNING: student ing",
}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def preprocessing_stats(csv_path: Path, task_type: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    raw_rows = int(len(df))
    raw_features = int(df.shape[1] - 1)
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    target_notna = target.notna()
    after_target_rows = int(target_notna.sum())
    features_after_target = features.loc[target_notna]
    target_after_target = target.loc[target_notna]

    full_feature_notna = features_after_target.notna().all(axis=1)
    after_full_feature_drop_rows = int(full_feature_notna.sum())
    strict_base = features_after_target.loc[full_feature_notna]
    strict_target = target_after_target.loc[full_feature_notna]

    strict_drop = strict_base.select_dtypes(include=[np.number])
    onehot_base = pd.get_dummies(
        strict_base,
        columns=list(strict_base.select_dtypes(exclude=[np.number]).columns),
        drop_first=False,
    )

    legacy_drop = features_after_target.select_dtypes(include=[np.number]).dropna()

    regression_target_valid = after_full_feature_drop_rows
    if task_type == "regression":
        y = pd.to_numeric(strict_target, errors="coerce")
        regression_target_valid = int(y.notna().sum())

    return {
        "raw_rows": raw_rows,
        "raw_features": raw_features,
        "target_missing_rows": int(raw_rows - after_target_rows),
        "target_missing_rate": safe_div(raw_rows - after_target_rows, raw_rows),
        "feature_missing_rows_after_target_drop": int(after_target_rows - after_full_feature_drop_rows),
        "feature_missing_drop_rate_after_target_drop": safe_div(after_target_rows - after_full_feature_drop_rows, after_target_rows),
        "any_missing_drop_rate_raw": safe_div(raw_rows - after_full_feature_drop_rows, raw_rows),
        "strict_drop_rows": int(len(strict_drop)),
        "strict_drop_features": int(strict_drop.shape[1]),
        "onehot_rows": int(len(onehot_base)),
        "onehot_features": int(onehot_base.shape[1]),
        "legacy_drop_rows": int(len(legacy_drop)),
        "legacy_drop_features": int(legacy_drop.shape[1]),
        "strict_minus_legacy_rows": int(len(strict_drop) - len(legacy_drop)),
        "regression_target_valid_rows_after_strict_drop": regression_target_valid,
        "strict_pressure_proxy": int(len(strict_drop) * (strict_drop.shape[1] ** 2)) if strict_drop.shape[1] else None,
        "onehot_pressure_proxy": int(len(onehot_base) * (onehot_base.shape[1] ** 2)) if onehot_base.shape[1] else None,
    }


def build_missing_drop_rates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_type, dataset_dir in DATASET_DIRS.items():
        for csv_path in sorted(dataset_dir.glob("*.csv")):
            stats = preprocessing_stats(csv_path, task_type)
            rows.append({"task_type": task_type, "dataset": csv_path.name, **stats})
    return rows


def latest_statuses(datacollection_root: Path, group: str) -> dict[str, tuple[str, str]]:
    latest: dict[str, tuple[str, str]] = {}
    for path in (datacollection_root / group).glob("**/task_status.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        updated_at = str(data.get("updated_at", ""))
        for task_id, status in data.get("tasks", {}).items():
            if task_id not in latest or updated_at > latest[task_id][0]:
                latest[task_id] = (updated_at, str(status))
    return latest


def build_queue(datacollection_root: Path, missing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing_by_key = {(row["task_type"], row["dataset"]): row for row in missing_rows}
    queue_rows: list[dict[str, Any]] = []
    for task_type, handling, group in STATUS_GROUPS:
        by_dataset_model: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for task_id, (_, status) in latest_statuses(datacollection_root, group).items():
            parts = task_id.split("__")
            if len(parts) < 4:
                continue
            _, dataset, model, _ = parts
            if dataset in EXCLUDED_DATASETS:
                continue
            by_dataset_model[(dataset, model)][status] += 1

        for dataset, model in sorted({key for key in by_dataset_model}):
            if model not in {"xgb_cpu", "current_metagb"}:
                continue
        datasets = sorted({dataset for dataset, _ in by_dataset_model})
        for dataset in datasets:
            if dataset in EXCLUDED_DATASETS:
                continue
            xgb_counter = by_dataset_model.get((dataset, "xgb_cpu"), Counter())
            mgb_counter = by_dataset_model.get((dataset, "current_metagb"), Counter())
            xgb_pending = xgb_counter.get("pending", 0)
            mgb_pending = mgb_counter.get("pending", 0)
            if not xgb_pending and not mgb_pending:
                continue
            stats = missing_by_key.get((task_type, dataset), {})
            if handling == "drop":
                processed_rows = stats.get("strict_drop_rows")
                processed_features = stats.get("strict_drop_features")
                pressure = stats.get("strict_pressure_proxy")
            else:
                processed_rows = stats.get("onehot_rows")
                processed_features = stats.get("onehot_features")
                pressure = stats.get("onehot_pressure_proxy")
            note = SLOW_NOTES.get((task_type, handling, dataset), "")
            slow_rank = 2 if "VERY_SLOW" in note or "7d" in note else (1 if note else 0)
            preprocess_error = ""
            if not processed_rows or not processed_features:
                preprocess_error = "no rows/features after preprocessing"
            queue_rows.append(
                {
                    "task_type": task_type,
                    "handling": handling,
                    "dataset": dataset,
                    "processed_rows": processed_rows,
                    "processed_features": processed_features,
                    "pressure_proxy": pressure,
                    "xgb_pending_folds": xgb_pending,
                    "metagb_pending_folds": mgb_pending,
                    "note": note,
                    "slow_rank": slow_rank,
                    "preprocess_error": preprocess_error,
                }
            )
    return sorted(
        queue_rows,
        key=lambda row: (
            row["preprocess_error"] != "",
            row["slow_rank"] > 0,
            row["slow_rank"],
            row["pressure_proxy"] if row["pressure_proxy"] is not None else 10**30,
            row["task_type"],
            row["handling"],
            row["dataset"],
        ),
    )


def write_report(output_dir: Path, missing_rows: list[dict[str, Any]], queue_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(missing_rows)
    grouped = df.groupby("task_type", as_index=False).agg(
        datasets=("dataset", "count"),
        raw_rows=("raw_rows", "sum"),
        feature_missing_rows_after_target_drop=("feature_missing_rows_after_target_drop", "sum"),
        target_missing_rows=("target_missing_rows", "sum"),
    )
    grouped["feature_missing_drop_rate_after_target_drop"] = (
        grouped["feature_missing_rows_after_target_drop"] / (grouped["raw_rows"] - grouped["target_missing_rows"])
    )
    lines = [
        "# TabArena Missing-Row and Queue Audit",
        "",
        "## Missing Row Drop Rate by Task Type",
        "",
        grouped.to_markdown(index=False),
        "",
        "## Queue Notes",
        "",
        "- `in_vehicle_coupon_recommendation.csv` is excluded from the queue because one-hot strict missing-row removal leaves only 108 rows.",
        "- Drop preprocessing in `tabarena_30` uses strict row filtering first, then drops discrete columns; this aligns row counts with one-hot for datasets such as HR.",
        "- User-marked slow datasets are forced to the tail of `queue_priority.csv`.",
        "",
        "## First 20 Queue Items",
        "",
        pd.DataFrame(queue_rows).head(20).to_markdown(index=False),
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TabArena missing-row rates and tabarena_30 queue priority.")
    parser.add_argument("--datacollection-root", type=Path, default=ROOT / "datacollection0624")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tabarena_30" / "audits")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_rows = build_missing_drop_rates()
    queue_rows = build_queue(args.datacollection_root, missing_rows)
    pd.DataFrame(missing_rows).to_csv(args.output_dir / "tabarena_missing_drop_rates.csv", index=False)
    pd.DataFrame(queue_rows).to_csv(args.output_dir / "queue_priority.csv", index=False)
    write_report(args.output_dir, missing_rows, queue_rows)
    print(f"wrote {args.output_dir / 'tabarena_missing_drop_rates.csv'}")
    print(f"wrote {args.output_dir / 'queue_priority.csv'}")
    print(f"wrote {args.output_dir / 'audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
