from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


MODELS = ["lightgbm", "catboost", "random_forest"]
TIMESTAMP_RE = re.compile(r"^\[([^\]]+)\]")
CONFIG_RE = re.compile(
    r"model=(?P<model>\S+).*"
    r"task_type=(?P<task_type>\w+).*"
    r"outer_fold=\d+/(?P<outer_folds>\d+).*"
    r"inner_folds=(?P<inner_folds>\d+).*"
    r"candidates=(?P<candidates>\d+).*"
    r"rows=(?P<rows>\d+).*"
    r"features=(?P<features>\d+)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate synthetic baseline runtime from completed Random300 task logs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(
            "datacollection0702/datacollection_baselinemodel_random300/tasks"
        ),
    )
    parser.add_argument("--min-rows", type=int, default=2500)
    parser.add_argument("--max-rows", type=int, default=3500)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--task-workers", type=int, default=15)
    parser.add_argument("--tasks-per-model", type=int, default=190)
    parser.add_argument("--json", action="store_true")
    return parser


def parse_log(path: Path) -> dict[str, Any] | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any("] finish " in line for line in lines):
        return None
    config_line = next((line for line in lines if "] config " in line), "")
    match = CONFIG_RE.search(config_line)
    if match is None:
        return None
    timestamps = []
    for line in lines:
        timestamp_match = TIMESTAMP_RE.match(line)
        if timestamp_match is None:
            continue
        try:
            timestamps.append(
                datetime.strptime(
                    timestamp_match.group(1), "%Y-%m-%d %H:%M:%S"
                )
            )
        except ValueError:
            continue
    if len(timestamps) < 2:
        return None
    payload: dict[str, Any] = match.groupdict()
    for key in (
        "outer_folds",
        "inner_folds",
        "candidates",
        "rows",
        "features",
    ):
        payload[key] = int(payload[key])
    payload["duration_sec"] = (timestamps[-1] - timestamps[0]).total_seconds()
    payload["path"] = str(path)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    records = []
    for path in args.logs_dir.glob("*.log"):
        record = parse_log(path)
        if record is None:
            continue
        if record["model"] not in MODELS:
            continue
        if record["task_type"] not in {"regression", "binary"}:
            continue
        if record["outer_folds"] != 5 or record["inner_folds"] != 4:
            continue
        if record["candidates"] != 300:
            continue
        if not args.min_rows <= record["rows"] <= args.max_rows:
            continue
        if not args.min_features <= record["features"] <= args.max_features:
            continue
        records.append(record)

    rows = []
    for model in MODELS:
        durations = [
            record["duration_sec"]
            for record in records
            if record["model"] == model
        ]
        if not durations:
            raise RuntimeError(f"No matching completed logs for {model}")
        median_sec = float(statistics.median(durations))
        rows.append(
            {
                "model": model,
                "matching_logs": len(durations),
                "median_task_sec": median_sec,
                "projected_worker_hours": (
                    median_sec * args.tasks_per_model / 3600.0
                ),
            }
        )
    total_worker_hours = sum(row["projected_worker_hours"] for row in rows)
    payload = {
        "filters": {
            "rows": [args.min_rows, args.max_rows],
            "features": [args.min_features, args.max_features],
            "task_types": ["regression", "binary"],
            "outer_folds": 5,
            "inner_folds": 4,
            "candidates": 300,
        },
        "tasks_per_model": args.tasks_per_model,
        "task_workers": args.task_workers,
        "models": rows,
        "total_projected_worker_hours": total_worker_hours,
        "ideal_projected_wall_hours": total_worker_hours / args.task_workers,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("| model | logs | median task sec | projected worker-hours |")
        print("| --- | ---: | ---: | ---: |")
        for row in rows:
            print(
                f"| {row['model']} | {row['matching_logs']} | "
                f"{row['median_task_sec']:.1f} | "
                f"{row['projected_worker_hours']:.1f} |"
            )
        print(
            f"\ntotal_worker_hours={total_worker_hours:.1f} "
            f"ideal_wall_hours={payload['ideal_projected_wall_hours']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
