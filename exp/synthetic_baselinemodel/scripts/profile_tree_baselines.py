from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300_baselinemodel.utils.candidates import (  # noqa: E402
    candidate_sets,
    load_search_space,
)
from random300_baselinemodel.utils.model_factory import build_estimator  # noqa: E402
from synthetic_crossgb.scenarios import SCENARIOS, generate_instance  # noqa: E402


TREE_MODELS = ["lightgbm", "catboost", "random_forest"]
DEFAULT_SCENARIOS = ["regime_slope", "xor_classification"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile build/fit/predict time for synthetic tree baselines."
    )
    parser.add_argument("--models", nargs="+", choices=TREE_MODELS, default=TREE_MODELS)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--dimensions", nargs="+", type=int, choices=[10, 20], default=[10, 20])
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--train-rows", type=int, default=1800)
    parser.add_argument("--eval-rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--search-space-json",
        type=Path,
        default=(
            ROOT
            / "synthetic_baselinemodel"
            / "configs"
            / "default_search_space.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "synthetic_baselinemodel" / "results" / "tree_profile",
    )
    return parser


def fit_model(model: Any, X: np.ndarray, y: np.ndarray) -> None:
    try:
        model.fit(X, y, verbose=False)
    except (TypeError, ValueError) as exc:
        if "verbose" not in str(exc):
            raise
        model.fit(X, y)


def predict_model(model: Any, X: np.ndarray, task_type: str) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBM.* was fitted with feature names",
            category=UserWarning,
        )
        if task_type == "regression":
            model.predict(X)
        else:
            model.predict_proba(X)


def tree_count(model: str, params: dict[str, Any]) -> int:
    if model == "catboost":
        return int(params["iterations"])
    return int(params["n_estimators"])


def tree_depth(model: str, params: dict[str, Any]) -> int | None:
    if model == "catboost":
        return int(params["depth"])
    value = params.get("max_depth")
    return None if value is None else int(value)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted(
        {(row["model"], row["scenario"], row["dimension"]) for row in rows}
    )
    for model, scenario, dimension in keys:
        group = [
            row
            for row in rows
            if row["model"] == model
            and row["scenario"] == scenario
            and row["dimension"] == dimension
        ]
        build_values = [float(row["build_sec"]) for row in group]
        fit_values = [float(row["fit_sec"]) for row in group]
        predict_values = [float(row["predict_sec"]) for row in group]
        total = sum(build_values) + sum(fit_values) + sum(predict_values)
        summaries.append(
            {
                "model": model,
                "scenario": scenario,
                "dimension": dimension,
                "fits": len(group),
                "mean_tree_count": statistics.mean(
                    int(row["tree_count"]) for row in group
                ),
                "mean_build_sec": statistics.mean(build_values),
                "mean_fit_sec": statistics.mean(fit_values),
                "median_fit_sec": statistics.median(fit_values),
                "p90_fit_sec": percentile(fit_values, 0.9),
                "mean_predict_sec": statistics.mean(predict_values),
                "fit_time_fraction": sum(fit_values) / total if total else 0.0,
                "fit_sec_per_100_trees": statistics.mean(
                    float(row["fit_sec"]) * 100.0 / int(row["tree_count"])
                    for row in group
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if args.train_rows + args.eval_rows > args.n_samples:
        raise ValueError("train rows plus eval rows exceed --n-samples")
    unknown = [name for name in args.scenarios if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")

    search_space = load_search_space(args.search_space_json)
    candidates = candidate_sets(
        args.models,
        args.candidate_count,
        args.seed,
        search_space,
    )
    model_args = argparse.Namespace(seed=args.seed, n_jobs=args.n_jobs)
    rows: list[dict[str, Any]] = []

    for scenario in args.scenarios:
        task_type = SCENARIOS[scenario].task_type
        noise = 0.05 if task_type != "regression" else 0.15
        for dimension in args.dimensions:
            X, y = generate_instance(
                scenario,
                args.seed,
                args.n_samples,
                noise,
                target_features=dimension,
            )
            X_train = X[: args.train_rows]
            y_train = y[: args.train_rows]
            X_eval = X[args.train_rows : args.train_rows + args.eval_rows]
            n_classes = 1 if task_type == "regression" else int(len(np.unique(y)))
            for model_name in args.models:
                for candidate_index, params in enumerate(
                    candidates[model_name], start=1
                ):
                    start = time.perf_counter()
                    estimator = build_estimator(
                        model_name,
                        params,
                        model_args,
                        task_type,
                        n_classes,
                    )
                    build_sec = time.perf_counter() - start

                    start = time.perf_counter()
                    fit_model(estimator, X_train, y_train)
                    fit_sec = time.perf_counter() - start

                    start = time.perf_counter()
                    predict_model(estimator, X_eval, task_type)
                    predict_sec = time.perf_counter() - start

                    row = {
                        "model": model_name,
                        "scenario": scenario,
                        "task_type": task_type,
                        "dimension": dimension,
                        "candidate_index": candidate_index,
                        "candidate_id": params["candidate_id"],
                        "tree_count": tree_count(model_name, params),
                        "depth": tree_depth(model_name, params),
                        "num_leaves": params.get("num_leaves"),
                        "min_samples_leaf": params.get("min_samples_leaf"),
                        "max_features": params.get("max_features"),
                        "build_sec": build_sec,
                        "fit_sec": fit_sec,
                        "predict_sec": predict_sec,
                    }
                    rows.append(row)
                    print(
                        f"{model_name} {scenario} dim={dimension} "
                        f"candidate={candidate_index}/{args.candidate_count} "
                        f"trees={row['tree_count']} depth={row['depth']} "
                        f"fit={fit_sec:.3f}s",
                        flush=True,
                    )

    summary = summarize(rows)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    write_csv(output_dir / "fits.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nSummary:")
    for row in summary:
        print(
            f"{row['model']:13s} {row['scenario']:18s} dim={row['dimension']:2d} "
            f"mean_fit={row['mean_fit_sec']:.3f}s "
            f"p90={row['p90_fit_sec']:.3f}s "
            f"fit_share={row['fit_time_fraction']:.1%} "
            f"sec/100trees={row['fit_sec_per_100_trees']:.3f}",
            flush=True,
        )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

