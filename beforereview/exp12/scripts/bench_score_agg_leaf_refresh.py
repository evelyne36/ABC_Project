"""Benchmark current MetaGB before/after explicit final-leaf refresh."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
DATASET_DIR = WORKSPACE / "dataset" / "TabArena_regression"
DATASETS = ["airfoil_self_noise.csv", "concrete_compressive_strength.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-package", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATASET_DIR / name).apply(pd.to_numeric, errors="coerce").dropna()
    return (
        frame.iloc[:, :-1].to_numpy(dtype=np.float32, copy=True),
        frame.iloc[:, -1].to_numpy(dtype=np.float32, copy=True),
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.python_package).resolve()))
    import xgboost as xgb

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "max_bin": 64,
        "tree_method": "hist",
        "device": "cpu",
        "objective": "reg:squarederror",
        "enable_metagb": True,
        "metagb_pred_colsample_bytree": 1.0,
        "metagb_lambda_l2": 1.0,
        "base_score": 0.0,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    }
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        X, y = load_dataset(dataset)
        for score_agg in range(4):
            for repeat in range(args.repeats + 1):
                model = xgb.XGBRegressor(**params, metagb_score_agg=score_agg)
                started = time.perf_counter()
                model.fit(X, y)
                fit_seconds = time.perf_counter() - started
                if repeat > 0:
                    rows.append(
                        {
                            "label": args.label,
                            "dataset": dataset,
                            "score_agg": score_agg,
                            "repeat": repeat,
                            "fit_seconds": fit_seconds,
                        }
                    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"label": args.label, "library": xgb.build_info()["libxgboost"]}))
    print(pd.DataFrame(rows).groupby(["dataset", "score_agg"]).fit_seconds.median())


if __name__ == "__main__":
    main()
