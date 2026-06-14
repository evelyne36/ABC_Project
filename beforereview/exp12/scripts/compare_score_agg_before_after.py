"""Run the fixed score_agg grid against one selected XGBoost binary."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


WORKSPACE = Path(__file__).resolve().parents[3]
DATASET_DIR = WORKSPACE / "dataset" / "TabArena_regression"
DATASETS = ["airfoil_self_noise.csv", "concrete_compressive_strength.csv"]
MAX_DEPTHS = [2, 4, 6]
N_ESTIMATORS_VALUES = [50, 200, 500]
SCORE_AGGS = [0, 1, 2, 3]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-package", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
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

    fixed_params = {
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
        "random_state": SEED,
        "n_jobs": 1,
        "verbosity": 0,
    }
    rows: list[dict[str, object]] = []
    folds = KFold(n_splits=3, shuffle=True, random_state=SEED)
    for dataset in DATASETS:
        X, y = load_dataset(dataset)
        split_indices = list(folds.split(X))
        for max_depth in MAX_DEPTHS:
            for n_estimators in N_ESTIMATORS_VALUES:
                for score_agg in SCORE_AGGS:
                    for fold, (train_idx, test_idx) in enumerate(split_indices, start=1):
                        model = xgb.XGBRegressor(
                            **fixed_params,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            metagb_score_agg=score_agg,
                        )
                        started = time.perf_counter()
                        model.fit(X[train_idx], y[train_idx])
                        fit_seconds = time.perf_counter() - started
                        prediction = model.predict(X[test_idx])
                        rows.append(
                            {
                                "label": args.label,
                                "dataset": dataset,
                                "max_depth": max_depth,
                                "n_estimators": n_estimators,
                                "score_agg": score_agg,
                                "fold": fold,
                                "test_rmse": mean_squared_error(y[test_idx], prediction) ** 0.5,
                                "test_r2": r2_score(y[test_idx], prediction),
                                "fit_seconds": fit_seconds,
                            }
                        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"{args.label}: {len(rows)} fits, library={xgb.build_info()['libxgboost']}, output={output}")


if __name__ == "__main__":
    main()
