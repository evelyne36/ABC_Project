"""Small fixed-parameter ablation for MetaGB score aggregation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
XGBOOST_PACKAGE = ROOT / "xgboost-master" / "python-package"
DATASET_DIR = WORKSPACE / "dataset" / "TabArena_regression"
OUTPUT_DIR = ROOT / "exp12" / "results" / "score_agg_ablation"

DATASETS = ["airfoil_self_noise.csv", "concrete_compressive_strength.csv"]
SCORE_AGGS = [0, 1, 2, 3]
MAX_DEPTHS = [2, 4, 6]
N_ESTIMATORS_VALUES = [50, 200, 500]
SEED = 42
N_SPLITS = 3

FIXED_PARAMS = {
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
    # score_agg=3 uses the original global rand() sequence during split evaluation.
    "n_jobs": 1,
    "verbosity": 0,
}


def import_local_xgboost():
    sys.path.insert(0, str(XGBOOST_PACKAGE))
    import xgboost as xgb

    return xgb


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna()
    X = numeric.iloc[:, :-1].to_numpy(dtype=np.float32, copy=True)
    y = numeric.iloc[:, -1].to_numpy(dtype=np.float32, copy=True)
    return X, y


def main() -> None:
    xgb = import_local_xgboost()
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows: list[dict[str, object]] = []

    print("Fixed parameters:")
    print(json.dumps(FIXED_PARAMS, indent=2, sort_keys=True))
    for dataset in DATASETS:
        X, y = load_dataset(DATASET_DIR / dataset)
        print(f"\n{dataset}: rows={X.shape[0]}, features={X.shape[1]}")
        split_indices = list(folds.split(X))
        for max_depth in MAX_DEPTHS:
            for n_estimators in N_ESTIMATORS_VALUES:
                print(f"  depth={max_depth}, trees={n_estimators}")
                for score_agg in SCORE_AGGS:
                    for fold, (train_idx, test_idx) in enumerate(split_indices, start=1):
                        model = xgb.XGBRegressor(
                            **FIXED_PARAMS,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            metagb_score_agg=score_agg,
                        )
                        started = time.perf_counter()
                        model.fit(X[train_idx], y[train_idx])
                        fit_seconds = time.perf_counter() - started
                        train_pred = model.predict(X[train_idx])
                        test_pred = model.predict(X[test_idx])
                        row = {
                            "dataset": dataset,
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                            "score_agg": score_agg,
                            "fold": fold,
                            "train_rmse": mean_squared_error(y[train_idx], train_pred) ** 0.5,
                            "test_rmse": mean_squared_error(y[test_idx], test_pred) ** 0.5,
                            "test_r2": r2_score(y[test_idx], test_pred),
                            "fit_seconds": fit_seconds,
                        }
                        rows.append(row)
                        print(
                            f"    agg={score_agg} fold={fold}: "
                            f"test_rmse={row['test_rmse']:.5f} "
                            f"r2={row['test_r2']:.5f} "
                            f"fit={fit_seconds:.2f}s"
                        )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dataset", "max_depth", "n_estimators", "score_agg"], as_index=False)
        .agg(
            train_rmse_mean=("train_rmse", "mean"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .sort_values(["dataset", "max_depth", "n_estimators", "test_rmse_mean"])
    )
    best = summary.groupby(
        ["dataset", "max_depth", "n_estimators"], as_index=False
    ).first()[["dataset", "max_depth", "n_estimators", "score_agg", "test_rmse_mean"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_DIR / "score_agg_detail.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "score_agg_summary.csv", index=False)
    metadata = {
        "datasets": DATASETS,
        "score_aggs": SCORE_AGGS,
        "max_depths": MAX_DEPTHS,
        "n_estimators_values": N_ESTIMATORS_VALUES,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "fixed_params": FIXED_PARAMS,
        "xgboost_version": xgb.__version__,
    }
    (OUTPUT_DIR / "score_agg_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nSummary (lower test RMSE is better):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBest score_agg per dataset:")
    print(best.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nResults: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
