#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B timing for current_metagb histogram build modes.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset" / "TabArena_regression"))
    parser.add_argument("--datasets", nargs="+", default=["airfoil_self_noise.csv"])
    parser.add_argument("--xgboost-python-package", default=str(ROOT / "beforereview" / "xgboost-master" / "python-package"))
    parser.add_argument("--modes", nargs="+", default=["row_serial", "cf_parallel"])
    parser.add_argument("--hist-subtract", choices=["true", "false", "both"], default="true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--pred-feature-fraction", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=str(ROOT / "beforereview" / "exp12" / "results" / "hist_build_demo.csv"))
    return parser.parse_args()


def import_local_xgboost(python_package: Path):
    python_package = python_package.expanduser().resolve()
    for module_name in list(sys.modules):
        if module_name == "xgboost" or module_name.startswith("xgboost."):
            del sys.modules[module_name]
    sys.path.insert(0, str(python_package))
    import xgboost as xgb

    return xgb


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    X_df = df.iloc[:, :-1].copy()
    for column in X_df.select_dtypes(exclude=[np.number]).columns:
        X_df[column], _ = pd.factorize(X_df[column], sort=True)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    combined = X_df.copy()
    combined["__target__"] = y
    combined = combined.dropna()
    X = combined.drop(columns=["__target__"]).to_numpy(dtype=np.float32, copy=True)
    y_arr = combined["__target__"].to_numpy(dtype=np.float32, copy=True)
    return X, y_arr


def build_model(xgb, args: argparse.Namespace, hist_subtract: bool):
    return xgb.XGBRegressor(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        min_child_weight=args.min_child_weight,
        max_bin=args.max_bin,
        tree_method="hist",
        device="cpu",
        objective="reg:squarederror",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=0,
        enable_metagb=True,
        metagb_pred_colsample_bytree=args.pred_feature_fraction,
        metagb_score_agg=0,
        metagb_lambda_l2=1.0,
        metagb_hist_subtract=hist_subtract,
        base_score=0.0,
    )


def main() -> int:
    args = parse_args()
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    rows = []
    dataset_dir = Path(args.dataset_dir)
    for dataset_name in args.datasets:
        X, y = load_dataset(dataset_dir / dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        print(f"[{dataset_name}] rows={X.shape[0]} features={X.shape[1]}", flush=True)
        for mode in args.modes:
            os.environ["METAGB_HIST_BUILD_MODE"] = mode
            hist_subtract_values = (
                [True, False]
                if args.hist_subtract == "both"
                else [args.hist_subtract == "true"]
            )
            for hist_subtract in hist_subtract_values:
                for rep in range(args.repeat):
                    model = build_model(xgb, args, hist_subtract)
                    fit_start = time.perf_counter()
                    model.fit(X_train, y_train)
                    train_s = time.perf_counter() - fit_start
                    pred = model.predict(X_test)
                    rmse = mean_squared_error(y_test, pred) ** 0.5
                    row = {
                        "dataset": dataset_name,
                        "rows": X.shape[0],
                        "features": X.shape[1],
                        "mode": mode,
                        "hist_subtract": hist_subtract,
                        "repeat": rep + 1,
                        "n_jobs": args.n_jobs,
                        "n_estimators": args.n_estimators,
                        "max_depth": args.max_depth,
                        "colsample": args.colsample,
                        "pred_feature_fraction": args.pred_feature_fraction,
                        "train_s": train_s,
                        "rmse": rmse,
                    }
                    rows.append(row)
                    print(
                        f"  {mode:12s} hist_subtract={str(hist_subtract):5s} "
                        f"rep={rep + 1} train={train_s:.4f}s rmse={rmse:.6g}",
                        flush=True,
                    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
