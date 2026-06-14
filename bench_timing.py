"""
Quick timing/RMSE comparison on one train/test split.

Compared models:
  - original_metagb: standalone CFFI MetaGB in ./metagb
  - current_metagb: MetaGB embedded in ./beforereview/xgboost-master
  - xgb_cpu:        standard CPU hist from the same embedded XGBoost checkout

This is a fast smoke benchmark, not a nested-CV experiment. It keeps the same
data split and comparable hyperparameters across the three models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CURRENT_ROOT = ROOT / "beforereview"
DEFAULT_DATASET_DIR = CURRENT_ROOT / "dataset" / "tabArena_regression"
DEFAULT_XGBOOST_PACKAGE = CURRENT_ROOT / "xgboost-master" / "python-package"
DEFAULT_ORIGINAL_METAGB_DIR = ROOT / "metagb"

DEFAULT_DATASETS = [
    "airfoil_self_noise.csv",
    "concrete_compressive_strength.csv",
    "wine_quality.csv",
    "physiochemical_protein.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick original/current MetaGB/XGBoost timing benchmark.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--dataset-dir", default=os.environ.get("METAGB_DATASET_DIR", str(DEFAULT_DATASET_DIR)))
    parser.add_argument(
        "--xgboost-python-package",
        default=os.environ.get("METAGB_XGBOOST_PYTHON_PACKAGE", str(DEFAULT_XGBOOST_PACKAGE)),
    )
    parser.add_argument(
        "--original-metagb-dir",
        default=os.environ.get("ORIGINAL_METAGB_DIR", str(DEFAULT_ORIGINAL_METAGB_DIR)),
    )
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("BENCH_N_JOBS", "4")))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--pred-feature-fraction", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=10.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--metagb-lambda-l2", type=float, default=0.0)
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=str(CURRENT_ROOT / "exp12" / "results" / "bench_timing_latest.csv"))
    return parser.parse_args()


def import_local_xgboost(python_package: Path) -> Any:
    python_package = python_package.expanduser().resolve()
    if not python_package.exists():
        raise FileNotFoundError(f"Local XGBoost python package not found: {python_package}")
    for module_name in list(sys.modules):
        if module_name == "xgboost" or module_name.startswith("xgboost."):
            del sys.modules[module_name]
    sys.path.insert(0, str(python_package))
    import xgboost as xgb

    return xgb


def import_original_metagb(original_metagb_dir: Path) -> Any:
    original_metagb_dir = original_metagb_dir.expanduser().resolve()
    if not original_metagb_dir.exists():
        raise FileNotFoundError(f"Original MetaGB directory not found: {original_metagb_dir}")
    if str(original_metagb_dir) not in sys.path:
        sys.path.insert(0, str(original_metagb_dir))
    from metagb_utils import MetaGB

    return MetaGB


class OriginalMetaGBRegressor:
    def __init__(self, metagb_class: Any, params: Dict[str, Any], args: argparse.Namespace) -> None:
        self.model = metagb_class(
            learning_rate=float(params["learning_rate"]),
            num_trees=int(params["n_estimators"]),
            feature_fraction=float(params["colsample"]),
            pred_feature_fraction=float(params["pred_feature_fraction"]),
            bagging_fraction=float(params["subsample"]),
            min_data_in_leaf=max(1, int(round(float(params["min_child_weight"])))),
            max_depth=int(params["max_depth"]),
            max_bin=int(params["max_bin"]),
            version=3,
            leaf_only_use_bias=0,
            cond_pred_share=0,
            use_delta=1,
            score_agg=int(args.metagb_score_agg),
            lambda_l2=float(args.metagb_lambda_l2),
            min_gain_to_split=-1e10,
            split_sample=0,
            seed=int(args.seed),
        )

    def fit(self, X: Any, y: Any) -> "OriginalMetaGBRegressor":
        self.model.fit(X.astype("double", copy=False), y.astype("double", copy=False))
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X.astype("double", copy=False))


def load_dataset(path: Path) -> tuple[Any, Any]:
    df = pd.read_csv(path)
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    X_df = df.iloc[:, :-1].copy()
    categorical_cols = list(X_df.select_dtypes(exclude=[np.number]).columns)
    for column in categorical_cols:
        X_df[column], _ = pd.factorize(X_df[column], sort=True)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    combined = X_df.copy()
    combined["__target__"] = y
    combined = combined.dropna()
    X = combined.drop(columns=["__target__"]).to_numpy(dtype=np.float32, copy=True)
    y_arr = combined["__target__"].to_numpy(dtype=np.float32, copy=True)
    return X, y_arr


def build_models(xgb: Any, metagb_class: Any, params: Dict[str, Any], args: argparse.Namespace) -> List[tuple[str, Any]]:
    xgb_common = {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample"]),
        "min_child_weight": max(1, int(round(float(params["min_child_weight"])))),
        "max_bin": int(params["max_bin"]),
        "tree_method": "hist",
        "device": "cpu",
        "objective": "reg:squarederror",
        "random_state": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "verbosity": 0,
    }
    return [
        ("original_metagb", OriginalMetaGBRegressor(metagb_class, params, args)),
        (
            "current_metagb",
            xgb.XGBRegressor(
                **xgb_common,
                enable_metagb=True,
                metagb_pred_colsample_bytree=float(params["pred_feature_fraction"]),
                metagb_score_agg=int(args.metagb_score_agg),
                metagb_lambda_l2=float(args.metagb_lambda_l2),
                base_score=0.0,
            ),
        ),
        ("xgb_cpu", xgb.XGBRegressor(**xgb_common, reg_lambda=0.0)),
    ]


def main() -> int:
    args = parse_args()
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    metagb_class = import_original_metagb(Path(args.original_metagb_dir))
    dataset_dir = Path(args.dataset_dir).expanduser()
    params = {
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample": args.colsample,
        "pred_feature_fraction": args.pred_feature_fraction,
        "min_child_weight": args.min_child_weight,
        "max_bin": args.max_bin,
    }

    rows: List[Dict[str, Any]] = []
    print("params:", json.dumps(params, sort_keys=True))
    print(f"n_jobs for embedded XGBoost models: {args.n_jobs}")
    print("original_metagb is standalone CFFI C code and does not use n_jobs.")
    for dataset_name in args.datasets:
        dataset_path = dataset_dir / dataset_name
        if not dataset_path.exists():
            print(f"[warn] dataset not found, skip: {dataset_path}")
            continue
        X, y = load_dataset(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        print(f"\n[{dataset_name}] rows={X.shape[0]} features={X.shape[1]}")
        for model_name, model in build_models(xgb, metagb_class, params, args):
            fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - fit_start
            pred_start = time.perf_counter()
            pred = model.predict(X_test)
            pred_time = time.perf_counter() - pred_start
            rmse = mean_squared_error(y_test, pred) ** 0.5
            row = {
                "dataset": dataset_name,
                "rows": X.shape[0],
                "features": X.shape[1],
                "model": model_name,
                "train_s": fit_time,
                "predict_s": pred_time,
                "rmse": rmse,
            }
            rows.append(row)
            print(f"  {model_name:16s} train={fit_time:8.3f}s predict={pred_time:7.3f}s rmse={rmse:.6g}")

    if not rows:
        raise RuntimeError("No benchmark rows generated.")

    out = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("\n=== Train time (s) ===")
    train_pivot = out.pivot(index="dataset", columns="model", values="train_s")
    if {"current_metagb", "original_metagb"}.issubset(train_pivot.columns):
        train_pivot["current_vs_original_%"] = (
            (train_pivot["current_metagb"] / train_pivot["original_metagb"] - 1.0) * 100.0
        )
    print(train_pivot.to_string(float_format=lambda value: f"{value:.3f}"))

    print("\n=== RMSE ===")
    print(out.pivot(index="dataset", columns="model", values="rmse").to_string(float_format=lambda value: f"{value:.6g}"))
    print(f"\nCSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
