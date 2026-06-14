from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("METAGB_PROJECT_ROOT", REPO_ROOT)).expanduser().resolve()
XGBOOST_PYTHON_PACKAGE = Path(
    os.environ.get("METAGB_XGBOOST_PYTHON_PACKAGE", PROJECT_ROOT / "xgboost-master" / "python-package")
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("METAGB_OUTPUT_DIR", PROJECT_ROOT / "results_current_metagb_vs_xgbcpu")
).expanduser()


def default_dataset_dirs() -> List[Path]:
    raw_dirs = os.environ.get("METAGB_DATASET_DIRS")
    if raw_dirs:
        return [Path(item).expanduser() for item in raw_dirs.split(os.pathsep) if item]
    return [
        PROJECT_ROOT / "Exp1_54CV_ordinal" / "dataset" / "tabArena_regression",
        PROJECT_ROOT / "dataset" / "tabArena_regression",
    ]


DEFAULT_DATASET_DIRS = default_dataset_dirs()

EXP1_DATASETS = [
    "airfoil_self_noise.csv",
    "Another-Dataset-on-used-Fiat-500.csv",
    "concrete_compressive_strength.csv",
    "diamonds.csv",
    "Food_Delivery_Time.csv",
    "healthcare_insurance_expenses.csv",
    "houses.csv",
    "miami_housing.csv",
    "physiochemical_protein.csv",
    "QSAR_fish_toxicity.csv",
    "wine_quality.csv",
]

np = None
pd = None
KFold = None


def ensure_runtime_dependencies() -> None:
    global np, pd, KFold
    if np is not None and pd is not None and KFold is not None:
        return
    import numpy as numpy_module
    import pandas as pandas_module
    from sklearn.model_selection import KFold as kfold_class

    np = numpy_module
    pd = pandas_module
    KFold = kfold_class


def import_local_xgboost(python_package: Path):
    python_package = python_package.resolve()
    if not python_package.exists():
        raise FileNotFoundError(f"Local XGBoost python package not found: {python_package}")
    for module_name in list(sys.modules):
        if module_name == "xgboost" or module_name.startswith("xgboost."):
            del sys.modules[module_name]
    if str(python_package) not in sys.path:
        sys.path.insert(0, str(python_package))
    import xgboost as xgb

    return xgb


def default_cpu_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def sample_log_uniform(rng: Any, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_log_uniform_int(rng: Any, low: int, high: int) -> int:
    sampled = np.exp(rng.uniform(np.log(low), np.log(high)))
    return int(np.clip(np.round(sampled), low, high))


def attach_n_estimators(candidates: List[Dict[str, Any]], rng: Any) -> None:
    values = [200, 400, 600, 800]
    assigned = np.tile(values, int(np.ceil(len(candidates) / len(values))))[: len(candidates)]
    rng.shuffle(assigned)
    for params, n_estimators in zip(candidates, assigned):
        params["n_estimators"] = int(n_estimators)


def generate_exp1_style_candidates(num_candidates: int, seed: int) -> List[Dict[str, Any]]:
    ensure_runtime_dependencies()
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    rng = np.random.default_rng(seed + 3000)
    pred_feature_classes = [round(i * 0.1, 1) for i in range(1, 11)]
    candidates: List[Dict[str, Any]] = []

    per_class = max(1, num_candidates // len(pred_feature_classes))
    for pred_feature_fraction in pred_feature_classes:
        for _ in range(per_class):
            if len(candidates) >= num_candidates:
                break
            candidates.append(
                {
                    "pred_feature_fraction": float(pred_feature_fraction),
                    "learning_rate": sample_log_uniform(rng, 0.005, 0.1),
                    "max_depth": sample_log_uniform_int(rng, 4, 10),
                    "min_child_weight": sample_log_uniform(rng, 0.001, 5.0),
                    "subsample": float(rng.uniform(0.6, 1.0)),
                    "colsample": float(rng.uniform(0.6, 1.0)),
                    "version": 3,
                    "max_bin": 255,
                }
            )

    while len(candidates) < num_candidates:
        pred_feature_fraction = float(rng.choice(pred_feature_classes))
        candidates.append(
            {
                "pred_feature_fraction": pred_feature_fraction,
                "learning_rate": sample_log_uniform(rng, 0.005, 0.1),
                "max_depth": sample_log_uniform_int(rng, 4, 10),
                "min_child_weight": sample_log_uniform(rng, 0.001, 5.0),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample": float(rng.uniform(0.6, 1.0)),
                "version": 3,
                "max_bin": 255,
            }
        )

    attach_n_estimators(candidates, np.random.default_rng(seed + 4000))
    return candidates


def candidates_for_model(model_name: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    base_candidates = generate_exp1_style_candidates(args.max_candidates, args.seed)
    if model_name != "current_metagb" or args.pred_feature_fractions is None:
        return base_candidates

    expanded: List[Dict[str, Any]] = []
    for base_index, base_params in enumerate(base_candidates, start=1):
        for pred_feature_fraction in args.pred_feature_fractions:
            params = dict(base_params)
            params["grid_base_candidate_index"] = base_index
            params["pred_feature_fraction"] = float(pred_feature_fraction)
            expanded.append(params)
    return expanded


def find_dataset_path(dataset_name: str, dataset_dirs: Sequence[Path]) -> Path:
    for dataset_dir in dataset_dirs:
        candidate = dataset_dir / dataset_name
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in dataset_dirs)
    raise FileNotFoundError(f"Dataset not found: {dataset_name}. Searched: {searched}")


def load_dataset(csv_path: Path, categorical_handling: str) -> Tuple[Any, Any]:
    ensure_runtime_dependencies()
    df = pd.read_csv(csv_path)
    if categorical_handling == "drop":
        df = df.select_dtypes(include=[np.number]).dropna()
    elif categorical_handling == "onehot":
        categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df = df.dropna()
    elif categorical_handling == "ordinal":
        categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
        for column in categorical_cols:
            df[column], _ = pd.factorize(df[column], sort=True)
        df = df.dropna()
    else:
        raise ValueError("categorical_handling must be one of: drop, onehot, ordinal")

    if len(df) == 0:
        raise ValueError(f"No rows remain after preprocessing: {csv_path}")

    X = df.iloc[:, :-1].to_numpy(dtype=np.float32, copy=True)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32, copy=True)
    return X, y


def build_estimator(xgb: Any, model_name: str, params: Dict[str, Any], args: argparse.Namespace):
    common = {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample"]),
        "min_child_weight": max(1, int(round(float(params["min_child_weight"])))),
        "max_bin": int(params.get("max_bin", 255)),
        "tree_method": "hist",
        "device": "cpu",
        "objective": "reg:squarederror",
        "random_state": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "verbosity": 0,
    }
    if model_name == "xgb_cpu":
        common["reg_lambda"] = float(args.xgb_reg_lambda)
        return xgb.XGBRegressor(**common)

    if model_name == "current_metagb":
        common.update(
            {
                "enable_metagb": True,
                "metagb_pred_colsample_bytree": float(params["pred_feature_fraction"]),
                "metagb_cond_pred_share": bool(args.metagb_cond_pred_share),
                "metagb_leaf_only_use_bias": bool(args.metagb_leaf_only_use_bias),
                "metagb_split_sample": int(args.metagb_split_sample),
                "metagb_score_agg": int(args.metagb_score_agg),
                "metagb_lambda_l2": float(args.metagb_lambda_l2),
                "base_score": 0.0,
            }
        )
        return xgb.XGBRegressor(**common)

    raise ValueError(f"Unknown model: {model_name}")


def rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_predict_timed(model: Any, X_train: Any, y_train: Any, X_eval: Any) -> Tuple[Any, float, float]:
    fit_start = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    fit_time = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    pred = model.predict(X_eval)
    predict_time = time.perf_counter() - predict_start
    return pred, fit_time, predict_time


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("inf"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(np.mean(finite)), float(np.std(finite))


@dataclass
class CandidateRow:
    dataset: str
    model: str
    outer_fold: int
    candidate_index: int
    selected: bool
    status: str
    val_rmse_mean: float
    val_rmse_std: float
    inner_train_time_sec: float
    inner_predict_time_sec: float
    final_train_time_sec: float
    final_predict_time_sec: float
    test_rmse: float
    params_json: str


def selected_outer_fold_indices(args: argparse.Namespace) -> List[int]:
    if args.run_all_outer_folds or args.mode == "full":
        return list(range(args.outer_folds))
    if args.outer_fold_index < 0 or args.outer_fold_index >= args.outer_folds:
        raise ValueError("--outer-fold-index is zero-based and must be in [0, outer_folds)")
    return [args.outer_fold_index]


def run_benchmark(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_runtime_dependencies()
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    dataset_dirs = [Path(path).expanduser() for path in args.dataset_dirs]
    datasets = EXP1_DATASETS if args.datasets == ["all"] else args.datasets
    rows: List[CandidateRow] = []
    summary_rows: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        dataset_path = find_dataset_path(dataset_name, dataset_dirs)
        X, y = load_dataset(dataset_path, args.categorical_handling)
        outer_kf = KFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        outer_splits = list(outer_kf.split(X))

        for outer_index in selected_outer_fold_indices(args):
            train_index, test_index = outer_splits[outer_index]
            X_train_outer, X_test_outer = X[train_index], X[test_index]
            y_train_outer, y_test_outer = y[train_index], y[test_index]
            inner_kf = KFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_index)
            inner_splits = list(inner_kf.split(X_train_outer))

            for model_name in args.models:
                candidates = candidates_for_model(model_name, args)
                fold_start = time.perf_counter()
                print(
                    f"[{model_name}] dataset={dataset_name}, outer_fold={outer_index + 1}/{args.outer_folds}, "
                    f"candidates={len(candidates)}",
                    flush=True,
                )
                candidate_scores: List[Tuple[float, float, float, float, int, str]] = []

                for candidate_index, params in enumerate(candidates, start=1):
                    val_rmses: List[float] = []
                    inner_train_time = 0.0
                    inner_predict_time = 0.0
                    status = "ok"

                    for inner_train_idx, inner_val_idx in inner_splits:
                        X_inner_train = X_train_outer[inner_train_idx]
                        y_inner_train = y_train_outer[inner_train_idx]
                        X_inner_val = X_train_outer[inner_val_idx]
                        y_inner_val = y_train_outer[inner_val_idx]
                        try:
                            estimator = build_estimator(xgb, model_name, params, args)
                            pred_val, fit_time, pred_time = fit_predict_timed(
                                estimator, X_inner_train, y_inner_train, X_inner_val
                            )
                            val_rmses.append(rmse(y_inner_val, pred_val))
                            inner_train_time += fit_time
                            inner_predict_time += pred_time
                        except Exception as exc:  # noqa: BLE001
                            status = f"failed:{type(exc).__name__}"
                            val_rmses = [float("inf")]
                            inner_train_time = 0.0
                            inner_predict_time = 0.0
                            break

                    val_rmse_mean, val_rmse_std = mean_std(val_rmses)
                    candidate_scores.append(
                        (
                            val_rmse_mean,
                            val_rmse_std,
                            inner_train_time,
                            inner_predict_time,
                            candidate_index,
                            status,
                        )
                    )
                    rows.append(
                        CandidateRow(
                            dataset=dataset_name,
                            model=model_name,
                            outer_fold=outer_index + 1,
                            candidate_index=candidate_index,
                            selected=False,
                            status=status,
                            val_rmse_mean=val_rmse_mean,
                            val_rmse_std=val_rmse_std,
                            inner_train_time_sec=inner_train_time,
                            inner_predict_time_sec=inner_predict_time,
                            final_train_time_sec=float("nan"),
                            final_predict_time_sec=float("nan"),
                            test_rmse=float("nan"),
                            params_json=json.dumps(params, sort_keys=True),
                        )
                    )
                    if (
                        candidate_index == 1
                        or candidate_index == len(candidates)
                        or candidate_index % args.progress_every == 0
                        or status != "ok"
                    ):
                        print(
                            f"  candidate {candidate_index}/{len(candidates)}: "
                            f"val_rmse={val_rmse_mean:.6g}, status={status}",
                            flush=True,
                        )

                valid_scores = [score for score in candidate_scores if math.isfinite(score[0])]
                if not valid_scores:
                    summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "outer_fold": outer_index + 1,
                            "status": "no_valid_candidate",
                        }
                    )
                    continue

                valid_scores.sort(key=lambda item: (item[0], item[2] + item[3], item[4]))
                best_val_mean, best_val_std, best_inner_train, best_inner_predict, best_index, _ = valid_scores[0]
                best_params = candidates[best_index - 1]

                try:
                    final_estimator = build_estimator(xgb, model_name, best_params, args)
                    test_pred, final_train_time, final_predict_time = fit_predict_timed(
                        final_estimator, X_train_outer, y_train_outer, X_test_outer
                    )
                    test_rmse = rmse(y_test_outer, test_pred)
                    final_status = "ok"
                except Exception as exc:  # noqa: BLE001
                    final_train_time = float("nan")
                    final_predict_time = float("nan")
                    test_rmse = float("nan")
                    final_status = f"failed:{type(exc).__name__}"

                for row in reversed(rows):
                    if (
                        row.dataset == dataset_name
                        and row.model == model_name
                        and row.outer_fold == outer_index + 1
                        and row.candidate_index == best_index
                    ):
                        row.selected = True
                        row.final_train_time_sec = final_train_time
                        row.final_predict_time_sec = final_predict_time
                        row.test_rmse = test_rmse
                        row.status = final_status
                        break

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "outer_fold": outer_index + 1,
                        "outer_folds": args.outer_folds,
                        "inner_folds": args.inner_folds,
                        "selected_candidate_index": best_index,
                        "val_rmse_mean": best_val_mean,
                        "val_rmse_std": best_val_std,
                        "inner_train_time_sec": best_inner_train,
                        "inner_predict_time_sec": best_inner_predict,
                        "final_train_time_sec": final_train_time,
                        "final_predict_time_sec": final_predict_time,
                        "test_rmse": test_rmse,
                        "total_train_time_sec": best_inner_train + final_train_time,
                        "total_predict_time_sec": best_inner_predict + final_predict_time,
                        "status": final_status,
                        "params_json": json.dumps(best_params, sort_keys=True),
                    }
                )
                print(
                    f"[{model_name}] done dataset={dataset_name}, outer_fold={outer_index + 1}, "
                    f"best={best_index}, val_rmse={best_val_mean:.6g}, test_rmse={test_rmse:.6g}, "
                    f"elapsed={time.perf_counter() - fold_start:.1f}s",
                    flush=True,
                )

    return [asdict(row) for row in rows], summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not summary_rows:
        return []
    ensure_runtime_dependencies()
    df = pd.DataFrame(summary_rows)
    metric_cols = [
        "val_rmse_mean",
        "test_rmse",
        "inner_train_time_sec",
        "inner_predict_time_sec",
        "final_train_time_sec",
        "final_predict_time_sec",
        "total_train_time_sec",
        "total_predict_time_sec",
    ]
    available = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(["dataset", "model"], as_index=False)[available].agg(["mean", "std"])
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    return grouped.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-CV speed/RMSE comparison for current embedded MetaGB and CPU XGBoost."
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--datasets", nargs="+", default=["airfoil_self_noise.csv"])
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_DATASET_DIRS],
        help="Directories containing the Exp1 tabArena_regression csv files.",
    )
    parser.add_argument("--xgboost-python-package", default=str(XGBOOST_PYTHON_PACKAGE))
    parser.add_argument("--models", nargs="+", default=["current_metagb", "xgb_cpu"])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--outer-fold-index", type=int, default=0, help="zero-based outer fold for quick mode")
    parser.add_argument("--run-all-outer-folds", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means quick=5, full=200")
    parser.add_argument("--categorical-handling", choices=["drop", "onehot", "ordinal"], default="ordinal")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=default_cpu_jobs())
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--xgb-reg-lambda", type=float, default=0.0)
    parser.add_argument("--metagb-lambda-l2", type=float, default=1.0)
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--metagb-split-sample", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--pred-feature-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Explicit current_metagb pred_feature_fraction grid, e.g. 0.0 0.1 ... 1.0.",
    )
    parser.add_argument("--metagb-cond-pred-share", action="store_true")
    parser.add_argument("--metagb-leaf-only-use-bias", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.datasets == ["all"]:
        args.run_all_outer_folds = args.mode == "full" or args.run_all_outer_folds
    if args.mode == "full":
        args.run_all_outer_folds = True
    if args.max_candidates <= 0:
        args.max_candidates = 5 if args.mode == "quick" else 200

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows, summary_rows = run_benchmark(args)
    aggregate_rows = build_aggregate(summary_rows)

    candidate_csv = output_dir / "current_metagb_vs_xgbcpu_nested_candidates.csv"
    summary_csv = output_dir / "current_metagb_vs_xgbcpu_nested_summary.csv"
    aggregate_csv = output_dir / "current_metagb_vs_xgbcpu_nested_aggregate.csv"
    metadata_json = output_dir / "current_metagb_vs_xgbcpu_nested_metadata.json"

    write_csv(candidate_csv, candidate_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(aggregate_csv, aggregate_rows)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": args.mode,
                "datasets": EXP1_DATASETS if args.datasets == ["all"] else args.datasets,
                "models": args.models,
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "run_all_outer_folds": bool(args.run_all_outer_folds),
                "max_candidates": args.max_candidates,
                "pred_feature_fractions": args.pred_feature_fractions,
                "categorical_handling": args.categorical_handling,
                "xgboost_python_package": args.xgboost_python_package,
                "candidate_csv": str(candidate_csv),
                "summary_csv": str(summary_csv),
                "aggregate_csv": str(aggregate_csv),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"candidate rows: {len(candidate_rows)}")
    print(f"summary rows: {len(summary_rows)}")
    print(f"aggregate rows: {len(aggregate_rows)}")
    print(f"outputs: {candidate_csv}, {summary_csv}, {aggregate_csv}, {metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
