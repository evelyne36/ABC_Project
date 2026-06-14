from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300.utils.metrics import (  # noqa: E402
    classification_metrics,
    primary_metric_direction,
    primary_metric_name,
    primary_metric_value,
    regression_metrics,
)
from random300.utils.runner import build_estimator, import_local_xgboost  # noqa: E402
from synthetic_metagb.scenarios import SCENARIOS, scenario_names  # noqa: E402


MODELS = ["xgb_cpu", "current_metagb_pred0", "current_metagb"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random300-style nested-CV benchmark on MetaGB synthetic scenarios.")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--scenarios", nargs="+", choices=scenario_names(), default=scenario_names())
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--outer-folds", type=int, default=0)
    parser.add_argument("--inner-folds", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--xgboost-python-package",
        default=str(ROOT / "beforereview" / "xgboost-master" / "python-package"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "synthetic_metagb" / "results"))
    parser.add_argument("--run-id", default=None)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "smoke": (400, 2, 2),
        "quick": (1000, 3, 2),
        "full": (3000, 5, 3),
    }
    n_samples, outer_folds, inner_folds = defaults[args.mode]
    args.n_samples = args.n_samples or n_samples
    args.outer_folds = args.outer_folds or outer_folds
    args.inner_folds = args.inner_folds or inner_folds
    args.run_id = args.run_id or f"synthetic_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.results_dir = str(Path(args.output_dir).expanduser().resolve() / args.run_id)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    args.binary_objective = "binary:logistic"
    args.multiclass_objective = "multi:softprob"
    args.xgb_reg_lambda = 1.0
    args.metagb_lambda_l2 = 1.0
    args.metagb_hist_subtract = False
    args.metagb_score_agg = 0
    args.metagb_pred_value_type = "raw"
    return args


def paired_backbones(mode: str) -> List[Dict[str, Any]]:
    configs = [
        (0.10, 1, 30, 1.0),
        (0.05, 2, 60, 1.0),
        (0.05, 3, 100, 1.0),
        (0.03, 4, 160, 0.8),
        (0.02, 5, 240, 0.8),
    ]
    limit = {"smoke": 2, "quick": 3, "full": 5}[mode]
    return [
        {
            "candidate_id": f"paired_{index:02d}",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": 1.0,
            "max_bin": 255,
            "reg_lambda": 1.0,
            "metagb_lambda_l2": 1.0,
            "pred_feature_fraction": 1.0,
            "metagb_score_agg": 0,
        }
        for index, (learning_rate, max_depth, n_estimators, subsample) in enumerate(configs[:limit], start=1)
    ]


def split_indices(task_type: str, X: np.ndarray, y: np.ndarray, folds: int, seed: int):
    if task_type == "regression":
        return list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(X))
    return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y))


def metrics_for(task_type: str, y_true: np.ndarray, pred: np.ndarray, proba: Any) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred, proba, task_type, 2)


def fit_predict(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, task_type: str):
    start = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    fit_sec = time.perf_counter() - start
    start = time.perf_counter()
    if task_type == "regression":
        pred = model.predict(X_test)
        proba = None
    else:
        proba = np.asarray(model.predict_proba(X_test))
        pred = proba.argmax(axis=1)
    predict_sec = time.perf_counter() - start
    return pred, proba, fit_sec, predict_sec


def better(value: float, incumbent: float, direction: str) -> bool:
    return value > incumbent if direction == "maximize" else value < incumbent


def candidate_for_model(candidate: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    out = dict(candidate)
    if model_name == "current_metagb_pred0":
        out["pred_feature_fraction"] = 0.0
    return out


def run_one(
    xgb: Any,
    args: argparse.Namespace,
    scenario_name: str,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scenario = SCENARIOS[scenario_name]
    direction = primary_metric_direction(scenario.task_type)
    metric_name = primary_metric_name(scenario.task_type)
    outer_splits = split_indices(scenario.task_type, X, y, args.outer_folds, args.seed)
    candidate_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_splits = split_indices(scenario.task_type, X_train, y_train, args.inner_folds, args.seed + outer_fold)
        best_candidate = None
        best_value = -math.inf if direction == "maximize" else math.inf

        for base_candidate in candidates:
            candidate = candidate_for_model(base_candidate, model_name)
            values = []
            fit_total = 0.0
            predict_total = 0.0
            for inner_train_idx, val_idx in inner_splits:
                estimator = build_estimator(xgb, model_name, candidate, args, scenario.task_type, 2)
                pred, proba, fit_sec, predict_sec = fit_predict(
                    estimator,
                    X_train[inner_train_idx],
                    y_train[inner_train_idx],
                    X_train[val_idx],
                    scenario.task_type,
                )
                metrics = metrics_for(scenario.task_type, y_train[val_idx], pred, proba)
                values.append(primary_metric_value(scenario.task_type, metrics))
                fit_total += fit_sec
                predict_total += predict_sec
            mean_value = float(np.mean(values))
            candidate_rows.append(
                {
                    "scenario": scenario_name,
                    "task_type": scenario.task_type,
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "candidate_id": candidate["candidate_id"],
                    "val_primary_metric": mean_value,
                    "primary_metric": metric_name,
                    "inner_fit_sec": fit_total,
                    "inner_predict_sec": predict_total,
                    "params_json": json.dumps(candidate, sort_keys=True),
                }
            )
            if best_candidate is None or better(mean_value, best_value, direction):
                best_candidate = candidate
                best_value = mean_value

        assert best_candidate is not None
        estimator = build_estimator(xgb, model_name, best_candidate, args, scenario.task_type, 2)
        pred, proba, fit_sec, predict_sec = fit_predict(estimator, X_train, y_train, X_test, scenario.task_type)
        metrics = metrics_for(scenario.task_type, y_test, pred, proba)
        fold_rows.append(
            {
                "scenario": scenario_name,
                "task_type": scenario.task_type,
                "hypothesis": scenario.hypothesis,
                "expected_winner": scenario.expected_winner,
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_candidate_id": best_candidate["candidate_id"],
                "primary_metric": metric_name,
                "test_primary_metric": primary_metric_value(scenario.task_type, metrics),
                "test_rmse": metrics.get("rmse", float("nan")),
                "test_auc": metrics.get("auc", float("nan")),
                "test_logloss": metrics.get("logloss", float("nan")),
                "fit_sec": fit_sec,
                "predict_sec": predict_sec,
                "params_json": json.dumps(best_candidate, sort_keys=True),
            }
        )
    return candidate_rows, fold_rows


def aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        folds.groupby(["scenario", "task_type", "hypothesis", "expected_winner", "model", "primary_metric"], as_index=False)
        .agg(
            mean_test_primary_metric=("test_primary_metric", "mean"),
            std_test_primary_metric=("test_primary_metric", "std"),
            mean_fit_sec=("fit_sec", "mean"),
        )
    )
    pivot = grouped.pivot(index="scenario", columns="model", values="mean_test_primary_metric")
    rows = []
    for _, row in grouped.iterrows():
        scenario = row["scenario"]
        xgb_value = float(pivot.loc[scenario, "xgb_cpu"]) if "xgb_cpu" in pivot.columns else float("nan")
        metagb_value = float(pivot.loc[scenario, "current_metagb"]) if "current_metagb" in pivot.columns else float("nan")
        direction = primary_metric_direction(row["task_type"])
        if math.isfinite(xgb_value) and math.isfinite(metagb_value):
            signed_advantage = xgb_value - metagb_value if direction == "minimize" else metagb_value - xgb_value
        else:
            signed_advantage = float("nan")
        item = row.to_dict()
        item["metagb_signed_advantage_vs_xgb"] = signed_advantage
        rows.append(item)
    return pd.DataFrame(rows)


def comparison_table(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    values = aggregate_df.pivot(index="scenario", columns="model", values="mean_test_primary_metric")
    details = aggregate_df.drop_duplicates("scenario").set_index("scenario")
    rows = []
    for scenario in values.index:
        task_type = str(details.loc[scenario, "task_type"])
        direction = primary_metric_direction(task_type)
        xgb_value = float(values.loc[scenario, "xgb_cpu"]) if "xgb_cpu" in values.columns else float("nan")
        pred0_value = (
            float(values.loc[scenario, "current_metagb_pred0"])
            if "current_metagb_pred0" in values.columns
            else float("nan")
        )
        metagb_value = (
            float(values.loc[scenario, "current_metagb"])
            if "current_metagb" in values.columns
            else float("nan")
        )

        def signed_advantage(reference: float) -> float:
            if not math.isfinite(reference) or not math.isfinite(metagb_value):
                return float("nan")
            return reference - metagb_value if direction == "minimize" else metagb_value - reference

        rows.append(
            {
                "scenario": scenario,
                "task_type": task_type,
                "primary_metric": details.loc[scenario, "primary_metric"],
                "hypothesis": details.loc[scenario, "hypothesis"],
                "expected_winner": details.loc[scenario, "expected_winner"],
                "xgb_cpu": xgb_value,
                "current_metagb_pred0": pred0_value,
                "current_metagb": metagb_value,
                "metagb_signed_advantage_vs_xgb": signed_advantage(xgb_value),
                "metagb_signed_advantage_vs_pred0": signed_advantage(pred0_value),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    candidates = paired_backbones(args.mode)
    candidate_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for scenario_index, scenario_name in enumerate(args.scenarios):
        scenario = SCENARIOS[scenario_name]
        noise = 0.15 if scenario.task_type == "regression" else 0.05
        X, y = scenario.generator(np.random.default_rng(args.seed + scenario_index * 1000), args.n_samples, noise)
        print(f"scenario={scenario_name} task={scenario.task_type} rows={len(y)} features={X.shape[1]}", flush=True)
        for model_name in args.models:
            print(f"  model={model_name}", flush=True)
            candidates_out, folds_out = run_one(xgb, args, scenario_name, model_name, X, y, candidates)
            candidate_rows.extend(candidates_out)
            fold_rows.extend(folds_out)

    results_dir = Path(args.results_dir)
    candidates_df = pd.DataFrame(candidate_rows)
    folds_df = pd.DataFrame(fold_rows)
    aggregate_df = aggregate(folds_df)
    comparison_df = comparison_table(aggregate_df)
    candidates_df.to_csv(results_dir / "candidates.csv", index=False)
    folds_df.to_csv(results_dir / "folds.csv", index=False)
    aggregate_df.to_csv(results_dir / "aggregate.csv", index=False)
    comparison_df.to_csv(results_dir / "comparison.csv", index=False)
    metadata = {
        "run_id": args.run_id,
        "mode": args.mode,
        "scenarios": args.scenarios,
        "models": args.models,
        "n_samples": args.n_samples,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "candidate_count_per_model": len(candidates),
        "elapsed_sec": time.perf_counter() - start,
        "results_dir": str(results_dir),
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison_df.to_string(index=False))
    print(f"results_dir={results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
