from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300_baselinemodel.utils.candidates import (  # noqa: E402
    candidate_config_payload,
    candidate_sets,
    load_search_space,
)
from random300_baselinemodel.utils.model_factory import build_estimator  # noqa: E402
from synthetic_baselinemodel.protocols import (  # noqa: E402
    BASELINE_MODELS,
    Protocol,
    protocol_by_name,
)
from synthetic_crossgb.scenarios import SCENARIOS, generate_instance  # noqa: E402
from tabular_extra_random300.utils.metrics import (  # noqa: E402
    classification_metrics,
    labels_from_proba,
    normalize_proba,
    primary_metric_direction,
    primary_metric_name,
    primary_metric_value,
    regression_metrics,
)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def index_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def implementation_hash() -> str:
    paths = [
        Path(__file__),
        ROOT / "synthetic_baselinemodel" / "protocols.py",
        ROOT / "synthetic_crossgb" / "protocols.py",
        ROOT / "synthetic_crossgb" / "scenarios.py",
        ROOT / "random300_baselinemodel" / "utils" / "candidates.py",
        ROOT / "random300_baselinemodel" / "utils" / "model_factory.py",
        ROOT / "tabular_extra_random300" / "utils" / "metrics.py",
    ]
    payload = {
        str(path.resolve().relative_to(ROOT.resolve())): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    return canonical_hash(payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def default_task_workers() -> int:
    return max(1, min(15, (os.cpu_count() or 1) - 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable synthetic-data baseline benchmark with Random300 nested CV."
    )
    parser.add_argument(
        "--protocol",
        choices=[
            "synthetic_baseline_framework_smoke",
            "synthetic_baseline_random300_single_seed",
        ],
        default="synthetic_baseline_framework_smoke",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=None,
        help="Optional subset; the full protocol uses all three frozen tree baselines.",
    )
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--cv-seed", type=int, default=314159)
    parser.add_argument("--model-seed", type=int, default=271828)
    parser.add_argument("--task-workers", type=int, default=default_task_workers())
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override samples per scenario for capacity calibration only.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "synthetic_baselinemodel" / "results"),
    )
    parser.add_argument(
        "--search-space-json",
        default=str(
            ROOT
            / "synthetic_baselinemodel"
            / "configs"
            / "default_search_space.json"
        ),
    )
    return parser


def normalize_protocol(args: argparse.Namespace) -> Protocol:
    protocol = protocol_by_name(args.protocol)
    scenarios = protocol.scenarios
    if args.scenario_limit is not None:
        if args.scenario_limit <= 0:
            raise ValueError("--scenario-limit must be positive")
        scenarios = scenarios[: args.scenario_limit]
    if args.n_samples is not None:
        if args.n_samples < 20:
            raise ValueError("--n-samples must be at least 20")
        scenarios = [replace(item, n_samples=args.n_samples) for item in scenarios]
    models = list(args.models) if args.models else protocol.models
    data_seeds = list(args.data_seeds) if args.data_seeds else protocol.data_seeds
    if len(models) != len(set(models)):
        raise ValueError("--models must not contain duplicates")
    if len(data_seeds) != len(set(data_seeds)):
        raise ValueError("--data-seeds must not contain duplicates")
    outer_folds = (
        protocol.outer_folds if args.outer_folds is None else args.outer_folds
    )
    inner_folds = (
        protocol.inner_folds if args.inner_folds is None else args.inner_folds
    )
    max_candidates = (
        protocol.max_candidates
        if args.max_candidates is None
        else args.max_candidates
    )
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("Nested CV requires at least two outer and inner folds")
    if max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    if args.task_workers <= 0 or args.n_jobs <= 0:
        raise ValueError("--task-workers and --n-jobs must be positive")
    return Protocol(
        name=protocol.name,
        scenarios=scenarios,
        data_seeds=data_seeds,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        max_candidates=max_candidates,
        models=models,
    )


def runtime_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(seed=seed, n_jobs=args.n_jobs)


def protocol_payload(args: argparse.Namespace, protocol: Protocol) -> Dict[str, Any]:
    return {
        "protocol": protocol.to_dict(),
        "search_seed": args.search_seed,
        "cv_seed": args.cv_seed,
        "model_seed": args.model_seed,
        "n_jobs": args.n_jobs,
        "search_space_json": str(Path(args.search_space_json).resolve()),
    }


def splits_for(
    task_type: str,
    X: np.ndarray,
    y: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    cv_seed: int,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
]:
    outer_splitter = (
        KFold(n_splits=outer_folds, shuffle=True, random_state=cv_seed)
        if task_type == "regression"
        else StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=cv_seed
        )
    )
    outer = list(
        outer_splitter.split(X, None if task_type == "regression" else y)
    )
    inner: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for outer_index, (train_idx, _) in enumerate(outer):
        inner_splitter = (
            KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=cv_seed + outer_index + 1,
            )
            if task_type == "regression"
            else StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=cv_seed + outer_index + 1,
            )
        )
        inner[outer_index] = list(
            inner_splitter.split(
                X[train_idx],
                None if task_type == "regression" else y[train_idx],
            )
        )
    return outer, inner


def build_split_manifest(protocol: Protocol, cv_seed: int) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for scenario in protocol.scenarios:
        task_type = SCENARIOS[scenario.base_scenario].task_type
        for data_seed in protocol.data_seeds:
            X, y = generate_instance(
                scenario.base_scenario,
                data_seed,
                scenario.n_samples,
                scenario.noise,
                scenario.extra_nuisance_features,
                scenario.target_features,
            )
            outer, inner = splits_for(
                task_type,
                X,
                y,
                protocol.outer_folds,
                protocol.inner_folds,
                cv_seed,
            )
            for outer_index, (train_idx, test_idx) in enumerate(outer):
                inner_rows = []
                for inner_index, (inner_train, inner_val) in enumerate(
                    inner[outer_index]
                ):
                    absolute_train = train_idx[inner_train]
                    absolute_val = train_idx[inner_val]
                    if set(absolute_train) & set(absolute_val):
                        raise AssertionError("Inner train/validation overlap")
                    if set(test_idx) & (
                        set(absolute_train) | set(absolute_val)
                    ):
                        raise AssertionError("Outer test leakage into inner CV")
                    inner_rows.append(
                        {
                            "inner_fold": inner_index + 1,
                            "train_hash": index_hash(absolute_train),
                            "validation_hash": index_hash(absolute_val),
                        }
                    )
                entries.append(
                    {
                        "scenario": scenario.name,
                        "data_seed": data_seed,
                        "outer_fold": outer_index + 1,
                        "train_hash": index_hash(train_idx),
                        "test_hash": index_hash(test_idx),
                        "inner": inner_rows,
                    }
                )
    return {
        "cv_seed": cv_seed,
        "entries": entries,
        "manifest_hash": canonical_hash(entries),
    }


def search_space_audit(
    search_space: Dict[str, Any],
    candidates_by_model: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    missing = [model for model in candidates_by_model if model not in search_space]
    if missing:
        raise ValueError(f"Models missing from search space: {missing}")
    counts = {model: len(values) for model, values in candidates_by_model.items()}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"Unequal candidate budgets: {counts}")

    lgb = search_space["lightgbm"]["space"]
    cat = search_space["catboost"]["space"]
    shared_checks = {
        "learning_rate": lgb["learning_rate"] == cat["learning_rate"],
        "tree_count": lgb["n_estimators"] == cat["iterations"],
        "depth": lgb["max_depth"] == cat["depth"],
        "l2_regularization": lgb["reg_lambda"] == cat["l2_leaf_reg"],
    }
    if not all(shared_checks.values()):
        raise AssertionError(f"Shared boosting space mismatch: {shared_checks}")
    return {
        "passed": True,
        "candidate_counts": counts,
        "selected_models": list(candidates_by_model),
        "shared_boosting_space_checks": shared_checks,
        "search_space_hash": canonical_hash(search_space),
    }


def evaluate_metrics(
    task_type: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: Any,
    n_classes: int,
) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred, proba, task_type, n_classes)


def fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    start = time.perf_counter()
    try:
        model.fit(X_train, y_train, verbose=False)
    except (TypeError, ValueError) as exc:
        if "verbose" not in str(exc):
            raise
        model.fit(X_train, y_train)
    return time.perf_counter() - start


def predict_model(
    model: Any,
    X_eval: np.ndarray,
    task_type: str,
    n_classes: int,
) -> Tuple[np.ndarray, Any, float]:
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"X does not have valid feature names, but LGBM.* "
                r"was fitted with feature names"
            ),
            category=UserWarning,
        )
        if task_type == "regression":
            pred, proba = np.asarray(model.predict(X_eval)), None
        else:
            try:
                proba = normalize_proba(
                    model.predict_proba(X_eval), n_classes
                )
                pred = (
                    labels_from_proba(proba)
                    if proba is not None
                    else np.asarray(model.predict(X_eval))
                )
            except Exception:
                proba = None
                pred = np.asarray(model.predict(X_eval))
    predict_sec = time.perf_counter() - start
    return pred, proba, predict_sec


def _better(value: float, incumbent: float, direction: str) -> bool:
    if not math.isfinite(value):
        return False
    return value > incumbent if direction == "maximize" else value < incumbent


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(**task["args"])
    scenario = task["scenario"]
    task_type = SCENARIOS[scenario["base_scenario"]].task_type
    X, y = generate_instance(
        scenario["base_scenario"],
        task["data_seed"],
        scenario["n_samples"],
        scenario["noise"],
        scenario["extra_nuisance_features"],
        scenario["target_features"],
    )
    n_classes = 1 if task_type == "regression" else int(len(np.unique(y)))
    outer, inner = splits_for(
        task_type,
        X,
        y,
        args.outer_folds,
        args.inner_folds,
        args.cv_seed,
    )
    train_idx, test_idx = outer[task["outer_index"]]
    model_args = runtime_args(
        args,
        args.model_seed + task["data_seed"] + task["outer_index"],
    )
    metric_name = primary_metric_name(task_type)
    direction = primary_metric_direction(task_type)
    best_value = -math.inf if direction == "maximize" else math.inf
    best_params: Dict[str, Any] | None = None
    best_inner_fit = float("nan")
    best_inner_predict = float("nan")
    candidate_rows: List[Dict[str, Any]] = []
    total_inner_fit = 0.0
    total_inner_predict = 0.0
    inner_fits_attempted = 0
    inner_fits_succeeded = 0
    task_start = time.perf_counter()

    for candidate_index, params in enumerate(task["candidates"], start=1):
        values: List[float] = []
        inner_fit = 0.0
        inner_predict = 0.0
        candidate_fits_attempted = 0
        candidate_fits_succeeded = 0
        status = "ok"
        for inner_train, inner_val in inner[task["outer_index"]]:
            try:
                estimator = build_estimator(
                    task["model"],
                    params,
                    model_args,
                    task_type,
                    n_classes,
                )
                inner_fits_attempted += 1
                candidate_fits_attempted += 1
                fit_sec = fit_model(
                    estimator,
                    X[train_idx[inner_train]],
                    y[train_idx[inner_train]],
                )
                inner_fits_succeeded += 1
                candidate_fits_succeeded += 1
                inner_fit += fit_sec
                pred, proba, predict_sec = predict_model(
                    estimator,
                    X[train_idx[inner_val]],
                    task_type,
                    n_classes,
                )
                inner_predict += predict_sec
                metrics = evaluate_metrics(
                    task_type,
                    y[train_idx[inner_val]],
                    pred,
                    proba,
                    n_classes,
                )
                values.append(primary_metric_value(task_type, metrics))
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
                values = []
                break
        total_inner_fit += inner_fit
        total_inner_predict += inner_predict
        mean_value = float(np.mean(values)) if values else float("nan")
        candidate_rows.append(
            {
                "task_id": task["task_id"],
                "scenario": scenario["name"],
                "family": scenario["family"],
                "case": scenario["case"],
                "dimension": scenario["dimension"],
                "data_seed": task["data_seed"],
                "outer_fold": task["outer_index"] + 1,
                "model": task["model"],
                "candidate_index": candidate_index,
                "candidate_id": params["candidate_id"],
                "status": status,
                "primary_metric": metric_name,
                "val_primary_metric": mean_value,
                "inner_fit_sec": inner_fit,
                "inner_predict_sec": inner_predict,
                "inner_fits_attempted": candidate_fits_attempted,
                "inner_fits_succeeded": candidate_fits_succeeded,
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        if best_params is None:
            if math.isfinite(mean_value):
                best_params = params
                best_value = mean_value
                best_inner_fit = inner_fit
                best_inner_predict = inner_predict
        elif _better(mean_value, best_value, direction):
            best_params = params
            best_value = mean_value
            best_inner_fit = inner_fit
            best_inner_predict = inner_predict

    fold_base = {
        "task_id": task["task_id"],
        "scenario": scenario["name"],
        "base_scenario": scenario["base_scenario"],
        "variant": scenario["variant"],
        "family": scenario["family"],
        "case": scenario["case"],
        "dimension": scenario["dimension"],
        "n_samples": scenario["n_samples"],
        "task_type": task_type,
        "data_seed": task["data_seed"],
        "outer_fold": task["outer_index"] + 1,
        "model": task["model"],
        "primary_metric": metric_name,
        "candidate_count": len(task["candidates"]),
        "total_inner_fit_sec": total_inner_fit,
        "total_inner_predict_sec": total_inner_predict,
        "inner_fits_attempted": inner_fits_attempted,
        "inner_fits_succeeded": inner_fits_succeeded,
    }
    if best_params is None:
        fold_row = {
            **fold_base,
            "status": "no_valid_candidate",
            "selected_candidate_id": "",
            "validation_primary_metric": float("nan"),
            "test_primary_metric": float("nan"),
            "test_rmse": float("nan"),
            "test_auc": float("nan"),
            "test_logloss": float("nan"),
            "test_accuracy": float("nan"),
            "test_balanced_accuracy": float("nan"),
            "selected_inner_fit_sec": float("nan"),
            "selected_inner_predict_sec": float("nan"),
            "final_fit_sec": float("nan"),
            "final_predict_sec": float("nan"),
            "final_fits_attempted": 0,
            "final_fits_succeeded": 0,
            "task_elapsed_sec": time.perf_counter() - task_start,
            "params_json": "{}",
        }
        return {"candidate_rows": candidate_rows, "fold_row": fold_row}

    final_fits_attempted = 0
    final_fits_succeeded = 0
    fit_sec = float("nan")
    predict_sec = float("nan")
    try:
        estimator = build_estimator(
            task["model"],
            best_params,
            model_args,
            task_type,
            n_classes,
        )
        final_fits_attempted = 1
        fit_sec = fit_model(
            estimator,
            X[train_idx],
            y[train_idx],
        )
        final_fits_succeeded = 1
        pred, proba, predict_sec = predict_model(
            estimator,
            X[test_idx],
            task_type,
            n_classes,
        )
        metrics = evaluate_metrics(task_type, y[test_idx], pred, proba, n_classes)
        status = "ok"
    except Exception as exc:
        metrics = {}
        status = f"failed:{type(exc).__name__}:{exc}"

    fold_row = {
        **fold_base,
        "status": status,
        "selected_candidate_id": best_params["candidate_id"],
        "validation_primary_metric": best_value,
        "test_primary_metric": (
            primary_metric_value(task_type, metrics)
            if metrics
            else float("nan")
        ),
        "test_rmse": metrics.get("rmse", float("nan")),
        "test_auc": metrics.get("auc", float("nan")),
        "test_logloss": metrics.get("logloss", float("nan")),
        "test_accuracy": metrics.get("accuracy", float("nan")),
        "test_balanced_accuracy": metrics.get(
            "balanced_accuracy", float("nan")
        ),
        "selected_inner_fit_sec": best_inner_fit,
        "selected_inner_predict_sec": best_inner_predict,
        "final_fit_sec": fit_sec,
        "final_predict_sec": predict_sec,
        "final_fits_attempted": final_fits_attempted,
        "final_fits_succeeded": final_fits_succeeded,
        "task_elapsed_sec": time.perf_counter() - task_start,
        "params_json": json.dumps(best_params, sort_keys=True),
    }
    return {"candidate_rows": candidate_rows, "fold_row": fold_row}


def create_tasks(
    args: argparse.Namespace,
    protocol: Protocol,
    candidates_by_model: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    task_args = {
        "outer_folds": protocol.outer_folds,
        "inner_folds": protocol.inner_folds,
        "cv_seed": args.cv_seed,
        "model_seed": args.model_seed,
        "n_jobs": args.n_jobs,
    }
    tasks: List[Dict[str, Any]] = []
    for model in protocol.models:
        for scenario in protocol.scenarios:
            for data_seed in protocol.data_seeds:
                for outer_index in range(protocol.outer_folds):
                    task_id = (
                        f"{scenario.name}__seed{data_seed}"
                        f"__out{outer_index + 1}__{model}"
                    )
                    tasks.append(
                        {
                            "task_id": task_id,
                            "scenario": scenario.__dict__,
                            "data_seed": data_seed,
                            "outer_index": outer_index,
                            "model": model,
                            "candidates": candidates_by_model[model],
                            "args": task_args,
                        }
                    )
    return tasks


def pending_tasks(
    tasks: Sequence[Dict[str, Any]],
    shard_dir: Path,
    config_hash: str | None = None,
) -> List[Dict[str, Any]]:
    pending = []
    for task in tasks:
        path = shard_dir / f"{task['task_id']}.json"
        if not path.exists():
            pending.append(task)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fold_row = payload["fold_row"]
            valid = (
                isinstance(payload["candidate_rows"], list)
                and len(payload["candidate_rows"]) == len(task["candidates"])
                and fold_row["task_id"] == task["task_id"]
                and fold_row["status"] == "ok"
                and (
                    config_hash is None
                    or payload.get("config_hash") == config_hash
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            pending.append(task)
    return pending


def collect_shards(shard_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_rows.extend(payload["candidate_rows"])
        fold_rows.append(payload["fold_row"])
    return pd.DataFrame(candidate_rows), pd.DataFrame(fold_rows)


def aggregate_folds(folds_df: pd.DataFrame) -> pd.DataFrame:
    if folds_df.empty:
        return pd.DataFrame()
    metric_columns = [
        "validation_primary_metric",
        "test_primary_metric",
        "test_rmse",
        "test_auc",
        "test_logloss",
        "test_accuracy",
        "test_balanced_accuracy",
        "task_elapsed_sec",
        "total_inner_fit_sec",
        "final_fit_sec",
    ]
    grouped = (
        folds_df.groupby(
            [
                "model",
                "scenario",
                "family",
                "case",
                "dimension",
                "task_type",
            ],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(item) for item in column if item).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns.to_flat_index()
    ]
    return grouped


def model_summary(folds_df: pd.DataFrame) -> pd.DataFrame:
    if folds_df.empty:
        return pd.DataFrame()
    rows = []
    for model, group in folds_df.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "task_count": int(len(group)),
                "successful_tasks": int((group["status"] == "ok").sum()),
                "failed_tasks": int((group["status"] != "ok").sum()),
                "task_elapsed_sec_sum": float(group["task_elapsed_sec"].sum()),
                "task_elapsed_sec_mean": float(group["task_elapsed_sec"].mean()),
                "total_inner_fit_sec_sum": float(
                    group["total_inner_fit_sec"].sum()
                ),
                "final_fit_sec_sum": float(group["final_fit_sec"].sum()),
                "inner_fits_attempted": int(
                    group["inner_fits_attempted"].sum()
                ),
                "inner_fits_succeeded": int(
                    group["inner_fits_succeeded"].sum()
                ),
                "final_fits_attempted": int(
                    group["final_fits_attempted"].sum()
                ),
                "final_fits_succeeded": int(
                    group["final_fits_succeeded"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()
    protocol = normalize_protocol(args)
    if args.n_jobs != 1 and args.task_workers > 1:
        print(
            "WARNING: --n-jobs > 1 with multiple task workers can oversubscribe CPU.",
            flush=True,
        )
    args.run_id = args.run_id or (
        f"{protocol.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir = Path(args.output_dir).resolve() / args.run_id
    shard_dir = results_dir / "tasks"
    results_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    search_space = load_search_space(Path(args.search_space_json))
    candidates_by_model = candidate_sets(
        protocol.models,
        protocol.max_candidates,
        args.search_seed,
        search_space,
    )
    audit = search_space_audit(search_space, candidates_by_model)
    candidate_config_hash = canonical_hash(candidates_by_model)
    payload = protocol_payload(args, protocol)
    payload["search_space_hash"] = audit["search_space_hash"]
    payload["candidate_config_hash"] = candidate_config_hash
    payload["implementation_hash"] = implementation_hash()
    payload["config_hash"] = canonical_hash(payload)
    config_path = results_dir / "protocol.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["config_hash"] != payload["config_hash"]:
            raise ValueError(
                "Existing run-id has a different protocol config hash"
            )
        if not args.resume:
            raise ValueError(
                "Run directory already exists; use --resume or a new --run-id"
            )
    else:
        write_json(config_path, payload)

    write_json(
        results_dir / "candidate_configs.json",
        candidate_config_payload(candidates_by_model),
    )
    audit["candidate_config_hash"] = candidate_config_hash
    write_json(results_dir / "parameter_audit.json", audit)
    write_json(
        results_dir / "split_manifest.json",
        build_split_manifest(protocol, args.cv_seed),
    )

    tasks = create_tasks(args, protocol, candidates_by_model)
    pending = pending_tasks(tasks, shard_dir, payload["config_hash"])
    print(
        f"run_id={args.run_id} total_tasks={len(tasks)} "
        f"pending_tasks={len(pending)}",
        flush=True,
    )
    start = time.perf_counter()
    execution_mode = "serial"
    if not pending:
        execution_mode = "no_pending_tasks"
    elif args.task_workers <= 1:
        for index, task in enumerate(pending, start=1):
            result = run_task(task)
            result["config_hash"] = payload["config_hash"]
            write_json(shard_dir / f"{task['task_id']}.json", result)
            print(f"[{index}/{len(pending)}] {task['task_id']}", flush=True)
    else:
        execution_mode = f"process_pool:{args.task_workers}"
        with ProcessPoolExecutor(
            max_workers=min(args.task_workers, len(pending))
        ) as executor:
            future_map = {
                executor.submit(run_task, task): task for task in pending
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                result = future.result()
                result["config_hash"] = payload["config_hash"]
                write_json(shard_dir / f"{task['task_id']}.json", result)
                print(
                    f"[{index}/{len(pending)}] {task['task_id']}",
                    flush=True,
                )

    candidates_df, folds_df = collect_shards(shard_dir)
    if len(folds_df) != len(tasks):
        raise AssertionError(
            f"Expected {len(tasks)} fold rows, found {len(folds_df)}"
        )
    expected_candidates = len(tasks) * protocol.max_candidates
    if len(candidates_df) != expected_candidates:
        raise AssertionError(
            f"Expected {expected_candidates} candidate rows, "
            f"found {len(candidates_df)}"
        )

    candidates_df.to_csv(results_dir / "candidates.csv", index=False)
    folds_df.to_csv(results_dir / "folds.csv", index=False)
    aggregate_folds(folds_df).to_csv(
        results_dir / "aggregate.csv", index=False
    )
    model_summary(folds_df).to_csv(
        results_dir / "model_summary.csv", index=False
    )
    elapsed_this_invocation = time.perf_counter() - start
    metadata_path = results_dir / "metadata.json"
    previous_elapsed = 0.0
    if metadata_path.exists():
        previous_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        previous_elapsed = float(
            previous_metadata.get("elapsed_cumulative_sec", 0.0)
        )
    split_manifest = json.loads(
        (results_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "run_id": args.run_id,
        "protocol": protocol.name,
        "config_hash": payload["config_hash"],
        "search_space_hash": audit["search_space_hash"],
        "candidate_config_hash": candidate_config_hash,
        "implementation_hash": payload["implementation_hash"],
        "split_manifest_hash": split_manifest["manifest_hash"],
        "task_count": len(tasks),
        "candidate_row_count": len(candidates_df),
        "planned_model_fit_count": protocol.to_dict()[
            "planned_model_fit_count"
        ],
        "actual_model_fits_attempted": int(
            folds_df["inner_fits_attempted"].sum()
            + folds_df["final_fits_attempted"].sum()
        ),
        "actual_model_fits_succeeded": int(
            folds_df["inner_fits_succeeded"].sum()
            + folds_df["final_fits_succeeded"].sum()
        ),
        "successful_tasks": int((folds_df["status"] == "ok").sum()),
        "failed_tasks": int((folds_df["status"] != "ok").sum()),
        "elapsed_this_invocation_sec": elapsed_this_invocation,
        "elapsed_cumulative_sec": previous_elapsed + elapsed_this_invocation,
        "requested_task_workers": args.task_workers,
        "n_jobs": args.n_jobs,
        "execution_mode": execution_mode,
        "completed": int((folds_df["status"] != "ok").sum()) == 0,
    }
    write_json(metadata_path, metadata)
    failed_tasks = int((folds_df["status"] != "ok").sum())
    write_json(
        results_dir / "completeness_audit.json",
        {
            "passed": failed_tasks == 0,
            "structural_completeness_passed": True,
            "expected_task_count": len(tasks),
            "actual_task_count": len(folds_df),
            "expected_candidate_row_count": expected_candidates,
            "actual_candidate_row_count": len(candidates_df),
            "unique_task_ids": int(folds_df["task_id"].nunique()),
            "models": sorted(folds_df["model"].unique().tolist()),
            "scenario_count": int(folds_df["scenario"].nunique()),
            "data_seed_count": int(folds_df["data_seed"].nunique()),
            "failed_tasks": failed_tasks,
        },
    )
    print(model_summary(folds_df).to_string(index=False))
    print(f"results_dir={results_dir}")
    return 0 if failed_tasks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
