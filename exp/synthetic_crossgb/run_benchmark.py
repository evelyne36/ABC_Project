from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300.utils.candidates import candidate_config_payload, candidate_sets, load_search_space  # noqa: E402
from random300.utils.metrics import (  # noqa: E402
    classification_metrics,
    primary_metric_direction,
    primary_metric_name,
    primary_metric_value,
    regression_metrics,
)
from random300.utils.parameter_audit import build_parameter_audit  # noqa: E402
from random300.utils.runner import build_estimator, import_local_xgboost  # noqa: E402
from synthetic_crossgb.protocols import Protocol, ScenarioInstance, protocol_by_name  # noqa: E402
from synthetic_crossgb.scenarios import SCENARIOS, generate_instance  # noqa: E402
from synthetic_crossgb.statistics import (  # noqa: E402
    case_type_summary,
    mark_confirmatory,
    paired_statistics,
    seed_level_results,
)


MODELS = ["xgb_cpu", "crossgb_pred0", "crossgb"]


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def index_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rigorous resumable nested-CV CrossGB synthetic benchmark.")
    parser.add_argument(
        "--protocol",
        choices=[
            "smoke",
            "random300_confirmatory",
            "random600_confirmatory",
            "synthetic_main_random300_single_seed",
            "synthetic_main_framework_smoke",
        ],
        default="smoke",
    )
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--cv-seed", type=int, default=314159)
    parser.add_argument("--model-seed", type=int, default=271828)
    parser.add_argument("--task-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "synthetic_crossgb" / "results"))
    parser.add_argument(
        "--xgboost-python-package",
        default=str(REPO_ROOT / "CrossGB" / "xgboost-master" / "python-package"),
    )
    parser.add_argument(
        "--search-space-json",
        default=str(ROOT / "random300" / "configs" / "default_search_space.json"),
    )
    return parser


def runtime_args(args: argparse.Namespace, model_seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        binary_objective="binary:logistic",
        multiclass_objective="multi:softprob",
        seed=model_seed,
        n_jobs=args.n_jobs,
        xgb_reg_lambda=1.0,
        crossgb_lambda_l2=1.0,
        crossgb_hist_subtract=False,
        crossgb_score_agg=0,
        categorical_handling="not_applicable_synthetic",
    )


def protocol_payload(args: argparse.Namespace, protocol: Protocol) -> Dict[str, Any]:
    return {
        "protocol": protocol.to_dict(),
        "models": protocol.models,
        "search_seed": args.search_seed,
        "cv_seed": args.cv_seed,
        "model_seed": args.model_seed,
        "n_jobs": args.n_jobs,
        "xgboost_python_package": str(Path(args.xgboost_python_package).resolve()),
        "search_space_json": str(Path(args.search_space_json).resolve()),
    }


def splits_for(
    task_type: str, X: np.ndarray, y: np.ndarray, outer_folds: int, inner_folds: int, cv_seed: int
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[int, List[Tuple[np.ndarray, np.ndarray]]]]:
    outer_splitter = (
        KFold(n_splits=outer_folds, shuffle=True, random_state=cv_seed)
        if task_type == "regression"
        else StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=cv_seed)
    )
    outer = list(outer_splitter.split(X, None if task_type == "regression" else y))
    inner: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for outer_index, (train_idx, _) in enumerate(outer):
        inner_splitter = (
            KFold(n_splits=inner_folds, shuffle=True, random_state=cv_seed + outer_index + 1)
            if task_type == "regression"
            else StratifiedKFold(
                n_splits=inner_folds, shuffle=True, random_state=cv_seed + outer_index + 1
            )
        )
        inner[outer_index] = list(
            inner_splitter.split(
                X[train_idx], None if task_type == "regression" else y[train_idx]
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
                task_type, X, y, protocol.outer_folds, protocol.inner_folds, cv_seed
            )
            for outer_index, (train_idx, test_idx) in enumerate(outer):
                inner_rows = []
                for inner_index, (inner_train, inner_val) in enumerate(inner[outer_index]):
                    absolute_train = train_idx[inner_train]
                    absolute_val = train_idx[inner_val]
                    if set(absolute_train) & set(absolute_val):
                        raise AssertionError("Inner train/validation overlap")
                    if set(test_idx) & (set(absolute_train) | set(absolute_val)):
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
    return {"cv_seed": cv_seed, "entries": entries, "manifest_hash": canonical_hash(entries)}


def metrics_for(task_type: str, y_true: np.ndarray, pred: np.ndarray, proba: Any) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred, proba, task_type, 2)


def fit_predict(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, task_type: str):
    start = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    fit_sec = time.perf_counter() - start
    start = time.perf_counter()
    if task_type == "regression":
        pred, proba = model.predict(X_eval), None
    else:
        proba = np.asarray(model.predict_proba(X_eval))
        pred = proba.argmax(axis=1)
    predict_sec = time.perf_counter() - start
    return pred, proba, fit_sec, predict_sec


def better(value: float, incumbent: float, direction: str) -> bool:
    return value > incumbent if direction == "maximize" else value < incumbent


def candidate_for_model(candidate: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    out = dict(candidate)
    if model_name == "crossgb_pred0":
        out["pred_feature_fraction"] = 0.0
        out["crossgb_score_agg"] = 0
    return out


def candidate_fairness_audit(candidates_by_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    counts = {model: len(values) for model, values in candidates_by_model.items()}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"Unequal model candidate budgets: {counts}")
    mismatches = []
    shared_keys = [
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "n_estimators",
        "max_bin",
    ]
    pred0_candidates = candidates_by_model.get("crossgb_pred0")
    for index, (xgb, crossgb) in enumerate(
        zip(candidates_by_model["xgb_cpu"], candidates_by_model["crossgb"])
    ):
        for key in shared_keys:
            values = [xgb[key], crossgb[key]]
            if pred0_candidates is not None:
                values.append(pred0_candidates[index][key])
            if len(set(values)) != 1:
                mismatches.append({"backbone_id": xgb.get("backbone_id"), "key": key})
        if xgb["reg_lambda"] != crossgb["crossgb_lambda_l2"]:
            mismatches.append({"backbone_id": xgb.get("backbone_id"), "key": "lambda_l2"})
    combinations = {
        (item["pred_feature_fraction"], item["crossgb_score_agg"])
        for item in candidates_by_model["crossgb"]
    }
    if mismatches:
        raise AssertionError(f"Candidate fairness mismatches: {mismatches[:5]}")
    return {
        "passed": True,
        "candidate_counts": counts,
        "shared_backbone_keys": shared_keys,
        "crossgb_joint_combination_count": len(combinations),
        "crossgb_joint_combinations": sorted([list(item) for item in combinations]),
    }


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(**task["args"])
    scenario = ScenarioInstance(**task["scenario"])
    task_type = SCENARIOS[scenario.base_scenario].task_type
    X, y = generate_instance(
        scenario.base_scenario,
        task["data_seed"],
        scenario.n_samples,
        scenario.noise,
        scenario.extra_nuisance_features,
        scenario.target_features,
    )
    outer, inner = splits_for(
        task_type, X, y, args.outer_folds, args.inner_folds, args.cv_seed
    )
    train_idx, test_idx = outer[task["outer_index"]]
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    model_args = runtime_args(args, args.model_seed + task["data_seed"] + task["outer_index"])
    metric_name = primary_metric_name(task_type)
    direction = primary_metric_direction(task_type)
    candidate_rows: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] | None = None
    best_value = -math.inf if direction == "maximize" else math.inf

    for candidate_index, params in enumerate(task["candidates"], start=1):
        values: List[float] = []
        inner_fit = 0.0
        inner_predict = 0.0
        for inner_train, inner_val in inner[task["outer_index"]]:
            estimator = build_estimator(xgb, task["model"], params, model_args, task_type, 2)
            pred, proba, fit_sec, predict_sec = fit_predict(
                estimator,
                X[train_idx[inner_train]],
                y[train_idx[inner_train]],
                X[train_idx[inner_val]],
                task_type,
            )
            values.append(primary_metric_value(task_type, metrics_for(task_type, y[train_idx[inner_val]], pred, proba)))
            inner_fit += fit_sec
            inner_predict += predict_sec
        mean_value = float(np.mean(values))
        candidate_rows.append(
            {
                "task_id": task["task_id"],
                "scenario": scenario.name,
                "data_seed": task["data_seed"],
                "outer_fold": task["outer_index"] + 1,
                "model": task["model"],
                "candidate_index": candidate_index,
                "candidate_id": params["candidate_id"],
                "backbone_id": params.get("backbone_id", ""),
                "val_primary_metric": mean_value,
                "inner_fit_sec": inner_fit,
                "inner_predict_sec": inner_predict,
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        if best_params is None or better(mean_value, best_value, direction):
            best_params, best_value = params, mean_value

    assert best_params is not None
    estimator = build_estimator(xgb, task["model"], best_params, model_args, task_type, 2)
    pred, proba, fit_sec, predict_sec = fit_predict(estimator, X[train_idx], y[train_idx], X[test_idx], task_type)
    metrics = metrics_for(task_type, y[test_idx], pred, proba)
    fold_row = {
        "task_id": task["task_id"],
        "scenario": scenario.name,
        "base_scenario": scenario.base_scenario,
        "variant": scenario.variant,
        "family": scenario.family,
        "case": scenario.case or scenario.base_scenario,
        "dimension": scenario.dimension or X.shape[1],
        "task_type": task_type,
        "data_seed": task["data_seed"],
        "outer_fold": task["outer_index"] + 1,
        "model": task["model"],
        "primary_metric": metric_name,
        "selected_candidate_id": best_params["candidate_id"],
        "selected_backbone_id": best_params.get("backbone_id", ""),
        "test_primary_metric": primary_metric_value(task_type, metrics),
        "test_rmse": metrics.get("rmse", float("nan")),
        "test_auc": metrics.get("auc", float("nan")),
        "test_logloss": metrics.get("logloss", float("nan")),
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "params_json": json.dumps(best_params, sort_keys=True),
    }
    return {"candidate_rows": candidate_rows, "fold_row": fold_row}


def create_tasks(
    args: argparse.Namespace, protocol: Protocol, candidates_by_model: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    task_args = {
        "outer_folds": protocol.outer_folds,
        "inner_folds": protocol.inner_folds,
        "cv_seed": args.cv_seed,
        "model_seed": args.model_seed,
        "n_jobs": args.n_jobs,
        "xgboost_python_package": str(Path(args.xgboost_python_package).resolve()),
    }
    tasks = []
    for scenario in protocol.scenarios:
        for data_seed in protocol.data_seeds:
            for outer_index in range(protocol.outer_folds):
                for model in protocol.models:
                    task_id = f"{scenario.name}__seed{data_seed}__out{outer_index + 1}__{model}"
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


def collect_shards(shard_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_rows.extend(payload["candidate_rows"])
        fold_rows.append(payload["fold_row"])
    return pd.DataFrame(candidate_rows), pd.DataFrame(fold_rows)


def pending_tasks(tasks: Sequence[Dict[str, Any]], shard_dir: Path) -> List[Dict[str, Any]]:
    return [task for task in tasks if not (shard_dir / f"{task['task_id']}.json").exists()]


def main() -> int:
    args = build_parser().parse_args()
    protocol = protocol_by_name(args.protocol)
    if args.data_seeds:
        protocol = Protocol(
            name=protocol.name,
            scenarios=protocol.scenarios,
            data_seeds=args.data_seeds,
            outer_folds=protocol.outer_folds,
            inner_folds=protocol.inner_folds,
            max_candidates=protocol.max_candidates,
            confirmatory=protocol.confirmatory,
            models=protocol.models,
        )
    args.run_id = args.run_id or f"{protocol.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(args.output_dir).resolve() / args.run_id
    shard_dir = results_dir / "tasks"
    results_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    payload = protocol_payload(args, protocol)
    payload["config_hash"] = canonical_hash(payload)
    config_path = results_dir / "protocol.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["config_hash"] != payload["config_hash"]:
            raise ValueError("Existing run-id has a different protocol config hash")
        if not args.resume:
            raise ValueError("Run directory already exists; use --resume or a new --run-id")
    else:
        write_json(config_path, payload)

    search_space = load_search_space(Path(args.search_space_json))
    candidates_by_model, warnings = candidate_sets(
        search_space, protocol.max_candidates, args.search_seed, None, protocol.models
    )
    write_json(results_dir / "candidate_configs.json", candidate_config_payload(candidates_by_model))
    fairness = candidate_fairness_audit(candidates_by_model)
    write_json(results_dir / "candidate_fairness_audit.json", fairness)
    audit_args = runtime_args(args, args.model_seed)
    audit = build_parameter_audit(audit_args, search_space, warnings)
    audit["candidate_counts"] = {model: len(values) for model, values in candidates_by_model.items()}
    audit["candidate_config_hash"] = canonical_hash(candidates_by_model)
    write_json(results_dir / "parameter_audit.json", audit)
    write_json(results_dir / "split_manifest.json", build_split_manifest(protocol, args.cv_seed))

    tasks = create_tasks(args, protocol, candidates_by_model)
    pending = pending_tasks(tasks, shard_dir)
    print(f"run_id={args.run_id} total_tasks={len(tasks)} pending_tasks={len(pending)}", flush=True)
    start = time.perf_counter()
    execution_mode = "serial"
    if not pending:
        execution_mode = "no_pending_tasks"
    elif args.task_workers <= 1:
        for index, task in enumerate(pending, start=1):
            write_json(shard_dir / f"{task['task_id']}.json", run_task(task))
            print(f"[{index}/{len(pending)}] {task['task_id']}", flush=True)
    else:
        execution_mode = f"process_pool:{args.task_workers}"
        try:
            with ProcessPoolExecutor(max_workers=min(args.task_workers, len(pending) or 1)) as executor:
                future_map = {executor.submit(run_task, task): task for task in pending}
                for index, future in enumerate(as_completed(future_map), start=1):
                    task = future_map[future]
                    write_json(shard_dir / f"{task['task_id']}.json", future.result())
                    print(f"[{index}/{len(pending)}] {task['task_id']}", flush=True)
        except PermissionError as exc:
            execution_mode = "serial_fallback"
            print(
                "WARNING: process pool unavailable; falling back to serial execution "
                f"({exc}).",
                flush=True,
            )
            remaining = [task for task in pending if not (shard_dir / f"{task['task_id']}.json").exists()]
            for index, task in enumerate(remaining, start=1):
                write_json(shard_dir / f"{task['task_id']}.json", run_task(task))
                print(f"[fallback {index}/{len(remaining)}] {task['task_id']}", flush=True)

    candidates_df, folds_df = collect_shards(shard_dir)
    if len(folds_df) != len(tasks):
        raise AssertionError(f"Expected {len(tasks)} fold rows, found {len(folds_df)}")
    expected_candidates = len(tasks) * protocol.max_candidates
    if len(candidates_df) != expected_candidates:
        raise AssertionError(f"Expected {expected_candidates} candidate rows, found {len(candidates_df)}")
    seed_df = seed_level_results(folds_df)
    stats_df = mark_confirmatory(paired_statistics(seed_df), protocol.confirmatory)
    case_summary_df, type_summary_df = case_type_summary(stats_df)
    complexity_rows = []
    for row in folds_df.to_dict(orient="records"):
        params = json.loads(row["params_json"])
        complexity_rows.append(
            {
                "scenario": row["scenario"],
                "family": row.get("family", "legacy"),
                "case": row.get("case", row["base_scenario"]),
                "dimension": row.get("dimension", 0),
                "data_seed": row["data_seed"],
                "outer_fold": row["outer_fold"],
                "model": row["model"],
                "max_depth": params.get("max_depth"),
                "n_estimators": params.get("n_estimators"),
                "pred_feature_fraction": params.get("pred_feature_fraction"),
                "crossgb_score_agg": params.get("crossgb_score_agg"),
                "fit_sec": row["fit_sec"],
                "predict_sec": row["predict_sec"],
                "test_primary_metric": row["test_primary_metric"],
            }
        )
    complexity_df = pd.DataFrame(complexity_rows)
    candidates_df.to_csv(results_dir / "candidates.csv", index=False)
    folds_df.to_csv(results_dir / "folds.csv", index=False)
    seed_df.to_csv(results_dir / "seed_results.csv", index=False)
    stats_df.to_csv(results_dir / "paired_statistics.csv", index=False)
    case_summary_df.to_csv(results_dir / "case_summary.csv", index=False)
    type_summary_df.to_csv(results_dir / "type_summary.csv", index=False)
    complexity_df.to_csv(results_dir / "selected_complexity.csv", index=False)
    elapsed_this_invocation = time.perf_counter() - start
    metadata_path = results_dir / "metadata.json"
    previous_elapsed = 0.0
    if metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_elapsed = float(
            previous_metadata.get(
                "elapsed_cumulative_sec",
                previous_metadata.get("elapsed_this_invocation_sec", 0.0),
            )
        )
    metadata = {
        "run_id": args.run_id,
        "protocol": protocol.name,
        "confirmatory": protocol.confirmatory,
        "config_hash": payload["config_hash"],
        "candidate_config_hash": audit["candidate_config_hash"],
        "split_manifest_hash": json.loads((results_dir / "split_manifest.json").read_text())["manifest_hash"],
        "task_count": len(tasks),
        "candidate_row_count": len(candidates_df),
        "elapsed_this_invocation_sec": elapsed_this_invocation,
        "elapsed_cumulative_sec": previous_elapsed + elapsed_this_invocation,
        "requested_task_workers": args.task_workers,
        "execution_mode": execution_mode,
        "completed": True,
    }
    write_json(metadata_path, metadata)
    write_json(
        results_dir / "completeness_audit.json",
        {
            "passed": True,
            "expected_task_count": len(tasks),
            "actual_task_count": len(folds_df),
            "expected_candidate_row_count": expected_candidates,
            "actual_candidate_row_count": len(candidates_df),
            "unique_task_ids": int(folds_df["task_id"].nunique()),
            "models": sorted(folds_df["model"].unique().tolist()),
            "scenario_count": int(folds_df["scenario"].nunique()),
            "data_seed_count": int(folds_df["data_seed"].nunique()),
        },
    )
    print(stats_df.to_string(index=False))
    print(f"results_dir={results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
