from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from exp12.utils.hyperparams import DEFAULT_DATASETS, RANDOM_SEED, candidates_for_experiment


np = None
pd = None
KFold = None


class SimpleKFold:
    """Small fallback for sklearn.model_selection.KFold."""

    def __init__(self, n_splits: int, shuffle: bool = False, random_state: Optional[int] = None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, X: Any):
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot be greater than the number of samples")
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_index = indices[start:stop]
            train_index = np.concatenate((indices[:start], indices[stop:]))
            yield train_index, test_index
            current = stop


def ensure_runtime_dependencies() -> None:
    global np, pd, KFold
    if np is not None and pd is not None and KFold is not None:
        return
    import numpy as numpy_module
    import pandas as pandas_module

    try:
        from sklearn.model_selection import KFold as kfold_class
    except ModuleNotFoundError:
        kfold_class = SimpleKFold

    np = numpy_module
    pd = pandas_module
    KFold = kfold_class


def project_root() -> Path:
    raw = os.environ.get("METAGB_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def default_xgboost_python_package(root: Path) -> Path:
    raw = os.environ.get("METAGB_XGBOOST_PYTHON_PACKAGE")
    if raw:
        return Path(raw).expanduser().resolve()
    return root / "xgboost-master" / "python-package"


def default_dataset_dir(root: Path) -> Path:
    raw = os.environ.get("METAGB_DATASET_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return root / "dataset" / "tabArena_regression"


def default_output_root(root: Path) -> Path:
    raw = os.environ.get("METAGB_OUTPUT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return root / "exp12"


def default_cpu_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


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


def setup_logger(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_dir / f"{run_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def write_json_log(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def append_task_log(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def select_datasets(requested: Sequence[str], mode: str) -> List[str]:
    if list(requested) == ["all"]:
        return list(DEFAULT_DATASETS if mode == "full" else DEFAULT_DATASETS[:1])
    return list(requested)


def find_dataset_path(dataset_name: str, dataset_dir: Path) -> Path:
    path = dataset_dir / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def load_dataset(csv_path: Path, categorical_handling: str) -> Tuple[Any, Any]:
    ensure_runtime_dependencies()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must contain at least one feature and one target: {csv_path}")

    X_df = df.iloc[:, :-1].copy()
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")

    if categorical_handling == "drop":
        X_df = X_df.select_dtypes(include=[np.number])
    elif categorical_handling == "onehot":
        categorical_cols = list(X_df.select_dtypes(exclude=[np.number]).columns)
        if categorical_cols:
            X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    elif categorical_handling == "ordinal":
        categorical_cols = list(X_df.select_dtypes(exclude=[np.number]).columns)
        for column in categorical_cols:
            X_df[column], _ = pd.factorize(X_df[column], sort=True)
    else:
        raise ValueError("categorical_handling must be one of: drop, onehot, ordinal")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    combined = X_df.copy()
    combined["__target__"] = y
    combined = combined.dropna()
    if combined.empty:
        raise ValueError(f"No rows remain after preprocessing: {csv_path}")
    if combined.shape[1] < 2:
        raise ValueError(f"No usable feature columns remain after preprocessing: {csv_path}")

    X = combined.drop(columns=["__target__"]).to_numpy(dtype=np.float32, copy=True)
    y_arr = combined["__target__"].to_numpy(dtype=np.float32, copy=True)
    return X, y_arr


def build_estimator(xgb: Any, model_name: str, params: Dict[str, Any], args: argparse.Namespace) -> Any:
    common = {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": max(1, int(round(float(params.get("min_child_weight", params.get("min_data_in_leaf", 1)))))),
        "max_bin": int(params.get("max_bin", 255)),
        "tree_method": "hist",
        "device": "cpu",
        "objective": "reg:squarederror",
        "random_state": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "verbosity": 0,
    }
    if model_name == "xgb_cpu":
        common["reg_lambda"] = float(params.get("reg_lambda", args.xgb_reg_lambda))
        return xgb.XGBRegressor(**common)

    if model_name == "current_metagb":
        common.update(
            {
                "enable_metagb": True,
                "metagb_pred_colsample_bytree": float(params.get("pred_feature_fraction", 1.0)),
                "metagb_cond_pred_share": bool(args.metagb_cond_pred_share),
                "metagb_leaf_only_use_bias": bool(args.metagb_leaf_only_use_bias),
                "metagb_hist_subtract": bool(args.metagb_hist_subtract),
                "metagb_score_agg": int(args.metagb_score_agg),
                "metagb_lambda_l2": float(args.metagb_lambda_l2),
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

    pred_start = time.perf_counter()
    pred = model.predict(X_eval)
    pred_time = time.perf_counter() - pred_start
    return pred, fit_time, pred_time


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("inf"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(np.mean(finite)), float(np.std(finite))


def selected_outer_fold_indices(args: argparse.Namespace) -> List[int]:
    if args.run_all_outer_folds or args.mode == "full":
        return list(range(args.outer_folds))
    if args.outer_fold_index < 0 or args.outer_fold_index >= args.outer_folds:
        raise ValueError("--outer-fold-index must be zero-based and in [0, outer_folds)")
    return [args.outer_fold_index]


@dataclass
class CandidateRow:
    run_name: str
    experiment: str
    categorical_handling: str
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


def run_benchmark(
    args: argparse.Namespace,
    run_name: str,
    experiment: str,
    categorical_handling: str,
    logger: logging.Logger,
    logs_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_runtime_dependencies()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    datasets = select_datasets(args.datasets, args.mode)
    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("Datasets: %s", ", ".join(datasets))
    logger.info("Models: %s", ", ".join(args.models))
    logger.info("Task workers: %d", args.task_workers)
    logger.info("Model n_jobs per task: %d", args.n_jobs)
    if args.n_jobs != 1:
        logger.warning("n_jobs=%d can oversubscribe CPU when task_workers > 1", args.n_jobs)

    tasks: List[Dict[str, Any]] = []
    task_log_dir = logs_dir / "tasks"
    task_index = 0
    for dataset_name in datasets:
        dataset_path = find_dataset_path(dataset_name, dataset_dir)
        for outer_index in selected_outer_fold_indices(args):
            for model_name in args.models:
                task_index += 1
                task_id = f"{safe_name(dataset_name)}__{model_name}__out{outer_index + 1}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "task_index": task_index,
                        "run_name": run_name,
                        "experiment": experiment,
                        "categorical_handling": categorical_handling,
                        "dataset": dataset_name,
                        "dataset_path": str(dataset_path),
                        "model": model_name,
                        "outer_index": outer_index,
                        "outer_folds": args.outer_folds,
                        "inner_folds": args.inner_folds,
                        "max_candidates": args.max_candidates,
                        "limit_grid_candidates": args.limit_grid_candidates,
                        "seed": args.seed,
                        "n_jobs": args.n_jobs,
                        "progress_every": args.progress_every,
                        "xgboost_python_package": args.xgboost_python_package,
                        "xgb_reg_lambda": args.xgb_reg_lambda,
                        "metagb_lambda_l2": args.metagb_lambda_l2,
                        "metagb_score_agg": args.metagb_score_agg,
                        "metagb_cond_pred_share": args.metagb_cond_pred_share,
                        "metagb_leaf_only_use_bias": args.metagb_leaf_only_use_bias,
                        "metagb_hist_subtract": args.metagb_hist_subtract,
                        "project_root": str(project_root()),
                        "task_log_path": str(task_log_dir / f"{task_id}.log"),
                    }
                )

    if not tasks:
        return [], []

    logger.info("Scheduled tasks: %d", len(tasks))
    worker_count = min(max(1, args.task_workers), len(tasks))
    candidate_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    status_path = logs_dir / "task_status.json"
    task_status: Dict[str, Dict[str, Any]] = {
        task["task_id"]: {
            "task_id": task["task_id"],
            "task_index": task["task_index"],
            "dataset": task["dataset"],
            "model": task["model"],
            "outer_fold": task["outer_index"] + 1,
            "status": "pending",
            "log": task["task_log_path"],
        }
        for task in tasks
    }
    write_json_log(status_path, {"run_name": run_name, "tasks": task_status})

    def record_task_result(
        finished: int,
        task: Dict[str, Any],
        task_candidate_rows: List[Dict[str, Any]],
        task_summary: Dict[str, Any],
    ) -> None:
        candidate_rows.extend(task_candidate_rows)
        summary_rows.append(task_summary)
        task_status[task["task_id"]].update(
            {
                "status": task_summary.get("status", "done"),
                "candidate_count": task_summary.get("candidate_count"),
                "selected_candidate_index": task_summary.get("selected_candidate_index"),
                "test_rmse": task_summary.get("test_rmse"),
                "task_elapsed_sec": task_summary.get("task_elapsed_sec"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json_log(
            status_path,
            {"run_name": run_name, "finished": finished, "total": len(tasks), "tasks": task_status},
        )

    def record_task_failure(finished: int, task: Dict[str, Any], exc: Exception) -> None:
        task_status[task["task_id"]].update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json_log(
            status_path,
            {"run_name": run_name, "finished": finished, "total": len(tasks), "tasks": task_status},
        )
        summary_rows.append(
            {
                "run_name": run_name,
                "experiment": experiment,
                "categorical_handling": categorical_handling,
                "dataset": task["dataset"],
                "model": task["model"],
                "outer_fold": task["outer_index"] + 1,
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "status": f"failed:{type(exc).__name__}:{exc}",
            }
        )

    if worker_count == 1:
        for task in tasks:
            task_status[task["task_id"]]["status"] = "submitted"
        write_json_log(status_path, {"run_name": run_name, "tasks": task_status})
        for finished, task in enumerate(tasks, start=1):
            label = f"{task['dataset']} | {task['model']} | out{task['outer_index'] + 1}"
            try:
                task_candidate_rows, task_summary = run_outer_task(task)
                record_task_result(finished, task, task_candidate_rows, task_summary)
                logger.info("[%d/%d] done %s status=%s", finished, len(tasks), label, task_summary.get("status"))
            except Exception as exc:
                logger.exception("[%d/%d] failed %s: %s", finished, len(tasks), label, exc)
                record_task_failure(finished, task, exc)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(run_outer_task, task): task for task in tasks}
            for task in tasks:
                task_status[task["task_id"]]["status"] = "submitted"
            write_json_log(status_path, {"run_name": run_name, "tasks": task_status})
            for finished, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                label = f"{task['dataset']} | {task['model']} | out{task['outer_index'] + 1}"
                try:
                    task_candidate_rows, task_summary = future.result()
                    record_task_result(finished, task, task_candidate_rows, task_summary)
                    logger.info("[%d/%d] done %s status=%s", finished, len(tasks), label, task_summary.get("status"))
                except Exception as exc:
                    logger.exception("[%d/%d] failed %s: %s", finished, len(tasks), label, exc)
                    record_task_failure(finished, task, exc)

    candidate_rows.sort(key=lambda row: (row["dataset"], row["model"], row["outer_fold"], row["candidate_index"]))
    summary_rows.sort(key=lambda row: (row["dataset"], row["model"], row["outer_fold"]))
    return candidate_rows, summary_rows


def run_outer_task(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ensure_runtime_dependencies()
    task_log_path = Path(task["task_log_path"])
    append_task_log(
        task_log_path,
        f"start task={task['task_id']} dataset={task['dataset']} model={task['model']} "
        f"outer_fold={task['outer_index'] + 1}/{task['outer_folds']} n_jobs={task['n_jobs']}",
    )
    xgb = import_local_xgboost(Path(task["xgboost_python_package"]))
    args = argparse.Namespace(**task)
    X, y = load_dataset(Path(task["dataset_path"]), task["categorical_handling"])
    outer_kf = KFold(n_splits=task["outer_folds"], shuffle=True, random_state=task["seed"])
    outer_splits = list(outer_kf.split(X))
    train_index, test_index = outer_splits[task["outer_index"]]
    X_train_outer, X_test_outer = X[train_index], X[test_index]
    y_train_outer, y_test_outer = y[train_index], y[test_index]
    inner_kf = KFold(
        n_splits=task["inner_folds"],
        shuffle=True,
        random_state=task["seed"] + task["outer_index"],
    )
    inner_splits = list(inner_kf.split(X_train_outer))
    candidates = candidates_for_experiment(
        np,
        task["experiment"],
        task["model"],
        task["dataset"],
        task["max_candidates"],
        task["seed"],
        Path(task["project_root"]),
        task["limit_grid_candidates"],
    )
    append_task_log(
        task_log_path,
        f"candidate_count={len(candidates)} search_type={'random' if task['experiment'] == 'exp1' else 'grid'}",
    )

    candidate_rows: List[CandidateRow] = []
    candidate_scores: List[Tuple[float, float, float, float, int, str]] = []
    task_start = time.perf_counter()

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
                estimator = build_estimator(xgb, task["model"], params, args)
                pred_val, fit_time, pred_time = fit_predict_timed(
                    estimator,
                    X_inner_train,
                    y_inner_train,
                    X_inner_val,
                )
                val_rmses.append(rmse(y_inner_val, pred_val))
                inner_train_time += fit_time
                inner_predict_time += pred_time
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
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
        candidate_rows.append(
            CandidateRow(
                run_name=task["run_name"],
                experiment=task["experiment"],
                categorical_handling=task["categorical_handling"],
                dataset=task["dataset"],
                model=task["model"],
                outer_fold=task["outer_index"] + 1,
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
            or candidate_index % max(1, task["progress_every"]) == 0
            or status != "ok"
        ):
            append_task_log(
                task_log_path,
                f"candidate {candidate_index}/{len(candidates)} val_rmse={val_rmse_mean:.6g} "
                f"status={status}",
            )

    valid_scores = [score for score in candidate_scores if math.isfinite(score[0])]
    if not valid_scores:
        append_task_log(task_log_path, "no valid candidate")
        return [asdict(row) for row in candidate_rows], {
            "run_name": task["run_name"],
            "experiment": task["experiment"],
            "categorical_handling": task["categorical_handling"],
            "dataset": task["dataset"],
            "model": task["model"],
            "outer_fold": task["outer_index"] + 1,
            "outer_folds": task["outer_folds"],
            "inner_folds": task["inner_folds"],
            "candidate_count": len(candidates),
            "status": "no_valid_candidate",
        }

    valid_scores.sort(key=lambda item: (item[0], item[2] + item[3], item[4]))
    best_val_mean, best_val_std, best_inner_train, best_inner_predict, best_index, _ = valid_scores[0]
    best_params = candidates[best_index - 1]
    append_task_log(
        task_log_path,
        f"best candidate={best_index} val_rmse={best_val_mean:.6g} params={json.dumps(best_params, sort_keys=True)}",
    )

    try:
        final_estimator = build_estimator(xgb, task["model"], best_params, args)
        test_pred, final_train_time, final_predict_time = fit_predict_timed(
            final_estimator,
            X_train_outer,
            y_train_outer,
            X_test_outer,
        )
        test_rmse = rmse(y_test_outer, test_pred)
        final_status = "ok"
    except Exception as exc:
        final_train_time = float("nan")
        final_predict_time = float("nan")
        test_rmse = float("nan")
        final_status = f"failed:{type(exc).__name__}:{exc}"

    for row in candidate_rows:
        if row.candidate_index == best_index:
            row.selected = True
            row.final_train_time_sec = final_train_time
            row.final_predict_time_sec = final_predict_time
            row.test_rmse = test_rmse
            row.status = final_status
            break

    summary = {
        "run_name": task["run_name"],
        "experiment": task["experiment"],
        "categorical_handling": task["categorical_handling"],
        "dataset": task["dataset"],
        "model": task["model"],
        "outer_fold": task["outer_index"] + 1,
        "outer_folds": task["outer_folds"],
        "inner_folds": task["inner_folds"],
        "candidate_count": len(candidates),
        "search_type": "random" if task["experiment"] == "exp1" else "grid",
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
        "task_elapsed_sec": time.perf_counter() - task_start,
        "status": final_status,
        "params_json": json.dumps(best_params, sort_keys=True),
    }
    append_task_log(
        task_log_path,
        f"finish status={final_status} test_rmse={test_rmse:.6g} elapsed={summary['task_elapsed_sec']:.1f}s",
    )
    return [asdict(row) for row in candidate_rows], summary


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


def build_parser(run_name: str, categorical_handling: str) -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=f"{run_name}: integrated MetaGB vs CPU XGBoost nested-CV benchmark."
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--outer-fold-index", type=int, default=0)
    parser.add_argument("--run-all-outer-folds", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0, help="Exp1 random candidates; Exp2 ignores this unless --limit-grid-candidates is set")
    parser.add_argument("--models", nargs="+", default=["current_metagb", "xgb_cpu"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=1, help="Model-level threads inside each async task; keep 1 to avoid CPU oversubscription")
    parser.add_argument("--task-workers", type=int, default=env_int("EXP12_TASK_WORKERS", default_cpu_jobs()))
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--xgboost-python-package", default=str(default_xgboost_python_package(root)))
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir(root)))
    parser.add_argument("--output-root", default=str(default_output_root(root)))
    parser.add_argument("--run-id", default=None, help="Optional fixed run id for logs/results; default adds a timestamp")
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--metagb-lambda-l2", type=float, default=1.0)
    parser.add_argument("--metagb-cond-pred-share", action="store_true")
    parser.add_argument("--metagb-leaf-only-use-bias", action="store_true")
    parser.add_argument("--metagb-hist-subtract", dest="metagb_hist_subtract", action="store_true", default=True, help="Enable current MetaGB parent-sibling histogram subtraction cache")
    parser.add_argument("--no-metagb-hist-subtract", dest="metagb_hist_subtract", action="store_false", help="Disable current MetaGB parent-sibling histogram subtraction cache")
    parser.add_argument("--xgb-reg-lambda", type=float, default=0.0)
    parser.add_argument("--limit-grid-candidates", action="store_true", help="Debug only: apply --max-candidates to Exp2 grid search")
    parser.set_defaults(categorical_handling=categorical_handling)
    return parser


def normalize_args(args: argparse.Namespace, experiment: str) -> None:
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least 2")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2")
    if args.mode == "full":
        args.run_all_outer_folds = True
    if experiment == "exp1" and args.max_candidates <= 0:
        args.max_candidates = 5 if args.mode == "quick" else 200
    if experiment == "exp2" and args.max_candidates <= 0:
        args.max_candidates = 5 if args.mode == "quick" else 0
    if args.progress_every <= 0:
        args.progress_every = max(1, args.max_candidates)


def create_run_dirs(args: argparse.Namespace, run_name: str) -> Tuple[str, Path, Path]:
    if args.run_id:
        run_id = safe_name(str(args.run_id))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_name}_{timestamp}"
    output_root = Path(args.output_root).expanduser().resolve()
    logs_dir = output_root / "logs" / run_id
    results_dir = output_root / "results" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_id, logs_dir, results_dir


def main_for_script(run_name: str, experiment: str, categorical_handling: str) -> int:
    parser = build_parser(run_name, categorical_handling)
    args = parser.parse_args()
    normalize_args(args, experiment)
    run_id, logs_dir, results_dir = create_run_dirs(args, run_name)
    logger = setup_logger(logs_dir, run_id)

    try:
        candidate_rows, summary_rows = run_benchmark(
            args,
            run_id,
            experiment,
            categorical_handling,
            logger,
            logs_dir,
        )
        aggregate_rows = build_aggregate(summary_rows)

        candidate_csv = results_dir / f"{run_id}_candidates.csv"
        summary_csv = results_dir / f"{run_id}_summary.csv"
        aggregate_csv = results_dir / f"{run_id}_aggregate.csv"
        metadata_json = results_dir / f"{run_id}_metadata.json"

        write_csv(candidate_csv, candidate_rows)
        write_csv(summary_csv, summary_rows)
        write_csv(aggregate_csv, aggregate_rows)
        with metadata_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment": experiment,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "project_root": str(project_root()),
                    "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
                    "xgboost_python_package": str(Path(args.xgboost_python_package).expanduser().resolve()),
                    "categorical_handling": categorical_handling,
                    "datasets": select_datasets(args.datasets, args.mode),
                    "models": args.models,
                    "outer_folds": args.outer_folds,
                    "inner_folds": args.inner_folds,
                    "run_all_outer_folds": bool(args.run_all_outer_folds),
                    "max_candidates": args.max_candidates,
                    "task_workers": args.task_workers,
                    "model_n_jobs": args.n_jobs,
                    "limit_grid_candidates": bool(args.limit_grid_candidates),
                    "args": vars(args),
                    "candidate_csv": str(candidate_csv),
                    "summary_csv": str(summary_csv),
                    "aggregate_csv": str(aggregate_csv),
                    "log_dir": str(logs_dir),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("candidate rows: %d", len(candidate_rows))
        logger.info("summary rows: %d", len(summary_rows))
        logger.info("aggregate rows: %d", len(aggregate_rows))
        logger.info("results dir: %s", results_dir)
        logger.info("logs dir: %s", logs_dir)
        return 0
    except Exception:
        logger.exception("Run failed")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
