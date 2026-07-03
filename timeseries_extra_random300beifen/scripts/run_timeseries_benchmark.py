from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from random600.utils.candidates import candidate_config_payload, candidate_sets, load_search_space  # noqa: E402
from random600.utils.parameter_audit import build_parameter_audit  # noqa: E402
from random600.utils.runner import build_estimator, import_local_xgboost, safe_name  # noqa: E402
from timeseries_extra_random300.scripts.summarize_results import summarize  # noqa: E402
from timeseries_extra_random300.utils.data import (  # noqa: E402
    FoldArrays,
    load_raw_series,
    load_registry,
    make_fold_arrays,
    resolve_dataset_name,
    supervised_row_count,
)
from timeseries_extra_random300.utils.metrics import finite_mean, finite_std, regression_metrics, select_min_rmse  # noqa: E402
from timeseries_extra_random300.utils.splits import (  # noqa: E402
    assert_time_ordered,
    expanding_window_splits,
    fixed_chronological_split,
)


@dataclass
class CandidateRow:
    run_id: str
    split_protocol: str
    seed: int
    dataset: str
    model: str
    outer_fold: int
    seq_len: int
    pred_len: int
    feature_mode: str
    candidate_index: int
    candidate_id: str
    backbone_id: str
    source: str
    pred_feature_fraction: float
    metagb_score_agg: float
    selected: bool
    status: str
    val_rmse_mean: float
    val_rmse_std: float
    val_mae_mean: float
    val_mse_mean: float
    inner_train_time_sec: float
    inner_predict_time_sec: float
    final_train_time_sec: float
    final_predict_time_sec: float
    test_rmse: float
    test_mae: float
    test_mse: float
    params_json: str


class ColumnwiseMultiOutputRegressor:
    def __init__(self, estimator_factory: Any):
        self.estimator_factory = estimator_factory
        self.estimators_: list[Any] = []
        self.return_1d_ = True

    def fit(self, X: Any, y: Any, verbose: bool = False) -> "ColumnwiseMultiOutputRegressor":
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            estimator = self.estimator_factory()
            estimator.fit(X, y_arr, verbose=verbose)
            self.estimators_ = [estimator]
            self.return_1d_ = True
            return self
        if y_arr.ndim != 2:
            raise ValueError(f"Unsupported multi-output target shape: {y_arr.shape}")
        self.return_1d_ = False
        self.estimators_ = []
        for output_index in range(y_arr.shape[1]):
            estimator = self.estimator_factory()
            estimator.fit(X, y_arr[:, output_index], verbose=verbose)
            self.estimators_.append(estimator)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("ColumnwiseMultiOutputRegressor is not fitted")
        preds = [np.asarray(estimator.predict(X), dtype=np.float32).reshape(-1) for estimator in self.estimators_]
        if self.return_1d_:
            return preds[0]
        return np.column_stack(preds).astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time-ordered nested CV benchmark for tabularized time-series XGBoost vs MetaGB."
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "exchange_rate"])
    parser.add_argument("--models", nargs="+", choices=["xgb_cpu", "current_metagb"], default=["xgb_cpu", "current_metagb"])
    parser.add_argument("--registry", default=str(ADAPTER_ROOT / "configs" / "dataset_registry.json"))
    parser.add_argument("--search-space-json", default=str(ADAPTER_ROOT / "configs" / "timeseries_search_space.json"))
    parser.add_argument("--xgboost-python-package", default=str(PROJECT_ROOT / "beforereview" / "xgboost-master" / "python-package"))
    parser.add_argument("--split-protocol", choices=["nested", "fixed"], default="nested")
    parser.add_argument("--feature-mode", choices=["window", "tabpfn_ts"], default="window")
    parser.add_argument("--target-mode", choices=["direct", "sequence"], default="direct")
    parser.add_argument("--output-scope", choices=["target", "all"], default="target")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=None)
    parser.add_argument("--max-raw-rows", type=int, default=0, help="Optional chronological head slice for fast smoke/debug runs.")
    parser.add_argument("--max-features", type=int, default=0, help="Optional feature/channel cap for lightweight smoke runs; target column is always retained.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--outer-fold-index", type=int, default=0)
    parser.add_argument("--run-all-outer-folds", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--task-workers", type=int, default=1, help="Number of outer tasks to run in parallel.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-root", default=str(ADAPTER_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-time-features", action="store_true", default=True)
    parser.add_argument("--no-time-features", dest="include_time_features", action="store_false")
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--metagb-lambda-l2", type=float, default=1.0)
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--metagb-fixed-score-agg",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Smoke-only MetaGB filter: keep candidates with this score aggregation after generating the balanced 10 pred x 3 agg grid.",
    )
    parser.add_argument("--metagb-pred-value-type", choices=["bin", "raw", "standardized_raw"], default="raw")
    parser.add_argument("--metagb-hist-subtract", dest="metagb_hist_subtract", action="store_true", default=False)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "smoke":
        args.max_candidates = args.max_candidates or 30
    else:
        args.max_candidates = args.max_candidates or 300
        args.run_all_outer_folds = True
    if (
        args.max_candidates % 30 != 0
        and "current_metagb" in args.models
        and args.metagb_fixed_score_agg is None
    ):
        raise ValueError("--max-candidates must be divisible by 30 when current_metagb is enabled")
    args.seq_lens = args.seq_lens or [args.seq_len]
    args.pred_lens = args.pred_lens or [args.pred_len]
    args.seeds = args.seeds or [args.seed]
    args.seq_len = int(args.seq_lens[0])
    args.pred_len = int(args.pred_lens[0])
    args.seed = int(args.seeds[0])
    if args.target_mode == "direct" and args.output_scope != "target":
        raise ValueError("--output-scope all is only supported with --target-mode sequence")
    if any(value < 1 for value in args.seq_lens) or any(value < 1 for value in args.pred_lens):
        raise ValueError("--seq-len/--seq-lens and --pred-len/--pred-lens must be positive")
    if args.max_raw_rows < 0:
        raise ValueError("--max-raw-rows must be non-negative")
    if args.max_features < 0:
        raise ValueError("--max-features must be non-negative")
    if args.outer_folds < 1 or args.inner_folds < 1:
        raise ValueError("--outer-folds and --inner-folds must be positive")
    if args.progress_every <= 0:
        args.progress_every = max(1, args.max_candidates)
    run_id = safe_name(args.run_id) if args.run_id else f"timeseries_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.run_id = run_id
    output_root = Path(args.output_root).expanduser().resolve()
    args.results_dir = str(output_root / "results" / run_id)
    args.logs_dir = str(output_root / "logs" / run_id)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    # Compatibility attributes consumed by random600.utils.runner.build_estimator.
    args.binary_objective = "binary:logistic"
    args.multiclass_objective = "multi:softprob"
    args.categorical_handling = "not_applicable_time_series_numeric_windows"
    return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def update_task_status_file(path: Path, task_id: str, status: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = {"tasks": {}}
    tasks = payload.setdefault("tasks", {})
    tasks[task_id] = status
    payload["completed"] = sum(1 for value in tasks.values() if value != "pending")
    payload["failed"] = sum(1 for value in tasks.values() if str(value).startswith("failed"))
    write_json(path, payload)


def selected_outer_indices(args: argparse.Namespace) -> list[int]:
    if args.run_all_outer_folds or args.mode == "full":
        return list(range(args.outer_folds))
    if args.outer_fold_index < 0 or args.outer_fold_index >= args.outer_folds:
        raise ValueError("--outer-fold-index is zero-based and must be in [0, outer_folds)")
    return [args.outer_fold_index]


def horizon_gap(args: argparse.Namespace) -> int:
    return max(0, int(args.pred_len) - 1) if args.target_mode == "sequence" else 0


def limit_series_features(series: Any, max_features: int) -> Any:
    if max_features <= 0 or series.n_features <= max_features:
        return series
    if max_features < 1:
        raise ValueError("--max-features must be at least 1 when used")
    target_index = int(series.target_index)
    selected = list(range(min(max_features, series.n_features)))
    if target_index not in selected:
        if len(selected) >= max_features:
            selected[-1] = target_index
        else:
            selected.append(target_index)
    selected = sorted(dict.fromkeys(selected))
    series.values = series.values[:, selected]
    series.columns = [series.columns[idx] for idx in selected]
    series.target_index = selected.index(target_index)
    return series


def fit_predict(estimator: Any, fold: FoldArrays) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    estimator.fit(fold.X_train, fold.y_train, verbose=False)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    pred = np.asarray(estimator.predict(fold.X_eval), dtype=np.float32)
    predict_time = time.perf_counter() - start
    return pred, train_time, predict_time


def build_timeseries_estimator(
    xgb: Any,
    model_name: str,
    params: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    def factory() -> Any:
        return build_estimator(xgb, model_name, params, args, "regression", 0)

    if args.target_mode == "sequence" and args.feature_mode == "window":
        return ColumnwiseMultiOutputRegressor(factory)
    return factory()


def output_dim_from_fold(fold: FoldArrays | None) -> int:
    if fold is None:
        return 0
    y = np.asarray(fold.y_train)
    return 1 if y.ndim == 1 else int(y.shape[1])


def input_dim_from_fold(fold: FoldArrays | None) -> int:
    if fold is None:
        return 0
    return int(np.asarray(fold.X_train).shape[1])


def split_timestamp(series_timestamps: list[str], position: int) -> str:
    if position < 0:
        return ""
    if position >= len(series_timestamps):
        return ""
    return series_timestamps[position]


def evaluate_task(
    *,
    xgb: Any,
    args: argparse.Namespace,
    series: Any,
    model_name: str,
    outer_index: int,
    candidates: list[dict[str, Any]],
    outer_splits: list[Any],
    split_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outer_train_rows, outer_test_rows, outer_split = outer_splits[outer_index]
    assert_time_ordered(outer_train_rows, outer_test_rows)
    inner_splits_relative = expanding_window_splits(
        len(outer_train_rows),
        args.inner_folds,
        initial_train_fraction=0.5,
        gap=horizon_gap(args),
    )
    inner_splits = []
    for inner_train_rel, inner_val_rel, inner_split in inner_splits_relative:
        inner_train_rows = outer_train_rows[inner_train_rel]
        inner_val_rows = outer_train_rows[inner_val_rel]
        assert_time_ordered(inner_train_rows, inner_val_rows)
        inner_splits.append((inner_train_rows, inner_val_rows, inner_split))

    rows: list[CandidateRow] = []
    scores: list[tuple[float, float, int, str]] = []
    task_start = time.perf_counter()
    for candidate_index, params in enumerate(candidates, start=1):
        val_rmses: list[float] = []
        val_maes: list[float] = []
        val_mses: list[float] = []
        inner_train_time = 0.0
        inner_predict_time = 0.0
        status = "ok"
        for inner_train_rows, inner_val_rows, _inner_split in inner_splits:
            try:
                fold = make_fold_arrays(
                    series,
                    inner_train_rows,
                    inner_val_rows,
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    include_time_features=args.include_time_features,
                    target_mode=args.target_mode,
                    output_scope=args.output_scope,
                    feature_mode=args.feature_mode,
                )
                estimator = build_timeseries_estimator(xgb, model_name, params, args)
                pred, train_time, predict_time = fit_predict(estimator, fold)
                metrics = regression_metrics(fold.y_eval, pred)
                val_rmses.append(metrics["rmse"])
                val_maes.append(metrics["mae"])
                val_mses.append(metrics["mse"])
                inner_train_time += train_time
                inner_predict_time += predict_time
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
                val_rmses = [float("nan")]
                val_maes = [float("nan")]
                val_mses = [float("nan")]
                inner_train_time = 0.0
                inner_predict_time = 0.0
                break

        val_rmse_mean = finite_mean(val_rmses)
        val_rmse_std = finite_std(val_rmses)
        scores.append((val_rmse_mean, inner_train_time + inner_predict_time, candidate_index, status))
        rows.append(
            CandidateRow(
                run_id=args.run_id,
                split_protocol=args.split_protocol,
                seed=args.seed,
                dataset=series.name,
                model=model_name,
                outer_fold=outer_index + 1,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                feature_mode=args.feature_mode,
                candidate_index=candidate_index,
                candidate_id=str(params.get("candidate_id", candidate_index)),
                backbone_id=str(params.get("backbone_id", "")),
                source=str(params.get("source", "")),
                pred_feature_fraction=float(params.get("pred_feature_fraction", float("nan"))),
                metagb_score_agg=float(params.get("metagb_score_agg", float("nan"))),
                selected=False,
                status=status,
                val_rmse_mean=val_rmse_mean,
                val_rmse_std=val_rmse_std,
                val_mae_mean=finite_mean(val_maes),
                val_mse_mean=finite_mean(val_mses),
                inner_train_time_sec=inner_train_time,
                inner_predict_time_sec=inner_predict_time,
                final_train_time_sec=float("nan"),
                final_predict_time_sec=float("nan"),
                test_rmse=float("nan"),
                test_mae=float("nan"),
                test_mse=float("nan"),
                params_json=json.dumps(params, sort_keys=True),
            )
        )
        if candidate_index == 1 or candidate_index == len(candidates) or candidate_index % args.progress_every == 0:
            print(
                f"{series.name} {model_name} outer={outer_index + 1} "
                f"candidate={candidate_index}/{len(candidates)} val_rmse={val_rmse_mean:.6g} status={status}",
                flush=True,
            )

    best_index = select_min_rmse(scores)
    if best_index < 0:
        return [asdict(row) for row in rows], {
            "run_id": args.run_id,
            "dataset": series.name,
            "model": model_name,
            "outer_fold": outer_index + 1,
            "status": "no_valid_candidate",
        }

    best_params = candidates[best_index - 1]
    try:
        final_fold = make_fold_arrays(
            series,
            outer_train_rows,
            outer_test_rows,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            include_time_features=args.include_time_features,
            target_mode=args.target_mode,
            output_scope=args.output_scope,
            feature_mode=args.feature_mode,
        )
        final_estimator = build_timeseries_estimator(xgb, model_name, best_params, args)
        test_pred, final_train_time, final_predict_time = fit_predict(final_estimator, final_fold)
        test_metrics = regression_metrics(final_fold.y_eval, test_pred)
        final_status = "ok"
    except Exception as exc:
        final_fold = None
        final_train_time = float("nan")
        final_predict_time = float("nan")
        test_metrics = {"rmse": float("nan"), "mae": float("nan"), "mse": float("nan")}
        final_status = f"failed:{type(exc).__name__}:{exc}"

    for row in rows:
        if row.candidate_index == best_index:
            row.selected = True
            row.status = final_status
            row.final_train_time_sec = final_train_time
            row.final_predict_time_sec = final_predict_time
            row.test_rmse = test_metrics["rmse"]
            row.test_mae = test_metrics["mae"]
            row.test_mse = test_metrics["mse"]
            break

    selected_score = scores[best_index - 1]
    split_record = {
        "dataset": series.name,
        "model": model_name,
        "outer_fold": outer_index + 1,
        "outer_split": outer_split.to_dict(),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "output_scope": args.output_scope,
        "input_features": input_dim_from_fold(final_fold),
        "output_dim": output_dim_from_fold(final_fold),
        "train_row_min": int(outer_train_rows.min()),
        "train_row_max": int(outer_train_rows.max()),
        "test_row_min": int(outer_test_rows.min()),
        "test_row_max": int(outer_test_rows.max()),
    }
    if final_fold is not None:
        split_record.update(
            {
                "scaler_fit_start": final_fold.scaler_fit_start,
                "scaler_fit_end": final_fold.scaler_fit_end,
                "train_target_start": final_fold.train_target_start,
                "train_target_end": final_fold.train_target_end,
                "test_target_start": final_fold.eval_target_start,
                "test_target_end": final_fold.eval_target_end,
                "train_target_start_time": split_timestamp(series.timestamps, final_fold.train_target_start),
                "train_target_end_time": split_timestamp(series.timestamps, final_fold.train_target_end - 1),
                "test_target_start_time": split_timestamp(series.timestamps, final_fold.eval_target_start),
                "test_target_end_time": split_timestamp(series.timestamps, final_fold.eval_target_end - 1),
            }
        )
    split_records.append(split_record)

    summary = {
        "run_id": args.run_id,
        "split_protocol": args.split_protocol,
        "seed": args.seed,
        "dataset": series.name,
        "family": series.spec.family,
        "model": model_name,
        "outer_fold": outer_index + 1,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "output_scope": args.output_scope,
        "input_features": input_dim_from_fold(final_fold),
        "output_dim": output_dim_from_fold(final_fold),
        "rows_raw": series.n_rows,
        "rows_supervised": supervised_row_count(series, args.seq_len, args.pred_len),
        "features_raw": series.n_features,
        "max_raw_rows": args.max_raw_rows,
        "max_features": args.max_features,
        "candidate_count": len(candidates),
        "selected_candidate_index": best_index,
        "selected_candidate_id": str(best_params.get("candidate_id", best_index)),
        "selected_backbone_id": str(best_params.get("backbone_id", "")),
        "selected_pred_feature_fraction": float(best_params.get("pred_feature_fraction", float("nan"))),
        "selected_metagb_score_agg": float(best_params.get("metagb_score_agg", float("nan"))),
        "val_rmse_mean": selected_score[0],
        "val_rmse_std": rows[best_index - 1].val_rmse_std,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "inner_train_time_sec": rows[best_index - 1].inner_train_time_sec,
        "inner_predict_time_sec": rows[best_index - 1].inner_predict_time_sec,
        "final_train_time_sec": final_train_time,
        "final_predict_time_sec": final_predict_time,
        "total_train_time_sec": rows[best_index - 1].inner_train_time_sec + final_train_time,
        "total_predict_time_sec": rows[best_index - 1].inner_predict_time_sec + final_predict_time,
        "task_elapsed_sec": time.perf_counter() - task_start,
        "status": final_status,
        "params_json": json.dumps(best_params, sort_keys=True),
    }
    return [asdict(row) for row in rows], summary


def evaluate_fixed_task(
    *,
    xgb: Any,
    args: argparse.Namespace,
    series: Any,
    model_name: str,
    candidates: list[dict[str, Any]],
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    fixed_split: Any,
    fixed_splits_by_seq_len: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, Any]] | None,
    split_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fixed_splits_by_seq_len is None:
        fixed_splits_by_seq_len = {args.seq_len: (train_rows, val_rows, test_rows, fixed_split)}
    rows: list[CandidateRow] = []
    scores: list[tuple[float, float, int, str]] = []
    selected_options: dict[int, tuple[int, int, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Any]] = {}
    task_start = time.perf_counter()
    candidate_index = 0
    for seq_len in [args.seq_len]:
        seq_train_rows, seq_val_rows, seq_test_rows, seq_fixed_split = fixed_splits_by_seq_len[seq_len]
        assert_time_ordered(seq_train_rows, seq_val_rows)
        assert_time_ordered(seq_val_rows, seq_test_rows)
        for base_candidate_index, params in enumerate(candidates, start=1):
            candidate_index += 1
            status = "ok"
            try:
                val_fold = make_fold_arrays(
                    series,
                    seq_train_rows,
                    seq_val_rows,
                    seq_len=seq_len,
                    pred_len=args.pred_len,
                    include_time_features=args.include_time_features,
                    target_mode=args.target_mode,
                    output_scope=args.output_scope,
                    feature_mode=args.feature_mode,
                )
                estimator = build_timeseries_estimator(xgb, model_name, params, args)
                pred, train_time, predict_time = fit_predict(estimator, val_fold)
                metrics = regression_metrics(val_fold.y_eval, pred)
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
                train_time = 0.0
                predict_time = 0.0
                metrics = {"rmse": float("nan"), "mae": float("nan"), "mse": float("nan")}

            scores.append((metrics["rmse"], train_time + predict_time, candidate_index, status))
            selected_options[candidate_index] = (
                seq_len,
                base_candidate_index,
                params,
                seq_train_rows,
                seq_val_rows,
                seq_test_rows,
                seq_fixed_split,
            )
            rows.append(
                CandidateRow(
                    run_id=args.run_id,
                    split_protocol=args.split_protocol,
                    seed=args.seed,
                    dataset=series.name,
                    model=model_name,
                    outer_fold=1,
                    seq_len=seq_len,
                    pred_len=args.pred_len,
                    feature_mode=args.feature_mode,
                    candidate_index=candidate_index,
                    candidate_id=str(params.get("candidate_id", base_candidate_index)),
                    backbone_id=str(params.get("backbone_id", "")),
                    source=str(params.get("source", "")),
                    pred_feature_fraction=float(params.get("pred_feature_fraction", float("nan"))),
                    metagb_score_agg=float(params.get("metagb_score_agg", float("nan"))),
                    selected=False,
                    status=status,
                    val_rmse_mean=metrics["rmse"],
                    val_rmse_std=0.0,
                    val_mae_mean=metrics["mae"],
                    val_mse_mean=metrics["mse"],
                    inner_train_time_sec=train_time,
                    inner_predict_time_sec=predict_time,
                    final_train_time_sec=float("nan"),
                    final_predict_time_sec=float("nan"),
                    test_rmse=float("nan"),
                    test_mae=float("nan"),
                    test_mse=float("nan"),
                    params_json=json.dumps(params, sort_keys=True),
                )
            )
            if (
                base_candidate_index == 1
                or base_candidate_index == len(candidates)
                or base_candidate_index % args.progress_every == 0
            ):
                print(
                    f"{series.name} {model_name} seed={args.seed} seq={seq_len} pred={args.pred_len} "
                    f"candidate={base_candidate_index}/{len(candidates)} "
                    f"val_rmse={metrics['rmse']:.6g} status={status}",
                    flush=True,
                )

    best_index = select_min_rmse(scores)
    if best_index < 0:
        return [asdict(row) for row in rows], {
            "run_id": args.run_id,
            "split_protocol": args.split_protocol,
            "seed": args.seed,
            "dataset": series.name,
            "model": model_name,
            "outer_fold": 1,
            "seq_lens": args.seq_lens,
            "pred_len": args.pred_len,
            "status": "no_valid_candidate",
        }

    (
        selected_seq_len,
        selected_base_candidate_index,
        best_params,
        selected_train_rows,
        selected_val_rows,
        selected_test_rows,
        selected_fixed_split,
    ) = selected_options[best_index]
    final_train_rows = np.concatenate([selected_train_rows, selected_val_rows])
    try:
        final_fold = make_fold_arrays(
            series,
            final_train_rows,
            selected_test_rows,
            seq_len=selected_seq_len,
            pred_len=args.pred_len,
            include_time_features=args.include_time_features,
            target_mode=args.target_mode,
            output_scope=args.output_scope,
            feature_mode=args.feature_mode,
        )
        final_estimator = build_timeseries_estimator(xgb, model_name, best_params, args)
        test_pred, final_train_time, final_predict_time = fit_predict(final_estimator, final_fold)
        test_metrics = regression_metrics(final_fold.y_eval, test_pred)
        final_status = "ok"
    except Exception as exc:
        final_fold = None
        final_train_time = float("nan")
        final_predict_time = float("nan")
        test_metrics = {"rmse": float("nan"), "mae": float("nan"), "mse": float("nan")}
        final_status = f"failed:{type(exc).__name__}:{exc}"

    for row in rows:
        if row.candidate_index == best_index:
            row.selected = True
            row.status = final_status
            row.final_train_time_sec = final_train_time
            row.final_predict_time_sec = final_predict_time
            row.test_rmse = test_metrics["rmse"]
            row.test_mae = test_metrics["mae"]
            row.test_mse = test_metrics["mse"]
            break

    split_record = {
        "dataset": series.name,
        "model": model_name,
        "split_protocol": args.split_protocol,
        "seed": args.seed,
        "outer_fold": 1,
        "fixed_split": selected_fixed_split.to_dict(),
        "seq_len": selected_seq_len,
        "pred_len": args.pred_len,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "output_scope": args.output_scope,
        "input_features": input_dim_from_fold(final_fold),
        "output_dim": output_dim_from_fold(final_fold),
        "train_row_min": int(selected_train_rows.min()),
        "train_row_max": int(selected_train_rows.max()),
        "val_row_min": int(selected_val_rows.min()),
        "val_row_max": int(selected_val_rows.max()),
        "test_row_min": int(selected_test_rows.min()),
        "test_row_max": int(selected_test_rows.max()),
    }
    if final_fold is not None:
        split_record.update(
            {
                "scaler_fit_start": final_fold.scaler_fit_start,
                "scaler_fit_end": final_fold.scaler_fit_end,
                "train_val_target_start": final_fold.train_target_start,
                "train_val_target_end": final_fold.train_target_end,
                "test_target_start": final_fold.eval_target_start,
                "test_target_end": final_fold.eval_target_end,
                "train_val_target_start_time": split_timestamp(series.timestamps, final_fold.train_target_start),
                "train_val_target_end_time": split_timestamp(series.timestamps, final_fold.train_target_end - 1),
                "test_target_start_time": split_timestamp(series.timestamps, final_fold.eval_target_start),
                "test_target_end_time": split_timestamp(series.timestamps, final_fold.eval_target_end - 1),
            }
        )
    split_records.append(split_record)

    selected_score = scores[best_index - 1]
    summary = {
        "run_id": args.run_id,
        "split_protocol": args.split_protocol,
        "seed": args.seed,
        "dataset": series.name,
        "family": series.spec.family,
        "model": model_name,
        "outer_fold": 1,
        "outer_folds": 1,
        "inner_folds": 1,
        "seq_len": selected_seq_len,
        "pred_len": args.pred_len,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "output_scope": args.output_scope,
        "input_features": input_dim_from_fold(final_fold),
        "output_dim": output_dim_from_fold(final_fold),
        "rows_raw": series.n_rows,
        "rows_supervised": supervised_row_count(series, selected_seq_len, args.pred_len),
        "features_raw": series.n_features,
        "window_features": selected_seq_len * series.n_features,
        "max_raw_rows": args.max_raw_rows,
        "max_features": args.max_features,
        "candidate_count": len(candidates),
        "seq_len_candidate_count": 1,
        "evaluated_candidate_count": len(rows),
        "selected_candidate_index": best_index,
        "selected_base_candidate_index": selected_base_candidate_index,
        "selected_candidate_id": str(best_params.get("candidate_id", selected_base_candidate_index)),
        "selected_backbone_id": str(best_params.get("backbone_id", "")),
        "selected_pred_feature_fraction": float(best_params.get("pred_feature_fraction", float("nan"))),
        "selected_metagb_score_agg": float(best_params.get("metagb_score_agg", float("nan"))),
        "val_rmse_mean": selected_score[0],
        "val_rmse_std": rows[best_index - 1].val_rmse_std,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "inner_train_time_sec": rows[best_index - 1].inner_train_time_sec,
        "inner_predict_time_sec": rows[best_index - 1].inner_predict_time_sec,
        "final_train_time_sec": final_train_time,
        "final_predict_time_sec": final_predict_time,
        "total_train_time_sec": rows[best_index - 1].inner_train_time_sec + final_train_time,
        "total_predict_time_sec": rows[best_index - 1].inner_predict_time_sec + final_predict_time,
        "task_elapsed_sec": time.perf_counter() - task_start,
        "status": final_status,
        "params_json": json.dumps(best_params, sort_keys=True),
    }
    return [asdict(row) for row in rows], summary


def create_tasks(
    args: argparse.Namespace,
    registry: dict[str, Any],
    candidates_by_seed: dict[int, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if args.split_protocol == "fixed":
        for seed in args.seeds:
            series_by_dataset: dict[str, Any] = {}
            for dataset_name in args.datasets:
                resolved_name = resolve_dataset_name(registry, dataset_name)
                series = load_raw_series(ADAPTER_ROOT, registry[resolved_name])
                if args.max_raw_rows and series.n_rows > args.max_raw_rows:
                    series.values = series.values[: args.max_raw_rows]
                    series.timestamps = series.timestamps[: args.max_raw_rows]
                series_by_dataset[resolved_name] = limit_series_features(series, args.max_features)
            for seq_len in args.seq_lens:
                for pred_len in args.pred_lens:
                    for dataset_name in args.datasets:
                        dataset_name = resolve_dataset_name(registry, dataset_name)
                        series = series_by_dataset[dataset_name]
                        n_rows = supervised_row_count(series, seq_len, pred_len)
                        task_args = dict(vars(args))
                        task_args.update({"seed": seed, "seq_len": seq_len, "pred_len": pred_len})
                        split_gap = horizon_gap(argparse.Namespace(**task_args))
                        train_rows, val_rows, test_rows, fixed_split = fixed_chronological_split(n_rows, gap=split_gap)
                        fixed_splits_by_seq_len = {seq_len: (train_rows, val_rows, test_rows, fixed_split)}
                        for model_name in args.models:
                            task_id = (
                                f"{safe_name(dataset_name)}__{safe_name(model_name)}"
                                f"__seed{seed}__s{seq_len}__p{pred_len}"
                            )
                            tasks.append(
                                {
                                    "task_id": task_id,
                                    "series": series,
                                    "model_name": model_name,
                                    "candidates": candidates_by_seed[seed][model_name],
                                    "train_rows": train_rows,
                                    "val_rows": val_rows,
                                    "test_rows": test_rows,
                                    "fixed_split": fixed_split,
                                    "fixed_splits_by_seq_len": fixed_splits_by_seq_len,
                                    "args": task_args,
                                }
                            )
    else:
        for dataset_name in args.datasets:
            dataset_name = resolve_dataset_name(registry, dataset_name)
            series = load_raw_series(ADAPTER_ROOT, registry[dataset_name])
            if args.max_raw_rows and series.n_rows > args.max_raw_rows:
                series.values = series.values[: args.max_raw_rows]
                series.timestamps = series.timestamps[: args.max_raw_rows]
            series = limit_series_features(series, args.max_features)
            for seed in args.seeds:
                for seq_len in args.seq_lens:
                    for pred_len in args.pred_lens:
                        n_rows = supervised_row_count(series, seq_len, pred_len)
                        task_args_base = dict(vars(args))
                        task_args_base.update({"seed": seed, "seq_len": seq_len, "pred_len": pred_len})
                        split_gap = horizon_gap(argparse.Namespace(**task_args_base))
                        outer_splits = expanding_window_splits(
                            n_rows,
                            args.outer_folds,
                            initial_train_fraction=0.5,
                            gap=split_gap,
                        )
                        for outer_index in selected_outer_indices(args):
                            for model_name in args.models:
                                task_args = dict(task_args_base)
                                task_id = (
                                    f"{safe_name(dataset_name)}__{safe_name(model_name)}"
                                    f"__seed{seed}__s{seq_len}__p{pred_len}__out{outer_index + 1}"
                                )
                                tasks.append(
                                    {
                                        "task_id": task_id,
                                        "series": series,
                                        "model_name": model_name,
                                        "outer_index": outer_index,
                                        "candidates": candidates_by_seed[seed][model_name],
                                        "outer_splits": outer_splits,
                                        "args": task_args,
                                    }
                                )
    return tasks


def run_outer_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    args = argparse.Namespace(**task["args"])
    xgb = import_local_xgboost(Path(args.xgboost_python_package))
    split_records: list[dict[str, Any]] = []
    if args.split_protocol == "fixed":
        candidate_rows, summary = evaluate_fixed_task(
            xgb=xgb,
            args=args,
            series=task["series"],
            model_name=task["model_name"],
            candidates=task["candidates"],
            train_rows=task["train_rows"],
            val_rows=task["val_rows"],
            test_rows=task["test_rows"],
            fixed_split=task["fixed_split"],
            fixed_splits_by_seq_len=task.get("fixed_splits_by_seq_len"),
            split_records=split_records,
        )
    else:
        candidate_rows, summary = evaluate_task(
            xgb=xgb,
            args=args,
            series=task["series"],
            model_name=task["model_name"],
            outer_index=task["outer_index"],
            candidates=task["candidates"],
            outer_splits=task["outer_splits"],
            split_records=split_records,
        )
    return candidate_rows, summary, split_records


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    registry = load_registry(Path(args.registry))
    args.datasets = [resolve_dataset_name(registry, name) for name in args.datasets]
    search_space = load_search_space(Path(args.search_space_json))
    candidates_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}
    candidate_warnings_by_seed: dict[int, list[str]] = {}
    for seed in args.seeds:
        generation_count = args.max_candidates
        if args.metagb_fixed_score_agg is not None and "current_metagb" in args.models:
            generation_count = max(30, args.max_candidates * 3)
        candidates_by_seed[seed], candidate_warnings_by_seed[seed] = candidate_sets(
            search_space, generation_count, seed, None, args.models
        )
        for model_name, candidates in list(candidates_by_seed[seed].items()):
            if model_name != "current_metagb" and len(candidates) > args.max_candidates:
                candidates_by_seed[seed][model_name] = candidates[: args.max_candidates]
        if args.metagb_fixed_score_agg is not None and "current_metagb" in candidates_by_seed[seed]:
            filtered = [
                candidate
                for candidate in candidates_by_seed[seed]["current_metagb"]
                if int(candidate.get("metagb_score_agg", -1)) == int(args.metagb_fixed_score_agg)
            ]
            candidates_by_seed[seed]["current_metagb"] = filtered[: args.max_candidates]
            candidate_warnings_by_seed[seed].append(
                f"current_metagb filtered to metagb_score_agg={args.metagb_fixed_score_agg}; "
                f"kept {len(candidates_by_seed[seed]['current_metagb'])} of {generation_count} generated candidates"
            )
    results_dir = Path(args.results_dir)
    if len(args.seeds) == 1:
        write_json(results_dir / "candidate_configs.json", candidate_config_payload(candidates_by_seed[args.seeds[0]]))
        audit_warnings = candidate_warnings_by_seed[args.seeds[0]]
    else:
        write_json(
            results_dir / "candidate_configs.json",
            {
                str(seed): candidate_config_payload(candidates_by_model)
                for seed, candidates_by_model in candidates_by_seed.items()
            },
        )
        audit_warnings = [
            f"seed={seed}: {warning}"
            for seed, warnings in candidate_warnings_by_seed.items()
            for warning in warnings
        ]
    write_json(results_dir / "parameter_audit.json", build_parameter_audit(args, search_space, audit_warnings))

    tasks = create_tasks(args, registry, candidates_by_seed)
    task_status_path = Path(args.logs_dir) / "task_status.json"
    write_json(
        task_status_path,
        {
            "run_id": args.run_id,
            "total": len(tasks),
            "completed": 0,
            "failed": 0,
            "tasks": {task["task_id"]: "pending" for task in tasks},
        },
    )
    all_candidate_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    if args.split_protocol == "fixed":
        task_batches = [
            [
                task
                for task in tasks
                if int(task["args"].get("seed", args.seed)) == int(seed)
                and int(task["args"].get("seq_len", args.seq_len)) == int(seq_len)
            ]
            for seed in args.seeds
            for seq_len in args.seq_lens
        ]
        task_batches = [batch for batch in task_batches if batch]
    else:
        task_batches = [tasks]
    worker_count = min(max(1, args.task_workers), len(tasks))
    task_executor = "sequential"
    completed_task_count = 0
    if worker_count == 1:
        for batch in task_batches:
            for task in batch:
                completed_task_count += 1
                candidate_rows, summary, task_split_records = run_outer_task(task)
                all_candidate_rows.extend(candidate_rows)
                all_summary_rows.append(summary)
                split_records.extend(task_split_records)
                update_task_status_file(task_status_path, task["task_id"], str(summary.get("status")))
                print(f"[{completed_task_count}/{len(tasks)}] {task['task_id']} status={summary.get('status')}", flush=True)
                write_csv(results_dir / "candidates.csv", all_candidate_rows)
                write_csv(results_dir / "summary.csv", all_summary_rows)
                write_json(results_dir / "split_manifest.json", {"splits": split_records})
    else:
        task_executor = "process"
        try:
            executor = ProcessPoolExecutor(max_workers=worker_count)
        except (OSError, PermissionError) as exc:
            task_executor = "thread"
            print(
                f"WARNING: ProcessPoolExecutor unavailable ({type(exc).__name__}: {exc}); "
                "falling back to ThreadPoolExecutor.",
                flush=True,
            )
            executor = ThreadPoolExecutor(max_workers=worker_count)
        with executor:
            for batch in task_batches:
                future_map = {executor.submit(run_outer_task, task): task for task in batch}
                for future in as_completed(future_map):
                    completed_task_count += 1
                    task = future_map[future]
                    try:
                        candidate_rows, summary, task_split_records = future.result()
                    except Exception as exc:
                        candidate_rows = []
                        task_split_records = []
                        summary = {
                            "run_id": args.run_id,
                            "dataset": task["series"].name,
                            "model": task["model_name"],
                            "outer_fold": int(task.get("outer_index", 0)) + 1,
                            "status": f"failed:{type(exc).__name__}:{exc}",
                        }
                    all_candidate_rows.extend(candidate_rows)
                    all_summary_rows.append(summary)
                    split_records.extend(task_split_records)
                    update_task_status_file(task_status_path, task["task_id"], str(summary.get("status")))
                    print(f"[{completed_task_count}/{len(tasks)}] {task['task_id']} status={summary.get('status')}", flush=True)
                    write_csv(results_dir / "candidates.csv", all_candidate_rows)
                    write_csv(results_dir / "summary.csv", all_summary_rows)
                    write_json(results_dir / "split_manifest.json", {"splits": split_records})
    all_candidate_rows.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("seed", 0),
            row.get("seq_len", 0),
            row.get("pred_len", 0),
            row.get("outer_fold", 0),
            row.get("candidate_index", 0),
        )
    )
    all_summary_rows.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("seed", 0),
            row.get("seq_len", 0),
            row.get("pred_len", 0),
            row.get("outer_fold", 0),
        )
    )
    split_records.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("seed", 0),
            row.get("seq_len", 0),
            row.get("pred_len", 0),
            row.get("outer_fold", 0),
        )
    )
    write_csv(results_dir / "candidates.csv", all_candidate_rows)
    write_csv(results_dir / "summary.csv", all_summary_rows)
    write_json(results_dir / "split_manifest.json", {"splits": split_records})
    write_json(
        results_dir / "metadata.json",
        {
            "run_id": args.run_id,
            "mode": args.mode,
            "datasets": args.datasets,
            "models": args.models,
            "split_protocol": args.split_protocol,
            "feature_mode": args.feature_mode,
            "target_mode": args.target_mode,
            "output_scope": args.output_scope,
            "seq_len": args.seq_len,
            "seq_lens": args.seq_lens,
            "pred_len": args.pred_len,
            "pred_lens": args.pred_lens,
            "seed": args.seed,
            "seeds": args.seeds,
            "max_raw_rows": args.max_raw_rows,
            "max_features": args.max_features,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "max_candidates": args.max_candidates,
            "task_workers": args.task_workers,
            "effective_task_workers": worker_count,
            "task_executor": task_executor,
            "results_dir": str(results_dir),
            "summary_rows": len(all_summary_rows),
            "candidate_rows": len(all_candidate_rows),
        },
    )
    summarize(results_dir)
    return {
        "results_dir": str(results_dir),
        "summary_rows": len(all_summary_rows),
        "candidate_rows": len(all_candidate_rows),
    }


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    metadata = run_all(args)
    print(f"results_dir: {metadata['results_dir']}")
    print(f"summary_rows: {metadata['summary_rows']}")
    print(f"candidate_rows: {metadata['candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
