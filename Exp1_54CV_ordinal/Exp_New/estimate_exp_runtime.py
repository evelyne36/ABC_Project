#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单进程单核单线程的 SingleTime 统计脚本（面向当前 Exp_New 工程）。

核心功能：
1) 强制单线程环境变量（OMP/MKL/OPENBLAS 等）；
2) 可选设置当前进程 CPU 亲和性（cores_per_process）；
3) 统计每个 exp*.py 下，每个数据集、每个模型（XGBoost/MetaGB）单次 fit() 的 SingleTime；
4) 基于 SingleTime 推演总时间（默认 5x4 折 + 200 次搜索）；
5) 生成 timeAnalysis<timestamp>.log 和 运行时间计算分析.md。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
]


def _set_thread_env(n_threads: int) -> None:
    n_threads = max(1, int(n_threads))
    for key in THREAD_ENV_KEYS:
        os.environ[key] = str(n_threads)


_set_thread_env(1)

import numpy as np  # noqa: E402


@dataclass
class Task:
    exp_file: str
    exp_kind: str
    dataset_name: str
    dataset_path: Path
    model_type: str
    params: Dict[str, Any]
    search_iterations: int
    outer_folds: int
    inner_folds: int


@dataclass
class Result:
    exp_file: str
    exp_kind: str
    dataset_name: str
    dataset_path: str
    model_type: str
    n_samples: Optional[int]
    n_features: Optional[int]
    single_time_sec: Optional[float]
    total_time_sec: Optional[float]
    status: str
    error: str


class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.fp = log_path.open("w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        self.fp.write(msg + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()


def _safe_first(v: Any) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            raise ValueError("参数网格存在空列表")
        return v[0]
    return v


def _first_combination(param_grid: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _safe_first(v) for k, v in param_grid.items()}


def _import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detect_exp_kind(exp_file: str) -> str:
    name = exp_file.lower()
    if "synthetic_nonlinear" in name:
        return "synthetic_exp2a"
    if "synthetic_interaction" in name:
        return "synthetic_exp2b"
    if "pred_fraction_study" in name:
        return "pred_fraction"
    return "tabarena_nested"


def _apply_cpu_affinity(cores_per_process: int, logger: TeeLogger) -> None:
    cores_per_process = max(1, int(cores_per_process))
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        current = p.cpu_affinity()
        if not current:
            return
        target = current[:cores_per_process]
        p.cpu_affinity(target)
        logger.log(f"[资源限制] cpu_affinity={target}")
    except Exception as exc:
        logger.log(f"[资源限制] 未设置 cpu_affinity（忽略）：{exc}")

    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        threadpool_limits(limits=1)
        logger.log("[资源限制] threadpoolctl 已限制为 1 线程")
    except Exception as exc:
        logger.log(f"[资源限制] threadpoolctl 不可用（忽略）：{exc}")


def _load_dataset_csv_fallback(csv_path: Path, categorical_handling: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if categorical_handling == "drop":
        df = df.select_dtypes(include=[np.number]).dropna()
    elif categorical_handling == "onehot":
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=list(categorical_cols), drop_first=True)
        df = df.dropna()
    elif categorical_handling == "ordinal":
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df[col], _ = pd.factorize(df[col], sort=True)
        df = df.dropna()
    else:
        raise ValueError(f"未知 categorical_handling: {categorical_handling}")

    if len(df) == 0:
        raise ValueError(f"空数据集: {csv_path}")

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y


def _map_xgb_params(params: Dict[str, Any]) -> Dict[str, Any]:
    xgb_params = dict(params)
    for k in [
        "pred_feature_fraction",
        "version",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "num_trees",
        "lambda_l2",
        "seed",
    ]:
        xgb_params.pop(k, None)

    if "colsample" in xgb_params and "colsample_bytree" not in xgb_params:
        xgb_params["colsample_bytree"] = xgb_params.pop("colsample")

    xgb_params.setdefault("objective", "reg:squarederror")
    xgb_params.setdefault("tree_method", "hist")
    xgb_params.setdefault("device", "cpu")
    xgb_params["n_jobs"] = 1

    for k in ["max_depth", "n_estimators", "max_bin", "min_child_weight"]:
        if k in xgb_params:
            xgb_params[k] = int(round(float(xgb_params[k])))
    return xgb_params


def _map_metagb_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    feature_fraction = p.get("feature_fraction", p.get("colsample_bytree", p.get("colsample", 0.8)))
    bagging_fraction = p.get("bagging_fraction", p.get("subsample", 0.8))
    min_data_in_leaf = p.get("min_data_in_leaf", p.get("min_child_weight", 20))
    num_trees = p.get("num_trees", p.get("n_estimators", 100))
    seed = p.get("seed", p.get("random_state", 42))
    pred_feature_fraction = p.get("pred_feature_fraction", feature_fraction)

    mapped = {
        "learning_rate": float(p.get("learning_rate", 0.1)),
        "num_trees": int(round(float(num_trees))),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "min_data_in_leaf": int(round(float(min_data_in_leaf))),
        "max_depth": int(round(float(p.get("max_depth", 6)))),
        "max_bin": int(round(float(p.get("max_bin", 255)))),
        "pred_feature_fraction": float(pred_feature_fraction),
        "version": int(round(float(p.get("version", 3)))),
        "seed": int(round(float(seed))),
    }
    if "lambda_l2" in p:
        mapped["lambda_l2"] = float(p["lambda_l2"])
    elif "reg_lambda" in p:
        mapped["lambda_l2"] = float(p["reg_lambda"])
    return mapped


def _measure_single_fit(model_type: str, params: Dict[str, Any], x: np.ndarray, y: np.ndarray, seed: int) -> float:
    start = time.perf_counter()
    if model_type == "xgboost":
        import xgboost as xgb

        xgb_params = _map_xgb_params(params)
        model = xgb.XGBRegressor(**xgb_params, random_state=seed)
        model.fit(x, y, verbose=False)
    else:
        from metagb_utils import MetaGB

        mgb_params = _map_metagb_params(params)
        model = MetaGB(**mgb_params)
        model.fit(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64))
    return time.perf_counter() - start


def _resolve_tabarena_dataset_paths(base_dir: Path, dataset_names: Iterable[str]) -> List[Tuple[str, Path]]:
    data_dir_candidates = [
        base_dir.parent / "dataset" / "tabArena_regression",
        base_dir / "dataset" / "tabArena_regression",
    ]
    data_dir = None
    for candidate in data_dir_candidates:
        if candidate.exists():
            data_dir = candidate
            break
    if data_dir is None:
        raise FileNotFoundError("未找到 tabArena_regression 数据目录")

    paths = []
    for ds in dataset_names:
        p = data_dir / ds
        paths.append((ds, p))
    return paths


def _resolve_npz_dataset_paths(base_dir: Path, sub_dir: str) -> List[Tuple[str, Path]]:
    data_dir = base_dir / "data" / sub_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    files = sorted(data_dir.glob("*.npz"))
    return [(f.name, f) for f in files]


def _pick_params_for_exp(module: Any, exp_kind: str, dataset_name: str) -> Dict[str, Dict[str, Any]]:
    if exp_kind == "tabarena_nested":
        if hasattr(module, "get_random_search_candidates"):
            xgb_candidates, _ = module.get_random_search_candidates("xgboost")
            mgb_candidates, _ = module.get_random_search_candidates("metagb")
            return {
                "xgboost": dict(xgb_candidates[0]),
                "metagb": dict(mgb_candidates[0]),
            }

        if not hasattr(module, "get_xgb_param_grid") or not hasattr(module, "get_metagb_param_grid"):
            raise RuntimeError("缺少 get_xgb_param_grid/get_metagb_param_grid")

        xgb_grid = module.get_xgb_param_grid(dataset_name)
        mgb_grid = module.get_metagb_param_grid(dataset_name)
        xgb_params = _first_combination(xgb_grid)
        mgb_params = _first_combination(mgb_grid)
        if "pred_feature_fraction" not in mgb_params and hasattr(module, "PRED_FEATURE_FRACTION_VALUES"):
            values = getattr(module, "PRED_FEATURE_FRACTION_VALUES")
            if values:
                mgb_params["pred_feature_fraction"] = values[0]
        return {"xgboost": xgb_params, "metagb": mgb_params}

    if exp_kind == "pred_fraction":
        base = _first_combination(getattr(module, "FIXED_PARAM_GRID"))
        n_est = _safe_first(getattr(module, "N_ESTIMATORS_GRID"))
        xgb_params = dict(base)
        xgb_params.update(getattr(module, "XGBOOST_FIXED_PARAMS"))
        xgb_params["n_estimators"] = n_est

        mgb_params = dict(base)
        mgb_params.update(getattr(module, "METAGB_FIXED_PARAMS"))
        mgb_params["n_estimators"] = n_est
        pfs = getattr(module, "PRED_FRACTIONS")
        mgb_params["pred_feature_fraction"] = _safe_first(pfs)
        return {"xgboost": xgb_params, "metagb": mgb_params}

    if exp_kind == "synthetic_exp2a":
        xgb_grid = module.get_exp2a_xgb_param_grid()
        mgb_grid = module.get_exp2a_metagb_param_grid()
        return {"xgboost": _first_combination(xgb_grid), "metagb": _first_combination(mgb_grid)}

    if exp_kind == "synthetic_exp2b":
        xgb_grid = module.get_exp2b_xgb_param_grid()
        mgb_grid = module.get_exp2b_metagb_param_grid()
        return {"xgboost": _first_combination(xgb_grid), "metagb": _first_combination(mgb_grid)}

    raise ValueError(f"未知 exp_kind: {exp_kind}")


def _load_xy(module: Any, exp_kind: str, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if exp_kind in ("synthetic_exp2a", "synthetic_exp2b"):
        with np.load(dataset_path, allow_pickle=True) as data:
            return data["X"], data["y"]

    categorical = getattr(module, "CATEGORICAL_HANDLING", "drop")
    if hasattr(module, "load_dataset_no_normalization"):
        return module.load_dataset_no_normalization(str(dataset_path), categorical)
    if hasattr(module, "load_dataset"):
        return module.load_dataset(str(dataset_path))
    return _load_dataset_csv_fallback(dataset_path, categorical)


def _build_tasks(
    base_dir: Path,
    exp_files: List[str],
    dataset_filter: Optional[List[str]],
    model_filter: Optional[List[str]],
    search_iterations: int,
    outer_folds: int,
    inner_folds: int,
) -> Tuple[List[Task], Dict[str, Any]]:
    modules: Dict[str, Any] = {}
    tasks: List[Task] = []

    for exp_file in exp_files:
        exp_path = base_dir / exp_file
        module_name = f"_time_estimate_{exp_path.stem}"
        module = _import_module_from_file(module_name, exp_path)
        modules[exp_file] = module

        exp_kind = _detect_exp_kind(exp_file)
        if exp_kind in ("tabarena_nested", "pred_fraction"):
            dataset_names = list(getattr(module, "EXP1_DATASETS"))
            dataset_paths = _resolve_tabarena_dataset_paths(base_dir, dataset_names)
        elif exp_kind == "synthetic_exp2a":
            dataset_paths = _resolve_npz_dataset_paths(base_dir, "exp2A")
        else:
            dataset_paths = _resolve_npz_dataset_paths(base_dir, "exp2B")

        if dataset_filter:
            accepted = set(dataset_filter)
            dataset_paths = [(n, p) for (n, p) in dataset_paths if n in accepted]

        for ds_name, ds_path in dataset_paths:
            params_map = _pick_params_for_exp(module, exp_kind, ds_name)
            for model_type in ["xgboost", "metagb"]:
                if model_filter and model_type not in model_filter:
                    continue
                tasks.append(
                    Task(
                        exp_file=exp_file,
                        exp_kind=exp_kind,
                        dataset_name=ds_name,
                        dataset_path=ds_path,
                        model_type=model_type,
                        params=params_map[model_type],
                        search_iterations=search_iterations,
                        outer_folds=outer_folds,
                        inner_folds=inner_folds,
                    )
                )
    return tasks, modules


def _schedule_time(durations: List[float], workers: int) -> float:
    if not durations:
        return 0.0
    workers = max(1, workers)
    workers = min(workers, len(durations))
    loads = [0.0] * workers
    for duration in sorted(durations, reverse=True):
        idx = min(range(workers), key=lambda i: loads[i])
        loads[idx] += duration
    return max(loads)


def _fmt_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return "-"
    if sec < 60:
        return f"{sec:.4f}s"
    if sec < 3600:
        return f"{sec/60:.2f}min"
    return f"{sec/3600:.2f}h"


def _to_markdown_table(rows: List[Result]) -> str:
    header = (
        "| Exp | Dataset | Model | Samples | Features | SingleTime(s) | TotalTime(s) | 状态 |\n"
        "|---|---|---:|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.exp_file} | {r.dataset_name} | {r.model_type} | "
            f"{r.n_samples if r.n_samples is not None else '-'} | "
            f"{r.n_features if r.n_features is not None else '-'} | "
            f"{r.single_time_sec if r.single_time_sec is not None else '-'} | "
            f"{r.total_time_sec if r.total_time_sec is not None else '-'} | {r.status} |"
        )
    return "\n".join(lines)


def _write_markdown(
    md_path: Path,
    rows: List[Result],
    search_iterations: int,
    outer_folds: int,
    inner_folds: int,
    total_cores: int,
    cores_per_process: int,
) -> None:
    ok_rows = [r for r in rows if r.status == "ok" and r.total_time_sec is not None]
    serial_total = sum(r.total_time_sec for r in ok_rows if r.total_time_sec is not None)
    slots = max(1, total_cores // max(1, cores_per_process))

    by_exp_dataset: Dict[Tuple[str, str], List[Result]] = {}
    for r in ok_rows:
        key = (r.exp_file, r.dataset_name)
        by_exp_dataset.setdefault(key, []).append(r)

    model_parallel_total = 0.0
    dataset_jobs: List[float] = []
    for _, items in by_exp_dataset.items():
        durations = [x.total_time_sec for x in items if x.total_time_sec is not None]
        durations_f = [float(d) for d in durations]
        model_parallel_total += _schedule_time(durations_f, workers=min(slots, len(durations_f)))
        dataset_jobs.append(sum(durations_f))
    dataset_parallel_total = _schedule_time(dataset_jobs, workers=min(slots, len(dataset_jobs)))

    lines: List[str] = []
    lines.append("# 运行时间计算分析")
    lines.append("")
    lines.append("## 统计口径")
    lines.append(f"- SingleTime 定义：单一数据集 + 单一模型 + 单一超参数组合，执行一次 `fit()` 的耗时。")
    lines.append(f"- 推演公式：`TotalTime = SingleTime × {outer_folds} × {inner_folds} × {search_iterations}`。")
    lines.append(f"- 当前折数因子：`{outer_folds}×{inner_folds}={outer_folds*inner_folds}`。")
    lines.append(f"- 进程槽位：`slots = floor({total_cores}/{cores_per_process}) = {slots}`。")
    lines.append("")
    lines.append("## 并行推演公式")
    lines.append("- 按模型并行（同数据集内模型并发）：")
    lines.append("  - `T_dataset_model_parallel ≈ schedule(model_times, workers=min(slots, 模型数))`")
    lines.append("  - `T_project_model_parallel = Σ T_dataset_model_parallel`")
    lines.append("- 按数据集并行（每数据集作为一个进程任务）：")
    lines.append("  - `T_dataset = Σ model_times`")
    lines.append("  - `T_project_dataset_parallel ≈ schedule(dataset_times, workers=min(slots, 数据集数))`")
    lines.append("")
    lines.append("## 每数据集/模型实测与推演")
    lines.append("")
    lines.append(_to_markdown_table(rows))
    lines.append("")
    lines.append("## 项目级时间估算")
    lines.append(f"- 串行总时长：`{serial_total:.4f}s`（{_fmt_seconds(serial_total)}）")
    lines.append(f"- 按模型并行估算：`{model_parallel_total:.4f}s`（{_fmt_seconds(model_parallel_total)}）")
    lines.append(f"- 按数据集并行估算：`{dataset_parallel_total:.4f}s`（{_fmt_seconds(dataset_parallel_total)}）")
    lines.append("")
    lines.append("## 失败任务")
    failed = [r for r in rows if r.status != "ok"]
    if not failed:
        lines.append("- 无")
    else:
        for r in failed:
            lines.append(f"- `{r.exp_file}` / `{r.dataset_name}` / `{r.model_type}`: {r.error}")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="估算当前工程 exp*.py 的 SingleTime 与 TotalTime")
    parser.add_argument("--exp-files", nargs="*", default=None, help="指定 exp 文件（默认自动发现当前目录 exp*.py）")
    parser.add_argument("--dataset", nargs="*", default=None, help="仅统计指定数据集文件名")
    parser.add_argument("--models", nargs="*", default=["xgboost", "metagb"], choices=["xgboost", "metagb"])
    parser.add_argument("--search-iterations", type=int, default=200, help="超参搜索次数（默认200）")
    parser.add_argument("--outer-folds", type=int, default=5, help="外层折数（默认5）")
    parser.add_argument("--inner-folds", type=int, default=4, help="内层折数（默认4）")
    parser.add_argument("--cores-per-process", type=int, default=1, help="每进程绑定核数（默认1）")
    parser.add_argument("--total-cores", type=int, default=16, help="用于并行推演的CPU总核数（默认16）")
    parser.add_argument("--output-md", default="运行时间计算分析.md", help="输出分析Markdown文件名")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base_dir / f"timeAnalysis{timestamp}.log"
    logger = TeeLogger(log_path)

    try:
        _set_thread_env(1)
        _apply_cpu_affinity(args.cores_per_process, logger)
        logger.log(
            f"[环境] 线程限制: "
            + ", ".join([f"{k}={os.environ.get(k)}" for k in THREAD_ENV_KEYS])
        )

        sys.path.insert(0, str(base_dir))
        sys.path.insert(0, str(base_dir / "utils"))
        sys.path.insert(0, str(base_dir / "metagb"))

        if args.exp_files:
            exp_files = args.exp_files
        else:
            exp_files = sorted(
                [
                    p.name
                    for p in base_dir.glob("exp*.py")
                    if p.name != Path(__file__).name
                ]
            )

        if not exp_files:
            logger.log("[错误] 未找到 exp*.py")
            return 1

        logger.log(f"[发现] exp 文件: {exp_files}")
        tasks, modules = _build_tasks(
            base_dir=base_dir,
            exp_files=exp_files,
            dataset_filter=args.dataset,
            model_filter=args.models,
            search_iterations=args.search_iterations,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
        )
        logger.log(f"[任务] 共 {len(tasks)} 个单次拟合任务")

        results: List[Result] = []
        for i, task in enumerate(tasks, start=1):
            module = modules[task.exp_file]
            logger.log(
                f"[{i}/{len(tasks)}] {task.exp_file} | {task.dataset_name} | {task.model_type}"
            )
            if not task.dataset_path.exists():
                err = f"数据不存在: {task.dataset_path}"
                logger.log(f"  -> 跳过: {err}")
                results.append(
                    Result(
                        exp_file=task.exp_file,
                        exp_kind=task.exp_kind,
                        dataset_name=task.dataset_name,
                        dataset_path=str(task.dataset_path),
                        model_type=task.model_type,
                        n_samples=None,
                        n_features=None,
                        single_time_sec=None,
                        total_time_sec=None,
                        status="missing_data",
                        error=err,
                    )
                )
                continue

            try:
                x, y = _load_xy(module, task.exp_kind, task.dataset_path)
                single_time = _measure_single_fit(
                    model_type=task.model_type,
                    params=task.params,
                    x=x,
                    y=y,
                    seed=int(getattr(module, "RANDOM_SEED", 42)),
                )
                total_time = (
                    single_time
                    * float(task.outer_folds)
                    * float(task.inner_folds)
                    * float(task.search_iterations)
                )
                logger.log(
                    f"  -> samples={len(x)}, features={x.shape[1]}, "
                    f"SingleTime={single_time:.6f}s, TotalTime={total_time:.2f}s"
                )
                results.append(
                    Result(
                        exp_file=task.exp_file,
                        exp_kind=task.exp_kind,
                        dataset_name=task.dataset_name,
                        dataset_path=str(task.dataset_path),
                        model_type=task.model_type,
                        n_samples=int(len(x)),
                        n_features=int(x.shape[1]),
                        single_time_sec=float(single_time),
                        total_time_sec=float(total_time),
                        status="ok",
                        error="",
                    )
                )
            except Exception as exc:
                logger.log(f"  -> 失败: {exc}")
                results.append(
                    Result(
                        exp_file=task.exp_file,
                        exp_kind=task.exp_kind,
                        dataset_name=task.dataset_name,
                        dataset_path=str(task.dataset_path),
                        model_type=task.model_type,
                        n_samples=None,
                        n_features=None,
                        single_time_sec=None,
                        total_time_sec=None,
                        status="error",
                        error=str(exc),
                    )
                )

        md_path = base_dir / args.output_md
        _write_markdown(
            md_path=md_path,
            rows=results,
            search_iterations=args.search_iterations,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            total_cores=args.total_cores,
            cores_per_process=args.cores_per_process,
        )

        ok_cnt = len([r for r in results if r.status == "ok"])
        logger.log("")
        logger.log(f"[完成] 成功 {ok_cnt}/{len(results)}")
        logger.log(f"[输出] 分析文档: {md_path}")
        logger.log(f"[输出] 运行日志: {log_path}")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

