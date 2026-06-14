#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[2]


def add_dataset_args(parser: argparse.ArgumentParser, task_type: str) -> None:
    parser.add_argument(
        f"--{task_type}-datasets",
        nargs="+",
        default=["all"],
        help=f"{task_type} dataset filenames to run, or all.",
    )
    parser.add_argument(
        f"--skip-{task_type}-datasets",
        nargs="*",
        default=[],
        help=f"{task_type} dataset filenames to skip.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Server launcher for Random600 current_metagb experiments. "
            "Defaults target the 16-core CPU full run: 5 outer folds, 4 inner folds, 600 candidates."
        )
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        choices=["regression", "binary", "multiclass"],
        default=["regression", "binary", "multiclass"],
        help="Task types to run on this server.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xgb_cpu", "current_metagb", "current_metagb_pred0"],
        default=["xgb_cpu", "current_metagb", "current_metagb_pred0"],
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--run-all-outer-folds",
        dest="run_all_outer_folds",
        action="store_true",
        default=True,
        help="Run all outer folds. This is the default for paper experiments.",
    )
    parser.add_argument(
        "--single-outer-fold",
        dest="run_all_outer_folds",
        action="store_false",
        help="Run only --outer-fold-index for smoke/debug runs.",
    )
    parser.add_argument("--outer-fold-index", type=int, default=0)
    parser.add_argument("--task-workers", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categorical-handling", choices=["drop", "onehot", "ordinal"], default="ordinal")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--experiment-name",
        default="random600",
        help="Prefix used when auto-generating run-id and log filenames.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Run-level log file. Defaults to random600/nohup_logs/<run_id>.log.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not redirect the delegated experiment output to a run-level log file.",
    )
    parser.add_argument("--output-root", default=str(ROOT / "random600"))
    parser.add_argument("--xgboost-python-package", default=str(ROOT / "beforereview" / "xgboost-master" / "python-package"))
    parser.add_argument("--xgb-config", default=None, help="Deprecated compatibility option; ignored by paired candidate generation.")
    parser.add_argument("--search-space-json", default=str(ROOT / "random600" / "configs" / "default_search_space.json"))
    parser.add_argument("--regression-dir", default=str(ROOT / "dataset" / "TabArena_regression"))
    parser.add_argument("--binary-dir", default=str(ROOT / "dataset" / "TabArena_binary"))
    parser.add_argument("--multiclass-dir", default=str(ROOT / "dataset" / "TabArena_multiclass"))
    add_dataset_args(parser, "regression")
    add_dataset_args(parser, "binary")
    add_dataset_args(parser, "multiclass")
    parser.add_argument("--skip-datasets", nargs="*", default=[], help="Dataset filenames to skip across all task types.")
    parser.add_argument("--binary-objective", choices=["binary:logistic", "binary:logitraw"], default="binary:logistic")
    parser.add_argument("--multiclass-objective", choices=["multi:softprob", "multi:softmax"], default="multi:softprob")
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--metagb-lambda-l2", type=float, default=1.0)
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2], help="Fallback for candidates without metagb_score_agg.")
    parser.add_argument("--metagb-pred-value-type", choices=["bin", "raw", "standardized_raw"], default="raw")
    parser.add_argument("--metagb-hist-subtract", action="store_true")
    parser.add_argument("--no-metagb-hist-subtract", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command without running it.")
    return parser


def extend_if_nonempty(cmd: List[str], flag: str, values: List[str]) -> None:
    if values:
        cmd.append(flag)
        cmd.extend(values)


def task_tag(task_types: List[str]) -> str:
    if task_types == ["regression", "binary", "multiclass"]:
        return "all"
    return "-".join(task_types)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_id(args: argparse.Namespace) -> str:
    run_id = args.run_id
    if run_id is None:
        run_id = f"{args.experiment_name}_{task_tag(args.task_types)}_{timestamp()}"
    return run_id


def build_command(args: argparse.Namespace, run_id: str) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "random600" / "scripts" / "exp1_random600_all.py"),
        "--mode",
        args.mode,
        "--task-types",
        *args.task_types,
        "--models",
        *args.models,
        "--max-candidates",
        str(args.max_candidates),
        "--outer-folds",
        str(args.outer_folds),
        "--inner-folds",
        str(args.inner_folds),
        "--task-workers",
        str(args.task_workers),
        "--n-jobs",
        str(args.n_jobs),
        "--progress-every",
        str(args.progress_every),
        "--seed",
        str(args.seed),
        "--categorical-handling",
        args.categorical_handling,
        "--run-id",
        run_id,
        "--output-root",
        args.output_root,
        "--xgboost-python-package",
        args.xgboost_python_package,
        "--search-space-json",
        args.search_space_json,
        "--regression-dir",
        args.regression_dir,
        "--binary-dir",
        args.binary_dir,
        "--multiclass-dir",
        args.multiclass_dir,
        "--binary-objective",
        args.binary_objective,
        "--multiclass-objective",
        args.multiclass_objective,
        "--xgb-reg-lambda",
        str(args.xgb_reg_lambda),
        "--metagb-lambda-l2",
        str(args.metagb_lambda_l2),
        "--metagb-score-agg",
        str(args.metagb_score_agg),
        "--metagb-pred-value-type",
        args.metagb_pred_value_type,
    ]
    if args.xgb_config:
        cmd.extend(["--xgb-config", args.xgb_config])
    if args.run_all_outer_folds:
        cmd.append("--run-all-outer-folds")
    else:
        cmd.extend(["--outer-fold-index", str(args.outer_fold_index)])
    if args.metagb_hist_subtract:
        cmd.append("--metagb-hist-subtract")
    if args.no_metagb_hist_subtract:
        cmd.append("--no-metagb-hist-subtract")

    extend_if_nonempty(cmd, "--regression-datasets", args.regression_datasets)
    extend_if_nonempty(cmd, "--binary-datasets", args.binary_datasets)
    extend_if_nonempty(cmd, "--multiclass-datasets", args.multiclass_datasets)
    extend_if_nonempty(cmd, "--skip-datasets", args.skip_datasets)
    extend_if_nonempty(cmd, "--skip-regression-datasets", args.skip_regression_datasets)
    extend_if_nonempty(cmd, "--skip-binary-datasets", args.skip_binary_datasets)
    extend_if_nonempty(cmd, "--skip-multiclass-datasets", args.skip_multiclass_datasets)
    return cmd


def resolve_log_file(args: argparse.Namespace, run_id: str) -> Optional[Path]:
    if args.no_log_file:
        return None
    if args.log_file:
        return Path(args.log_file).expanduser().resolve()
    return Path(args.output_root).expanduser().resolve() / "nohup_logs" / f"{run_id}.log"


def write_launch_header(handle: object, args: argparse.Namespace, cmd: List[str], run_id: str) -> None:
    print(f"launch_time: {datetime.now().isoformat(timespec='seconds')}", file=handle)
    print(f"run_id: {run_id}", file=handle)
    print(f"task_types: {' '.join(args.task_types)}", file=handle)
    print(f"models: {' '.join(args.models)}", file=handle)
    print(f"max_candidates: {args.max_candidates}", file=handle)
    print(f"outer_folds: {args.outer_folds}", file=handle)
    print(f"inner_folds: {args.inner_folds}", file=handle)
    print(f"task_workers: {args.task_workers}", file=handle)
    print(f"n_jobs: {args.n_jobs}", file=handle)
    print(f"regression_datasets: {' '.join(args.regression_datasets)}", file=handle)
    print(f"binary_datasets: {' '.join(args.binary_datasets)}", file=handle)
    print(f"multiclass_datasets: {' '.join(args.multiclass_datasets)}", file=handle)
    print(f"skip_datasets: {' '.join(args.skip_datasets)}", file=handle)
    print(f"skip_regression_datasets: {' '.join(args.skip_regression_datasets)}", file=handle)
    print(f"skip_binary_datasets: {' '.join(args.skip_binary_datasets)}", file=handle)
    print(f"skip_multiclass_datasets: {' '.join(args.skip_multiclass_datasets)}", file=handle)
    print("delegated_command:", file=handle)
    print(" ".join(cmd), file=handle)
    print("", file=handle)


def main() -> int:
    args = build_parser().parse_args()
    if args.max_candidates % 30 != 0 and "current_metagb" in args.models:
        raise ValueError("--max-candidates must be divisible by 30 when current_metagb is enabled.")

    run_id = resolve_run_id(args)
    cmd = build_command(args, run_id)
    log_file = resolve_log_file(args, run_id)
    env = os.environ.copy()
    python_path = str(Path(args.xgboost_python_package).expanduser().resolve())
    env["PYTHONPATH"] = python_path if not env.get("PYTHONPATH") else f"{python_path}{os.pathsep}{env['PYTHONPATH']}"

    print(f"run_id: {run_id}")
    if log_file is not None:
        print(f"run_log: {log_file}")
    print("Delegated command:")
    print(" ".join(cmd))
    print(f"PYTHONPATH={env['PYTHONPATH']}")
    if args.dry_run:
        return 0

    if log_file is None:
        return subprocess.call(cmd, cwd=str(ROOT), env=env)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        write_launch_header(handle, args, cmd, run_id)
        handle.flush()
        return subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    raise SystemExit(main())
