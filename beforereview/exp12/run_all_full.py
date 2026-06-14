#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_NAMES = [
    "exp1_ordinal",
    "exp1_drop",
    "exp2_ordinal",
    "exp2_onehot",
    "exp2_drop",
]


def project_root() -> Path:
    raw = os.environ.get("METAGB_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def output_root(root: Path) -> Path:
    raw = os.environ.get("METAGB_OUTPUT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return root / "exp12"


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def latest_run_dir(logs_root: Path) -> Optional[Path]:
    if not logs_root.exists():
        return None
    candidates = sorted(path for path in logs_root.glob("run_all_full_*") if path.is_dir())
    return candidates[-1] if candidates else None


def load_jobs(run_dir: Path) -> List[Dict[str, Any]]:
    jobs_json = run_dir / "jobs.json"
    if not jobs_json.exists():
        return []
    with jobs_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_child_command(args: argparse.Namespace, root: Path, script_name: str) -> List[str]:
    child_run_id = f"{args.run_id}__{script_name}" if getattr(args, "run_id", None) else script_name
    command = [
        args.python,
        str(root / "exp12" / "scripts" / f"{script_name}.py"),
        "--mode",
        "full",
        "--datasets",
        "all",
        "--run-all-outer-folds",
        "--max-candidates",
        str(args.max_candidates),
        "--outer-folds",
        str(args.outer_folds),
        "--inner-folds",
        str(args.inner_folds),
        "--models",
        *args.models,
        "--n-jobs",
        str(args.n_jobs),
        "--task-workers",
        str(args.task_workers),
        "--progress-every",
        str(args.progress_every),
        "--run-id",
        child_run_id,
    ]
    if args.metagb_score_agg is not None:
        command.extend(["--metagb-score-agg", str(args.metagb_score_agg)])
    if args.metagb_lambda_l2 is not None:
        command.extend(["--metagb-lambda-l2", str(args.metagb_lambda_l2)])
    if args.xgb_reg_lambda is not None:
        command.extend(["--xgb-reg-lambda", str(args.xgb_reg_lambda)])
    if args.metagb_cond_pred_share:
        command.append("--metagb-cond-pred-share")
    if args.metagb_leaf_only_use_bias:
        command.append("--metagb-leaf-only-use-bias")
    if not args.metagb_hist_subtract:
        command.append("--no-metagb-hist-subtract")
    if args.limit_grid_candidates:
        command.append("--limit-grid-candidates")
    return command


def start(args: argparse.Namespace) -> int:
    root = project_root()
    out_root = output_root(root)
    run_id = args.run_id or f"run_all_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.run_id = run_id
    run_dir = out_root / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base.setdefault("METAGB_PROJECT_ROOT", str(root))
    env_base.setdefault("METAGB_XGBOOST_PYTHON_PACKAGE", str(root / "xgboost-master" / "python-package"))
    env_base.setdefault("METAGB_DATASET_DIR", str(root / "dataset" / "tabArena_regression"))
    env_base.setdefault("METAGB_OUTPUT_ROOT", str(out_root))
    env_base.setdefault("PYTHONUNBUFFERED", "1")

    config = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "output_root": str(out_root),
        "python": args.python,
        "execution": "serial_scripts",
        "task_workers_per_script": args.task_workers,
        "model_n_jobs_per_task": args.n_jobs,
        "max_candidates": args.max_candidates,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "models": args.models,
        "limit_grid_candidates": args.limit_grid_candidates,
        "metagb_hist_subtract": args.metagb_hist_subtract,
        "metagb_xgboost_python_package": env_base["METAGB_XGBOOST_PYTHON_PACKAGE"],
        "metagb_dataset_dir": env_base["METAGB_DATASET_DIR"],
    }
    requested_threads = max(1, int(args.task_workers)) * max(1, int(args.n_jobs))
    cpu_count = os.cpu_count() or 1
    if requested_threads > cpu_count:
        config["oversubscription_warning"] = (
            f"task_workers * n_jobs = {requested_threads} exceeds detected CPU count {cpu_count}; "
            "prefer more task workers with n_jobs=1 for exp12 full runs."
        )
    write_json(run_dir / "run_config.json", config)

    if args.wait:
        return run_queue(args, root, run_dir, env_base)

    supervisor_log = run_dir / "supervisor.stdout.log"
    supervisor_command = [
        args.python,
        str(root / "exp12" / "run_all_full.py"),
        "__worker",
        "--run-id",
        run_id,
        "--python",
        args.python,
        "--n-jobs",
        str(args.n_jobs),
        "--task-workers",
        str(args.task_workers),
        "--max-candidates",
        str(args.max_candidates),
        "--outer-folds",
        str(args.outer_folds),
        "--inner-folds",
        str(args.inner_folds),
        "--progress-every",
        str(args.progress_every),
        "--models",
        *args.models,
        "--metagb-score-agg",
        str(args.metagb_score_agg),
        "--metagb-lambda-l2",
        str(args.metagb_lambda_l2),
        "--xgb-reg-lambda",
        str(args.xgb_reg_lambda),
    ]
    if args.metagb_cond_pred_share:
        supervisor_command.append("--metagb-cond-pred-share")
    if args.metagb_leaf_only_use_bias:
        supervisor_command.append("--metagb-leaf-only-use-bias")
    if not args.metagb_hist_subtract:
        supervisor_command.append("--no-metagb-hist-subtract")
    if args.limit_grid_candidates:
        supervisor_command.append("--limit-grid-candidates")
    if args.continue_on_failure:
        supervisor_command.append("--continue-on-failure")

    with supervisor_log.open("ab") as handle:
        supervisor = subprocess.Popen(
            supervisor_command,
            cwd=str(root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env_base,
            start_new_session=True,
        )
    write_json(
        run_dir / "supervisor.json",
        {
            "pid": supervisor.pid,
            "log": str(supervisor_log),
            "command": supervisor_command,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"started serial supervisor: pid={supervisor.pid}, log={supervisor_log}")
    print(f"run_id: {run_id}")
    print(f"run_dir: {run_dir}")
    print(f"status: {args.python} exp12/run_all_full.py status --run-id {run_id}")
    print(f"stop:   {args.python} exp12/run_all_full.py stop --run-id {run_id}")
    return 0


def run_queue(args: argparse.Namespace, root: Path, run_dir: Path, env_base: Dict[str, str]) -> int:
    job_records: List[Dict[str, Any]] = []
    exit_code = 0
    for script_name in SCRIPT_NAMES:
        log_path = run_dir / f"{script_name}.stdout.log"
        command = build_child_command(args, root, script_name)
        child_env = env_base.copy()
        child_env["OMP_NUM_THREADS"] = str(args.n_jobs)
        child_env["OPENBLAS_NUM_THREADS"] = str(args.n_jobs)
        child_env["MKL_NUM_THREADS"] = str(args.n_jobs)
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )
        job_records.append(
            {
                "script": script_name,
                "experiment_run_id": f"{args.run_id}__{script_name}",
                "pid": process.pid,
                "task_workers": args.task_workers,
                "n_jobs": args.n_jobs,
                "log": str(log_path),
                "experiment_log_dir": str(output_root(root) / "logs" / f"{args.run_id}__{script_name}"),
                "experiment_results_dir": str(output_root(root) / "results" / f"{args.run_id}__{script_name}"),
                "command": command,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "status": "running",
            }
        )
        write_json(run_dir / "jobs.json", job_records)
        print(
            f"started {script_name}: pid={process.pid}, task_workers={args.task_workers}, "
            f"model_n_jobs={args.n_jobs}, log={log_path}",
            flush=True,
        )
        code = process.wait()
        job_records[-1]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job_records[-1]["exit_code"] = code
        job_records[-1]["status"] = "ok" if code == 0 else "failed"
        write_json(run_dir / "jobs.json", job_records)
        if code != 0:
            print(f"failed {script_name}: pid={process.pid}, exit_code={code}", flush=True)
            exit_code = code
            if not args.continue_on_failure:
                break
        else:
            print(f"finished {script_name}: pid={process.pid}", flush=True)

    write_json(
        run_dir / "finished.json",
        {"finished_at": datetime.now().isoformat(timespec="seconds"), "exit_code": exit_code},
    )
    return exit_code


def resolve_run_dir(args: argparse.Namespace) -> Path:
    root = project_root()
    logs_root = output_root(root) / "logs"
    if args.run_id:
        return logs_root / args.run_id
    latest = latest_run_dir(logs_root)
    if latest is None:
        raise FileNotFoundError(f"No run_all_full_* directory found under {logs_root}")
    return latest


def status(args: argparse.Namespace) -> int:
    try:
        run_dir = resolve_run_dir(args)
    except FileNotFoundError as exc:
        print(str(exc))
        print("No full-run logs exist yet. Start one with: python exp12/run_all_full.py start")
        return 0
    print(f"run_dir: {run_dir}")
    jobs = load_jobs(run_dir)
    supervisor = load_json(run_dir / "supervisor.json")
    if supervisor:
        pid = int(supervisor["pid"])
        state = "running" if is_running(pid) else "finished/stopped"
        print(f"supervisor: {state}, pid={pid}, log={supervisor['log']}")
    if not jobs:
        print("No jobs.json found yet. The supervisor may still be starting the first script.")
        return 0
    running = 0
    for job in jobs:
        pid = int(job["pid"])
        active = is_running(pid)
        running += int(active)
        state = job.get("status") or ("running" if active else "finished/stopped")
        print(
            f"{job['script']}: {state}, pid={pid}, task_workers={job.get('task_workers')}, "
            f"model_n_jobs={job['n_jobs']}, log={job['log']}"
        )
        experiment_log_dir_raw = job.get("experiment_log_dir")
        if experiment_log_dir_raw:
            experiment_log_dir = Path(str(experiment_log_dir_raw))
            print(f"  experiment_logs: {experiment_log_dir}")
            print(f"  experiment_results: {job.get('experiment_results_dir')}")
            task_status_payload = load_json(experiment_log_dir / "task_status.json")
            if task_status_payload and isinstance(task_status_payload.get("tasks"), dict):
                task_values = list(task_status_payload["tasks"].values())
                counts: Dict[str, int] = {}
                for task_status in task_values:
                    status_name = str(task_status.get("status", "unknown"))
                    counts[status_name] = counts.get(status_name, 0) + 1
                counts_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
                finished = task_status_payload.get("finished", 0)
                total = task_status_payload.get("total", len(task_values))
                print(f"  task_progress: {finished}/{total}; {counts_text}")
                task_logs = [
                    Path(str(task_status.get("log")))
                    for task_status in task_values
                    if task_status.get("log") and Path(str(task_status.get("log"))).exists()
                ]
                recent_logs = sorted(task_logs, key=lambda path: path.stat().st_mtime, reverse=True)[:3]
                if recent_logs:
                    print("  recent_task_logs:")
                    for path in recent_logs:
                        print(f"    {path}")
    print(f"running_jobs: {running}/{len(jobs)}")
    return 0


def stop(args: argparse.Namespace) -> int:
    try:
        run_dir = resolve_run_dir(args)
    except FileNotFoundError as exc:
        print(str(exc))
        print("No full-run logs exist yet.")
        return 0
    jobs = load_jobs(run_dir)
    sig = signal.SIGKILL if args.kill else signal.SIGTERM
    supervisor_json = run_dir / "supervisor.json"
    if supervisor_json.exists():
        with supervisor_json.open("r", encoding="utf-8") as handle:
            supervisor = json.load(handle)
        pid = int(supervisor["pid"])
        if is_running(pid):
            try:
                os.killpg(pid, sig)
                print(f"sent {sig.name} to supervisor process group pid={pid}")
            except ProcessLookupError:
                pass
    for job in jobs:
        pid = int(job["pid"])
        if not is_running(pid):
            continue
        try:
            os.killpg(pid, sig)
            print(f"sent {sig.name} to process group pid={pid} ({job['script']})")
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"permission denied stopping pid={pid}: {exc}")
    write_json(
        run_dir / "stop_requested.json",
        {"stopped_at": datetime.now().isoformat(timespec="seconds"), "signal": sig.name},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch all exp12 full benchmarks on macOS/Linux.")
    subparsers = parser.add_subparsers(dest="command", metavar="{start,status,stop}")

    start_parser = subparsers.add_parser("start", help="start all five experiments in background")
    start_parser.add_argument("--python", default=os.environ.get("EXP12_PYTHON_BIN", sys.executable))
    start_parser.add_argument("--task-workers", type=int, default=env_int("EXP12_TASK_WORKERS", 16))
    start_parser.add_argument("--n-jobs", type=int, default=env_int("EXP12_MODEL_N_JOBS", 1))
    start_parser.add_argument("--max-candidates", type=int, default=env_int("EXP12_MAX_CANDIDATES", 200))
    start_parser.add_argument("--outer-folds", type=int, default=env_int("EXP12_OUTER_FOLDS", 5))
    start_parser.add_argument("--inner-folds", type=int, default=env_int("EXP12_INNER_FOLDS", 4))
    start_parser.add_argument("--progress-every", type=int, default=env_int("EXP12_PROGRESS_EVERY", 10))
    start_parser.add_argument("--models", nargs="+", default=os.environ.get("EXP12_MODELS", "current_metagb xgb_cpu").split())
    start_parser.add_argument("--run-id", default=os.environ.get("EXP12_RUN_ID"))
    start_parser.add_argument("--wait", action="store_true", help="run the serial queue in foreground")
    start_parser.add_argument("--continue-on-failure", action="store_true")
    start_parser.add_argument("--limit-grid-candidates", action="store_true", help="Debug only: apply --max-candidates to Exp2 grid search")
    start_parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2, 3])
    start_parser.add_argument("--metagb-lambda-l2", type=float, default=0.0)
    start_parser.add_argument("--xgb-reg-lambda", type=float, default=0.0)
    start_parser.add_argument("--metagb-cond-pred-share", action="store_true")
    start_parser.add_argument("--metagb-leaf-only-use-bias", action="store_true")
    start_parser.add_argument("--metagb-hist-subtract", dest="metagb_hist_subtract", action="store_true", default=True, help="Enable current MetaGB parent-sibling histogram subtraction cache")
    start_parser.add_argument("--no-metagb-hist-subtract", dest="metagb_hist_subtract", action="store_false", help="Disable current MetaGB parent-sibling histogram subtraction cache")
    start_parser.set_defaults(func=start)

    status_parser = subparsers.add_parser("status", help="show latest or selected run status")
    status_parser.add_argument("--run-id", default=None)
    status_parser.set_defaults(func=status)

    stop_parser = subparsers.add_parser("stop", help="stop latest or selected run")
    stop_parser.add_argument("--run-id", default=None)
    stop_parser.add_argument("--kill", action="store_true", help="use SIGKILL instead of SIGTERM")
    stop_parser.set_defaults(func=stop)

    return parser


def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("__worker")
    parser.add_argument("--python", required=True)
    parser.add_argument("--task-workers", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--outer-folds", type=int, required=True)
    parser.add_argument("--inner-folds", type=int, required=True)
    parser.add_argument("--progress-every", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metagb-score-agg", type=int, required=True)
    parser.add_argument("--metagb-lambda-l2", type=float, required=True)
    parser.add_argument("--xgb-reg-lambda", type=float, required=True)
    parser.add_argument("--metagb-cond-pred-share", action="store_true")
    parser.add_argument("--metagb-leaf-only-use-bias", action="store_true")
    parser.add_argument("--metagb-hist-subtract", dest="metagb_hist_subtract", action="store_true", default=True)
    parser.add_argument("--no-metagb-hist-subtract", dest="metagb_hist_subtract", action="store_false")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--limit-grid-candidates", action="store_true")
    return parser


def run_worker_from_argv() -> int:
    worker_args = build_worker_parser().parse_args()
    root = project_root()
    return run_queue(
        worker_args,
        root,
        output_root(root) / "logs" / worker_args.run_id,
        {
            **os.environ.copy(),
            "METAGB_PROJECT_ROOT": os.environ.get("METAGB_PROJECT_ROOT", str(root)),
            "METAGB_XGBOOST_PYTHON_PACKAGE": os.environ.get(
                "METAGB_XGBOOST_PYTHON_PACKAGE",
                str(root / "xgboost-master" / "python-package"),
            ),
            "METAGB_DATASET_DIR": os.environ.get(
                "METAGB_DATASET_DIR",
                str(root / "dataset" / "tabArena_regression"),
            ),
            "METAGB_OUTPUT_ROOT": os.environ.get("METAGB_OUTPUT_ROOT", str(output_root(root))),
        },
    )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "__worker":
        return run_worker_from_argv()
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        args = parser.parse_args(["start"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
