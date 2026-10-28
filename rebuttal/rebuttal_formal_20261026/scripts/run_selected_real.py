#!/usr/bin/env python3
"""Frozen CPU/Kin selected refits and descriptive leaf export; no new search."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

FORMAL = Path(__file__).resolve().parents[1]
for key in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[key] = "1"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgboost-python-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers not in range(1, 17) or args.timeout < 1: parser.error("Use 1..16 workers and a positive timeout")
    jobs = json.loads((FORMAL / "configs/N4_selected_queue.json").read_text())
    if not args.execute:
        print(json.dumps({"model_fit_calls": len(jobs), "crossgb_fit_calls": 10, "jobs": jobs}, indent=2)); return
    if platform.system() != "Linux": raise RuntimeError("Formal selected refits belong on Linux")
    if not args.gate: parser.error("Execution requires --gate R0_gate.json")
    sys.path.insert(0, str(args.xgboost_python_package.resolve()))
    import xgboost as xgb
    from run_cgb_task import current_source_hashes, helper_hashes
    gate = json.loads(args.gate.read_text())
    if (not gate.get("passed") or gate.get("native_library_sha256") != digest(Path(xgb.core._LIB._name))
        or gate.get("controlled_source") != current_source_hashes()
        or gate.get("helper_hashes") != helper_hashes(FORMAL.parent / "rebuttal_execution_20261025/source/exp")):
        raise RuntimeError("R0 gate does not match the runtime/source")
    args.output.mkdir(parents=True, exist_ok=False)
    def run(job):
        output = args.output / f"{job['dataset']}__{job['method']}__outer{job['outer_fold']}"
        command = [sys.executable, str(FORMAL / "scripts/run_real_fit.py"), "--dataset", job["dataset"],
            "--method", job["method"], "--outer-fold", str(job["outer_fold"]), "--params-file", str(FORMAL / job["params_file"]),
            "--xgboost-python-package", str(args.xgboost_python_package), "--output", str(output), "--mode", "final", "--diagnostics"]
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
                status = "success" if code == 0 else f"failed_exit_{code}"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(); status = "timeout"
        result = json.loads((output / "result.json").read_text()) if status == "success" else None
        return {**job, "status": status, "new_test_rmse": result["rmse"] if result else None,
            "rmse_difference_from_archive": result["rmse"] - job["archived_test_rmse"] if result else None,
            "structure": result.get("structure") if result else None,
            "scope": "frozen-parameter diagnostic refit; do not call it the original archived trained model"}
    with ThreadPoolExecutor(max_workers=args.workers) as pool: results = list(pool.map(run, jobs))
    (args.output / "selected_replay_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"jobs": len(results), "successful": sum(r["status"] == "success" for r in results)}))
    raise SystemExit(0 if all(r["status"] == "success" for r in results) else 2)


if __name__ == "__main__": main()
