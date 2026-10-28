#!/usr/bin/env python3
"""Plan or execute the frozen E0/LeafOnly queue, one fit per isolated process."""
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


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xgboost-python-package", required=True, type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers not in range(1, 17) or args.timeout < 1: parser.error("Use 1..16 workers and a positive timeout")
    jobs = []
    for item in json.loads(args.queue.read_text()):
        path = FORMAL / item["task"]
        task = json.loads(path.read_text())
        if args.scenarios and task["scenario"] not in args.scenarios: continue
        jobs.append({**item, "task_id": task["task_id"], "task_sha256": digest(path)})
    if not jobs: parser.error("No jobs selected")
    plan = {"jobs": jobs, "model_fit_calls": len(jobs),
        "crossgb_fit_calls": sum(j["variant"] != "xgb" for j in jobs),
        "workers": args.workers, "timeout_sec": args.timeout, "execute": args.execute}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if platform.system() != "Linux": raise RuntimeError("Formal CGB execution is reserved for Linux")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "queue_plan.json").write_text(json.dumps(plan, indent=2))
    # Resolve the actual runtime library before trusting any resumed fit.
    sys.path.insert(0, str(args.xgboost_python_package.resolve()))
    import xgboost as xgb
    library_hash = digest(Path(xgb.core._LIB._name))
    driver = FORMAL / "scripts/run_cgb_task.py"
    from run_cgb_task import current_source_hashes, helper_hashes
    source_hashes = current_source_hashes()
    construction_hashes = helper_hashes(FORMAL.parent / "rebuttal_execution_20261025/source/exp")
    def run(job):
        output = args.output / (job["task_id"] + "__" + job["variant"])
        previous = output / "result.json"
        if args.resume and previous.exists():
            result = json.loads(previous.read_text())
            if (result.get("status") == "success" and result.get("task_file_sha256") == job["task_sha256"]
                and result.get("native_library_sha256") == library_hash and result.get("driver_sha256") == digest(driver)):
                if result.get("controlled_source") != source_hashes or result.get("helper_hashes") != construction_hashes:
                    raise RuntimeError("Resume source identity changed: " + str(output))
                if digest(output / "model.ubj") != result["model_sha256"] or digest(output / "predictions.npz") != result["prediction_sha256"]:
                    raise RuntimeError("Resume model/prediction artifacts changed: " + str(output))
                return {"job": job["task_id"], "variant": job["variant"], "status": "cached_success"}
            raise RuntimeError("Resume identity mismatch: " + str(output))
        if output.exists(): raise FileExistsError("Incomplete task retained; inspect and choose a new result directory: " + str(output))
        command = [sys.executable, str(driver), "--task", str(FORMAL / job["task"]), "--variant", job["variant"],
            "--xgboost-python-package", str(args.xgboost_python_package), "--output", str(output)]
        if args.gate: command += ["--gate", str(args.gate)]
        with (args.output / (output.name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
                status = "success" if code == 0 else f"failed_exit_{code}"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(); status = "timeout"
        return {"job": job["task_id"], "variant": job["variant"], "status": status}
    with ThreadPoolExecutor(max_workers=args.workers) as pool: records = list(pool.map(run, jobs))
    (args.output / "queue_status.json").write_text(json.dumps(records, indent=2))
    failed = sum(r["status"] not in {"success", "cached_success"} for r in records)
    print(json.dumps({"tasks": len(records), "failed_or_timed_out": failed}))
    raise SystemExit(2 if failed else 0)


if __name__ == "__main__": main()
