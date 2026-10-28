#!/usr/bin/env python3
"""Frozen train/validation-only runtime pilot. No outer test arrays are read."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "rebuttal_baselines_20261025")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers not in range(1, 5) or args.timeout < 1:
        parser.error("Use 1..4 pilot workers and a positive timeout")
    base = args.baseline_root.resolve()
    sys.path.insert(0, str(base))
    from baselines.provenance import adapter_fingerprint, source_record
    jobs = []
    for dataset in ["cpu_activity", "regime_slope"]:
        for method in ["gbdt_pl", "linxgboost", "pilot", "tao"]:
            config = base / f"configs/{method}_300.json"
            params = json.loads(config.read_text())
            for index in ([1, 101, 201] if method in {"gbdt_pl", "linxgboost"} else [1]):
                jobs.append({"id": f"{dataset}__{method}__candidate{index}", "dataset": dataset,
                    "method": method, "candidate_index": index, "params": params[index - 1],
                    "candidate_file_sha256": digest(config),
                    "input_file_sha256": digest(base / f"inputs/{dataset}/data.npz"),
                    "source": source_record(method)})
    plan = {"purpose": "runtime/memory only; no method or hyperparameter selection using test performance",
        "test_access": False, "fits": len(jobs), "crossgb_fit_calls": 0,
        "workers": args.workers, "timeout_per_fit_sec": args.timeout,
        "adapter_fingerprint": adapter_fingerprint(), "jobs": jobs}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if platform.system() != "Linux":
        raise RuntimeError("Formal runtime pilot is reserved for the authorized Linux server")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    import numpy as np
    inputs = {}
    for dataset in ["cpu_activity", "regime_slope"]:
        manifest = json.loads((base / f"inputs/{dataset}/manifest.json").read_text())
        fold = manifest["outer_folds"][0]["inner"][0]
        path = args.output / f"{dataset}_outer1_inner1.npz"
        # Loading only X/y and the two predeclared INNER index arrays; outer test is never accessed.
        with np.load(base / f"inputs/{dataset}/data.npz", allow_pickle=False) as data:
            X, y = data["X"], data["y"]
            train, validation = data[fold["train"]], data[fold["validation"]]
            assert not np.intersect1d(train, validation).size
            np.savez_compressed(path, X_train=X[train], y_train=y[train], X_predict=X[validation])
        inputs[dataset] = path
    def run(job):
        destination = args.output / job["id"]
        memory = args.output / (job["id"] + ".memory.txt")
        command = ["/usr/bin/time", "-v", "-o", str(memory), sys.executable,
            str(base / "scripts/run_task.py"), "--method", job["method"], "--input", str(inputs[job["dataset"]]),
            "--output", str(destination), "--params", json.dumps(job["params"]),
            "--seed", "271828" if job["dataset"] == "regime_slope" else "42"]
        start = time.monotonic()
        with (args.output / (job["id"] + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
                status = "success" if code == 0 else f"failed_exit_{code}"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = "timeout"
        record = {**job, "status": status, "process_wall_sec": time.monotonic() - start}
        result = destination / "result.json"
        if result.exists():
            record["fit_report"] = json.loads(result.read_text())
        if memory.exists():
            match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", memory.read_text())
            if match: record["peak_rss_kib"] = int(match[1])
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run, jobs))
    report = {"status": "pilot_complete_with_censoring" if any(r["status"] != "success" for r in records) else "pilot_complete",
        "records": records, "crossgb_fit_calls": 0,
        "limitation": "Three sampled candidates do not establish full-search runtime; timeout is a lower bound, never a successful timing"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "fits": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
