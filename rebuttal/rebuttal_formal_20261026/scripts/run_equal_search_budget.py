#!/usr/bin/env python3
"""Equal WALL-CLOCK SEARCH budgets; selected final-refit cost reported separately."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time

FORMAL = Path(__file__).resolve().parents[1]
CHECKPOINTS = [60, 300, 900]
for key in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[key] = "1"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def incumbents(records, checkpoints):
    answer = []
    for budget in checkpoints:
        eligible = [r for r in records if r["status"] == "success" and r["completed_search_wall_sec"] <= budget]
        selected = min(eligible, key=lambda r: (r["validation_rmse"], r["order"])) if eligible else None
        answer.append({"search_budget_sec": budget, "selected_candidate_index": selected["candidate_index"] if selected else None,
            "validation_rmse": selected["validation_rmse"] if selected else None,
            "completed_candidates": len(eligible)})
    return answer


def execute(command, output, timeout, core):
    start = time.monotonic()
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(["taskset", "-c", str(core)] + command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            status = "success" if code == 0 else f"failed_exit_{code}"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(); status = "timeout"
    return status, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgboost-python-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--final-refit-timeout", type=int, default=600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.workers not in range(1, 9) or args.final_refit_timeout < 1: parser.error("Use 1..8 workers and a positive final-fit timeout")
    pool_file = FORMAL / "configs/real_candidate_pools.json"
    pools = json.loads(pool_file.read_text())
    sizes = {len(v) for v in pools.values()}
    if len(sizes) != 1: raise RuntimeError("Candidate pool sizes differ")
    order = list(range(1, sizes.pop() + 1)); random.Random(42).shuffle(order)
    tasks = [(d, f) for d in ["cpu_activity", "kin8nm", "energy_efficiency"] for f in range(1, 6)]
    plan = {"clock": "monotonic elapsed wall time", "search_budgets_sec": CHECKPOINTS,
        "timing_scope": "continuous 900-second search includes child startup, imports, input I/O, inner fitting, validation prediction and orchestration",
        "excluded_from_search_budget": ["final refits", "final test prediction", "post-search summarization"],
        "claim_scope": "equal search budget, NOT equal end-to-end training budget; final refit wall/CPU cost must appear alongside results",
        "selection": "lowest mean inner RMSE among four-fold-complete candidates finished by checkpoint; tie by frozen order",
        "candidate_order_seed": 42, "candidate_order": order, "candidate_pool_sha256": digest(pool_file),
        "candidate_limit": len(order), "pool_exhaustion": "Report explicitly if the finite frozen pool is exhausted before the wall cap",
        "datasets": ["cpu_activity", "kin8nm", "energy_efficiency"], "outer_folds": 5,
        "workers": args.workers, "search_wall_sum_upper_bound_sec": len(tasks) * 2 * max(CHECKPOINTS),
        "final_refits_max_per_algorithm": len(tasks) * len(CHECKPOINTS),
        "fits": "dynamic; log all completed/failed/timed-out candidate attempts; each successful candidate has four inner fits"}
    if not args.execute: print(json.dumps(plan, indent=2)); return
    if platform.system() != "Linux" or not shutil.which("taskset"):
        raise RuntimeError("Linux with taskset is required")
    if args.gate is None: parser.error("Execution requires --gate R0_gate.json")
    sys.path.insert(0, str(args.xgboost_python_package.resolve()))
    import xgboost as xgb
    native_hash = digest(Path(xgb.core._LIB._name))
    gate = json.loads(args.gate.read_text())
    from run_cgb_task import current_source_hashes, helper_hashes
    if not gate.get("passed") or gate.get("native_library_sha256") != native_hash or gate.get("controlled_source") != current_source_hashes():
        raise RuntimeError("R0 gate does not match the native runtime/source")
    if gate.get("helper_hashes") != helper_hashes(FORMAL.parent / "rebuttal_execution_20261025/source/exp"):
        raise RuntimeError("Model construction helpers changed after R0")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < args.workers: raise RuntimeError("Insufficient allocated CPU affinity")
    args.output.mkdir(parents=True, exist_ok=False)
    plan.update(assigned_cores=cores[:args.workers], uname=list(platform.uname()), python=sys.version,
        native_library_sha256=native_hash, gate_sha256=digest(args.gate), driver_sha256=digest(FORMAL / "scripts/run_real_fit.py"))
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    runner = FORMAL / "scripts/run_real_fit.py"
    def worker(worker_index):
        reports = []
        core = cores[worker_index]
        for task_index, (dataset, fold) in enumerate(tasks):
            if task_index % args.workers != worker_index: continue
            # Alternate method order across pairs before any testing; both use this same CPU.
            methods = ["xgb_cpu", "current_metagb"] if task_index % 2 == 0 else ["current_metagb", "xgb_cpu"]
            for method in methods:
                destination = args.output / f"{dataset}__outer{fold}__{method}"
                destination.mkdir()
                records = []
                start = time.monotonic()
                for rank, candidate in enumerate(order, 1):
                    remaining = max(CHECKPOINTS) - (time.monotonic() - start)
                    if remaining <= 0: break
                    params_file = destination / f"candidate{candidate}.json"
                    params_file.write_text(json.dumps(pools[method][candidate - 1], indent=2))
                    output = destination / f"candidate{candidate}"
                    command = [sys.executable, str(runner), "--dataset", dataset, "--method", method,
                        "--outer-fold", str(fold), "--params-file", str(params_file), "--xgboost-python-package", str(args.xgboost_python_package),
                        "--output", str(output), "--mode", "candidate"]
                    status, elapsed = execute(command, output, remaining, core)
                    record = {"order": rank, "candidate_index": candidate, "status": status,
                        "process_wall_sec": elapsed, "completed_search_wall_sec": time.monotonic() - start}
                    result = output / "result.json"
                    if status == "success": record["validation_rmse"] = json.loads(result.read_text())["rmse"]
                    progress = output / "progress.json"
                    if progress.exists(): record.update(json.loads(progress.read_text()))
                    records.append(record)
                    # Search log writing also consumes the declared wall budget.
                    (destination / "search_trace.json").write_text(json.dumps(records, indent=2))
                search_elapsed = time.monotonic() - start
                selected = incumbents(records, CHECKPOINTS)
                final = {}
                for item in selected:
                    candidate = item["selected_candidate_index"]
                    if candidate is None or candidate in final: continue
                    output = destination / f"final_candidate{candidate}"
                    command = [sys.executable, str(runner), "--dataset", dataset, "--method", method,
                        "--outer-fold", str(fold), "--params-file", str(destination / f"candidate{candidate}.json"),
                        "--xgboost-python-package", str(args.xgboost_python_package), "--output", str(output), "--mode", "final"]
                    status, elapsed = execute(command, output, args.final_refit_timeout, core)
                    result = json.loads((output / "result.json").read_text()) if status == "success" else None
                    final[candidate] = {"status": status, "process_wall_sec_outside_search_budget": elapsed,
                        "test_rmse": result["rmse"] if result else None,
                        "fit_wall_sec": result["costs"][0]["fit_wall_sec"] if result else None,
                        "fit_cpu_sec": result["costs"][0]["fit_cpu_sec"] if result else None}
                    progress = output / "progress.json"
                    if progress.exists(): final[candidate].update(json.loads(progress.read_text()))
                for item in selected:
                    item["final_refit"] = final.get(item["selected_candidate_index"])
                    item["search_cap_plus_selected_refit_wall_sec"] = (min(search_elapsed, item["search_budget_sec"]) + item["final_refit"]["process_wall_sec_outside_search_budget"]) if item["final_refit"] else None
                report = {"dataset": dataset, "outer_fold": fold, "method": method, "core": core,
                    "search_elapsed_sec": search_elapsed, "candidate_attempts": len(records), "checkpoints": selected,
                    "candidate_pool_exhausted": len(records) == len(order),
                    "search_fit_calls_started": sum(r.get("fits_started", 0) for r in records),
                    "search_fits_completed": sum(r.get("fits_completed", 0) for r in records),
                    "scope": plan["claim_scope"], "timeouts_are_not_successful_results": True}
                (destination / "budget_results.json").write_text(json.dumps(report, indent=2))
                reports.append(report)
        return reports
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = [r for batch in pool.map(worker, range(args.workers)) for r in batch]
    (args.output / "all_budget_results.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps({"completed_outer_method_pairs": len(reports), "scope": plan["claim_scope"]}))


if __name__ == "__main__": main()
