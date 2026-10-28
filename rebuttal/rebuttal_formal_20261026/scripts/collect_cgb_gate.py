#!/usr/bin/env python3
"""Require all seven archived scalar endpoint cases to match before LeafOnly."""
import argparse
import hashlib
import json
from pathlib import Path

FORMAL = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-check", required=True, type=Path)
    args = parser.parse_args()
    queue = json.loads((FORMAL / "configs/R0_E0_queue.json").read_text())
    results, problems = [], []
    for item in queue:
        task = FORMAL / item["task"]
        task_id = json.loads(task.read_text())["task_id"]
        path = args.results / (task_id + "__" + item["variant"]) / "result.json"
        if not path.exists():
            problems.append("missing:" + str(path)); continue
        result = json.loads(path.read_text())
        if result["task_file_sha256"] != digest(task): problems.append("task_changed:" + task_id)
        if not result.get("archive_prediction_pass"): problems.append("prediction_mismatch:" + str(path))
        if result.get("status") != "success": problems.append("failed:" + str(path))
        results.append(result)
    identities = ["native_library_sha256", "controlled_source", "driver_sha256", "helper_hashes"]
    for key in identities:
        if len({json.dumps(r[key], sort_keys=True) for r in results}) != 1:
            problems.append("inconsistent:" + key)
    native = json.loads(args.native_check.read_text())
    if not native.get("passed") or not results or native.get("native_library_sha256") != results[0]["native_library_sha256"]:
        problems.append("native_structural_check_not_passed_for_this_library")
    report = {"passed": not problems and len(results) == 28, "expected": 28, "completed": len(results),
        "problems": problems, "scope": "seven dim10 seed1001 outer1 endpoint checks; not an exhaustive replay of all historical fits",
        "cases": [{"task_id": r["task_id"], "variant": r["variant"], "difference": r["archive_max_abs_prediction_difference"]} for r in results]}
    if results: report.update({key: results[0][key] for key in identities})
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "completed": len(results), "problems": problems}))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
