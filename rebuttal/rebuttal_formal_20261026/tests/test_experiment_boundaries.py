"""Check selection/caching boundaries without fitting a model or using native CGB."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

FORMAL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget", FORMAL / "scripts/run_equal_search_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_budget_does_not_select_future_or_failed_candidates():
    records = [
        {"status": "success", "completed_search_wall_sec": 59, "order": 1, "candidate_index": 10, "validation_rmse": 3},
        {"status": "success", "completed_search_wall_sec": 61, "order": 2, "candidate_index": 20, "validation_rmse": 2},
        {"status": "failed", "completed_search_wall_sec": 62, "order": 3, "candidate_index": 30, "validation_rmse": 0},
        {"status": "success", "completed_search_wall_sec": 301, "order": 4, "candidate_index": 40, "validation_rmse": 1},
    ]
    result = budget.incumbents(records, [30, 60, 300, 900])
    assert [r["selected_candidate_index"] for r in result] == [None, 10, 20, 40]
    assert [r["completed_candidates"] for r in result] == [0, 1, 2, 3]


@pytest.mark.parametrize("issue", [None, "missing_case", "different_library", "prediction_mismatch"])
def test_leafonly_gate_rejects_incomplete_or_incompatible_evidence(tmp_path, issue):
    # Explicit fixtures exercise gate logic; these are not experiment results.
    queue = json.loads((FORMAL / "configs/R0_E0_queue.json").read_text())
    for index, item in enumerate(queue):
        if issue == "missing_case" and index == 0: continue
        task = FORMAL / item["task"]
        identifier = json.loads(task.read_text())["task_id"]
        directory = tmp_path / (identifier + "__" + item["variant"])
        directory.mkdir()
        record = {"task_id": identifier, "variant": item["variant"], "status": "success",
            "task_file_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
            "archive_prediction_pass": not (issue == "prediction_mismatch" and index == 0),
            "archive_max_abs_prediction_difference": 0,
            "native_library_sha256": "fixture_other_library" if issue == "different_library" and index == 0 else "fixture_library",
            "controlled_source": {}, "driver_sha256": "fixture_driver", "helper_hashes": {}}
        (directory / "result.json").write_text(json.dumps(record))
    native = tmp_path / "native.json"
    native.write_text(json.dumps({"passed": True, "native_library_sha256": "fixture_library"}))
    output = tmp_path / "gate.json"
    result = subprocess.run([sys.executable, str(FORMAL / "scripts/collect_cgb_gate.py"), "--results", str(tmp_path),
        "--native-check", str(native), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == (0 if issue is None else 2), result.stderr
    assert json.loads(output.read_text())["passed"] == (issue is None)
