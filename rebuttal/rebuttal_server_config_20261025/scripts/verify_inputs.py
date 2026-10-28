#!/usr/bin/env python3
"""Read-only configuration and transfer audit; never imports or fits a model."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--packages", action="store_true", help="Read package metadata only; no native imports")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    pack = Path(__file__).resolve().parents[1]
    errors, warnings = [], []
    manifest = json.loads((pack / "reports/input_transfer_manifest.json").read_text())
    checked = 0
    for entry in manifest["files"]:
        path = (project / entry["path"]).resolve()
        if not path.is_relative_to(project):
            errors.append(f"Path outside project root: {entry['path']}")
        elif not path.is_file():
            errors.append(f"Missing input: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"SHA256 mismatch: {entry['path']}")
        else:
            checked += 1
    configs = {}
    for path in sorted((pack / "configs").glob("*.json")):
        configs[path.stem] = json.loads(path.read_text())
    with (pack / "inputs/e1_task_matrix.csv").open(encoding="utf-8-sig") as stream:
        e1 = list(csv.DictReader(stream))
    with (pack / "inputs/e2_task_matrix.csv").open(encoding="utf-8-sig") as stream:
        e2 = list(csv.DictReader(stream))
    e1keys = {(r["scenario"], r["data_seed"], r["outer_index"], r["variant"]) for r in e1}
    if len(e1) != 420 or len(e1keys) != 420:
        errors.append("E1 must have 420 unique planned scenario/seed/fold/variant rows")
    if len(e2) != 100 or sum(int(r["planned_fits"]) for r in e2) != 9100:
        errors.append("E2 must have 100 planned outer-model jobs / 9100 fits")
    nine = json.loads((pack / "inputs/ts_tasks_nine.json").read_text())["tasks"]
    h12 = json.loads((pack / "inputs/ts_tasks_h12.json").read_text())["tasks"]
    tskeys = {(r["dataset"], r["seq_len"], r["pred_len"]) for r in nine + h12}
    if len(nine) != 9 or len(h12) != 1 or len(tskeys) != 10:
        errors.append("E5 must contain nine old-setting tasks plus one later Exchange h12 task")
    if {k for _, _, k in tskeys} != {1, 6, 12}:
        errors.append("E5 forecast horizons do not match the submitted panel")
    native = project / configs["server"]["native"]["source_project_relative"]
    updater = native / "src/tree/updater_metagb_hist.cc"
    expected = configs["server"]["native"]["expected_historical_updater_sha256"]
    if not updater.is_file() or sha256(updater) != expected:
        errors.append("Historical updater is missing or does not match the archived source hash")
    for path in [pack / "scripts/run_ts_reference_panel.py", *list((pack / "source/exp").rglob("*.py"))]:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Python syntax: {path.relative_to(pack)}: {exc}")
    packages = {}
    if args.packages:
        for name, wanted in {"numpy": "2.3.5", "pandas": "2.3.3", "scipy": "1.16.3",
                             "scikit-learn": "1.7.2", "lightgbm": "4.6.0"}.items():
            try:
                actual = importlib.metadata.version(name)
                packages[name] = actual
                if actual != wanted:
                    warnings.append(f"Version differs from proposed lock: {name}={actual}, expected {wanted}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing server package: {name}")
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count()
    if platform.system() != "Linux":
        warnings.append("Static audit only: this host is not the target Linux server")
    print(json.dumps({
        "status": "inputs_and_configs_valid" if not errors else "failed",
        "model_fit_calls": 0,
        "native_build_performed": False,
        "project_root": str(project),
        "manifest_files_verified": checked,
        "planned_e1_rows": len(e1),
        "planned_e2_outer_model_jobs": len(e2),
        "planned_e5_tasks": len(tskeys),
        "host": {"system": platform.system(), "architecture": platform.machine(),
                 "python": platform.python_version(), "available_cpus": cpus},
        "packages": packages,
        "errors": errors,
        "warnings": warnings,
        "readiness_boundary": "configuration/input audit only; E0/E1/E2/E3/E4/E6 implementations and server runtime verification remain",
    }, ensure_ascii=False, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
