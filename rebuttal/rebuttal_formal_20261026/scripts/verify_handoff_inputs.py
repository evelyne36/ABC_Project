#!/usr/bin/env python3
"""Verify outer manifest and extracted baseline archive. Standard library, zero fits."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest = root / "rebuttal_formal_20261026/deployment_manifest.json"
    rows = json.loads(manifest.read_text())["files"]
    errors = []
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root): raise ValueError("Manifest path leaves the task directory")
        if not path.is_file(): errors.append("missing:" + row["path"]); continue
        with path.open("rb") as stream: actual = digest(stream)
        if actual != row["sha256"]: errors.append("changed:" + row["path"])
    count = 0
    with tarfile.open(root / "rebuttal_baselines_20261025/baseline_linux_bundle.tar.gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk(): raise ValueError("Baseline archive contains a link")
            if not member.isfile(): continue
            path = (root / member.name).resolve()
            if not path.is_relative_to(root): raise ValueError("Baseline archive path leaves the task directory")
            if not path.is_file(): errors.append("baseline_missing:" + member.name); continue
            with archive.extractfile(member) as packed, path.open("rb") as extracted:
                if digest(packed) != digest(extracted): errors.append("baseline_changed:" + member.name)
            count += 1
    report = {"status": "verified" if not errors else "failed", "outer_manifest_files": len(rows),
        "baseline_extracted_files": count, "model_fit_calls": 0,
        "scope": "transfer and extraction integrity only; no Linux build or scientific runtime validation", "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(2 if errors else 0)


if __name__ == "__main__": main()
