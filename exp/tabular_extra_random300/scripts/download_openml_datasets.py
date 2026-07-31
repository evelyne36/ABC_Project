from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "tabular_extra_random300"
DATA_DIR = PROJECT_DIR / "data"


@dataclass(frozen=True)
class OpenMLDataset:
    key: str
    openml_id: int
    openml_name: str
    task_type: str
    filename: str
    status: str
    notes: str


REGISTRY: dict[str, OpenMLDataset] = {
    "phoneme": OpenMLDataset(
        key="phoneme",
        openml_id=1489,
        openml_name="phoneme",
        task_type="binary",
        filename="binary__phoneme.csv",
        status="reserve_candidate",
        notes="Low-dimensional binary OpenML candidate; download if first binary batch needs more evidence.",
    ),
    "pc1": OpenMLDataset(
        key="pc1",
        openml_id=1068,
        openml_name="pc1",
        task_type="binary",
        filename="binary__pc1.csv",
        status="reserve_candidate",
        notes="Compact software-defect binary task.",
    ),
    "balance-scale": OpenMLDataset(
        key="balance-scale",
        openml_id=11,
        openml_name="balance-scale",
        task_type="multiclass",
        filename="multiclass__balance-scale.csv",
        status="reserve_candidate",
        notes="Very small multiclass task; useful if additional low-cost evidence is needed.",
    ),
    "mfeat-fourier": OpenMLDataset(
        key="mfeat-fourier",
        openml_id=14,
        openml_name="mfeat-fourier",
        task_type="multiclass",
        filename="multiclass__mfeat-fourier.csv",
        status="reserve_candidate",
        notes="MFEAT view with moderate feature count.",
    ),
    "mfeat-karhunen": OpenMLDataset(
        key="mfeat-karhunen",
        openml_id=16,
        openml_name="mfeat-karhunen",
        task_type="multiclass",
        filename="multiclass__mfeat-karhunen.csv",
        status="reserve_candidate",
        notes="MFEAT view with lower feature count than Fourier.",
    ),
    "abalone": OpenMLDataset(
        key="abalone",
        openml_id=183,
        openml_name="abalone",
        task_type="regression",
        filename="regression__abalone.csv",
        status="reserve_candidate",
        notes="Small mixed tabular candidate; verify target semantics before paper use.",
    ),
    "boston": OpenMLDataset(
        key="boston",
        openml_id=531,
        openml_name="boston",
        task_type="regression",
        filename="regression__boston.csv",
        status="appendix_candidate",
        notes="Legacy housing benchmark; keep for appendix-style diagnostics only.",
    ),
    "mozilla4": OpenMLDataset(
        key="mozilla4",
        openml_id=1046,
        openml_name="mozilla4",
        task_type="binary",
        filename="binary__mozilla4.csv",
        status="reserve_candidate",
        notes="Binary performance dataset reserve candidate.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download selected tabular extension datasets from OpenML.")
    parser.add_argument("--datasets", nargs="+", default=["phoneme", "pc1", "balance-scale", "mfeat-karhunen"])
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="List the local registry and exit.")
    return parser


def selected_entries(names: Iterable[str]) -> list[OpenMLDataset]:
    entries: list[OpenMLDataset] = []
    for name in names:
        if name == "all":
            return list(REGISTRY.values())
        if name not in REGISTRY:
            raise KeyError(f"Unknown dataset {name}. Available: {', '.join(sorted(REGISTRY))}")
        entries.append(REGISTRY[name])
    return entries


def normalize_frame(frame: pd.DataFrame, target_name: str | None) -> pd.DataFrame:
    if target_name and target_name in frame.columns:
        target = frame.pop(target_name)
    else:
        target = frame.iloc[:, -1]
        frame = frame.iloc[:, :-1].copy()
    out = frame.copy()
    out["target"] = target
    return out.dropna(axis=0, how="any").reset_index(drop=True)


def download_one(entry: OpenMLDataset, output_dir: Path, force: bool) -> dict[str, object]:
    output_path = output_dir / entry.filename
    if output_path.exists() and not force:
        return {
            **asdict(entry),
            "output_path": str(output_path),
            "status": "skip_existing",
            "rows": None,
            "features": None,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        data_id=entry.openml_id,
        as_frame=True,
        parser="auto",
    )
    frame = dataset.frame
    if frame is None:
        raise ValueError(f"OpenML dataset {entry.openml_name} did not return a pandas frame")
    target_name = dataset.target_names[0] if dataset.target_names else None
    out = normalize_frame(frame, target_name)
    out.to_csv(output_path, index=False)
    return {
        **asdict(entry),
        "output_path": str(output_path),
        "status": "downloaded",
        "rows": int(out.shape[0]),
        "features": int(out.shape[1] - 1),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        print(json.dumps({key: asdict(value) for key, value in REGISTRY.items()}, indent=2, ensure_ascii=False))
        return 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_rows = []
    failures = 0
    for entry in selected_entries(args.datasets):
        try:
            row = download_one(entry, output_dir, args.force)
            print(json.dumps(row, ensure_ascii=False))
        except Exception as exc:
            failures += 1
            row = {**asdict(entry), "status": "failed", "error": repr(exc)}
            print(json.dumps(row, ensure_ascii=False), file=sys.stderr)
        manifest_rows.append(row)
    manifest_path = PROJECT_DIR / "openml_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
