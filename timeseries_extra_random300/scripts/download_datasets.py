from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timeseries_extra_random300.utils.data import load_registry, raw_path_for, resolve_dataset_name  # noqa: E402


MANIFEST_PATH = ADAPTER_ROOT / "timeseries_download_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download time-series benchmark datasets.")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "exchange_rate"])
    parser.add_argument("--registry", default=str(ADAPTER_ROOT / "configs" / "dataset_registry.json"))
    parser.add_argument("--force", action="store_true")
    return parser


def update_manifest(dataset: str, out_path: Path, status: str, fmt: str) -> None:
    rows = []
    if MANIFEST_PATH.exists():
        rows = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rel_path = str(out_path.relative_to(PROJECT_ROOT))
    next_rows = []
    replaced = False
    for row in rows:
        if row.get("dataset") != dataset:
            next_rows.append(row)
            continue
        updated = dict(row)
        updated.update({"dataset": dataset, "status": status, "file": rel_path, "format": fmt})
        if out_path.exists():
            updated["bytes"] = out_path.stat().st_size
        next_rows.append(updated)
        replaced = True
    if not replaced:
        record = {"dataset": dataset, "status": status, "file": rel_path, "format": fmt}
        if out_path.exists():
            record["bytes"] = out_path.stat().st_size
        next_rows.append(record)
    MANIFEST_PATH.write_text(json.dumps(next_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_one(name: str, registry_path: Path, force: bool) -> Path:
    registry = load_registry(registry_path)
    canonical_name = resolve_dataset_name(registry, name)
    spec = registry[canonical_name]
    out_path = raw_path_for(ADAPTER_ROOT, spec)
    if out_path.exists() and not force:
        update_manifest(canonical_name, out_path, "existing", spec.format)
        print(f"skip existing {name} -> {canonical_name}: {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {name} -> {canonical_name}: {spec.url}")
    with urllib.request.urlopen(spec.url, timeout=120) as response:
        payload = response.read()
    if spec.url.endswith(".gz"):
        payload = gzip.decompress(payload)
    out_path.write_bytes(payload)
    update_manifest(canonical_name, out_path, "downloaded", spec.format)
    print(f"wrote {out_path} bytes={out_path.stat().st_size}")
    return out_path


def main() -> int:
    args = build_parser().parse_args()
    registry_path = Path(args.registry).expanduser().resolve()
    for dataset in args.datasets:
        download_one(dataset, registry_path, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
