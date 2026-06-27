from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timeseries_extra_random300.utils.data import load_registry, raw_path_for  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download time-series benchmark datasets.")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "exchange_rate"])
    parser.add_argument("--registry", default=str(ADAPTER_ROOT / "configs" / "dataset_registry.json"))
    parser.add_argument("--force", action="store_true")
    return parser


def download_one(name: str, registry_path: Path, force: bool) -> Path:
    registry = load_registry(registry_path)
    if name not in registry:
        raise KeyError(f"Unknown dataset {name}. Available: {', '.join(sorted(registry))}")
    spec = registry[name]
    out_path = raw_path_for(ADAPTER_ROOT, spec)
    if out_path.exists() and not force:
        print(f"skip existing {name}: {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {name}: {spec.url}")
    with urllib.request.urlopen(spec.url, timeout=120) as response:
        payload = response.read()
    if spec.url.endswith(".gz"):
        payload = gzip.decompress(payload)
    out_path.write_bytes(payload)
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
