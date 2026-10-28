#!/usr/bin/env python3
"""Fetch pinned public baseline archives without importing or executing them."""
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.provenance import code_fingerprint
SOURCES = {
    "linxgboost": ("https://codeload.github.com/ldv1/LinXGBoost/tar.gz/0166fb3682a0e534c54aea0559186fee26192d6f", "0166fb3682a0e534c54aea0559186fee26192d6f"),
    "gbdt_pl": ("https://codeload.github.com/GBDT-PL/GBDT-PL/tar.gz/8942ed2fc881bed8211854f4995cbe6eb14e5cae", "8942ed2fc881bed8211854f4995cbe6eb14e5cae"),
    "pilot": ("https://codeload.github.com/STAN-UAntwerp/PILOT/tar.gz/ea769a0d55dd2babf4c57d31094758d7879f29ff", "ea769a0d55dd2babf4c57d31094758d7879f29ff"),
    "tao": ("https://faculty.ucmerced.edu/mcarreira-perpinan/research/software/code-TAO-v2026-07-22.tar.gz", "official-2026-07-22"),
}


def main():
    records = []
    for name, (url, revision) in SOURCES.items():
        archive = ROOT / "downloads" / f"{name}.tar.gz"
        if not archive.exists():
            print(f"Downloading {name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "baseline-reproducibility-setup/1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            temporary = archive.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(archive)
        target = ROOT / "vendor" / name
        if not target.exists():
            target.mkdir()
            with tarfile.open(archive, "r:gz") as source:
                source.extractall(target, filter="data")
        fingerprint,files=code_fingerprint(target)
        records.append({"method": name, "url": url, "revision": revision,
                        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        "archive_bytes": archive.stat().st_size,
                        "code_fingerprint":fingerprint,"code_file_count":len(files),
                        "extracted_path": str(target.relative_to(ROOT))})
        print(f"Ready {name}: {archive.stat().st_size} bytes", flush=True)
        (ROOT / "reports" / "sources.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
