#!/usr/bin/env python3
"""List, download, and verify datasets from the public repository registry."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = REPO_ROOT / "data_registry"
OPENML_MANIFEST = REGISTRY_ROOT / "openml_source_manifest.json"
TABARENA_MANIFEST = REGISTRY_ROOT / "tabarena_suite_457.json"
TIMESERIES_MANIFEST = REGISTRY_ROOT / "timeseries_registry.json"
CHECKSUM_MANIFEST = REGISTRY_ROOT / "checksums.sha256"
LICENSE_NOTE = "unknown; verify upstream dataset metadata"


@dataclass(frozen=True)
class Dataset:
    key: str
    group: str
    target: str | None
    source_url: str | None
    sha256: str | None
    license: str
    source_type: str
    source_id: str | None = None
    target_name: str | None = None
    source_md5: str | None = None
    expected_rows: int | None = None
    expected_features: int | None = None
    generator: str | None = None
    generator_seed: int | None = None
    compression: str | None = None
    unavailable_reason: str | None = None

    @property
    def downloadable(self) -> bool:
        if self.source_type == "generator":
            return (
                self.unavailable_reason is None
                and self.target is not None
                and self.sha256 is not None
                and self.generator is not None
                and self.generator_seed is not None
            )
        return (
            self.unavailable_reason is None
            and self.target is not None
            and self.source_url is not None
            and (self.sha256 is not None or self.source_md5 is not None)
        )


def load_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in CHECKSUM_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, target = line.split(maxsplit=1)
        checksums[target.strip()] = digest
    return checksums


def load_registry() -> dict[str, Dataset]:
    checksums = load_checksums()
    registry: dict[str, Dataset] = {}

    openml = json.loads(OPENML_MANIFEST.read_text(encoding="utf-8"))
    for row in openml["datasets"]:
        target = f"exp/tabular_extra_random300/data/{row['file']}"
        key = f"openml:{row['name']}"
        source_type = str(row.get("source_type", "openml"))
        registry[key] = Dataset(
            key=key,
            group="openml",
            target=target,
            source_url=(
                str(row["source_url"])
                if row.get("source_url") is not None
                else None
            ),
            sha256=checksums.get(target),
            license=LICENSE_NOTE,
            source_type=source_type,
            source_id=(
                str(row["openml_id"])
                if row.get("openml_id") is not None
                else None
            ),
            target_name=str(row["target"]),
            expected_rows=int(row["rows"]),
            expected_features=int(row["features"]),
            generator=row.get("generator"),
            generator_seed=(
                int(row["generator_seed"])
                if row.get("generator_seed") is not None
                else None
            ),
        )

    tabarena = json.loads(TABARENA_MANIFEST.read_text(encoding="utf-8"))
    for row in tabarena["datasets"]:
        task_type = str(row["task_type"])
        target = f"dataset/TabArena_{task_type}/{row['file']}"
        key = f"tabarena:{row['name']}"
        registry[key] = Dataset(
            key=key,
            group="tabarena",
            target=target,
            source_url=str(row["source_url"]),
            # This pins a retained CSV snapshot when present. Fresh downloads
            # also verify the immutable OpenML ARFF MD5, then write a local
            # sidecar because CSV rendering can vary across dependency
            # versions without changing the underlying dataset.
            sha256=checksums.get(target),
            license=str(row["license"]),
            source_type="openml",
            source_id=str(row["openml_data_id"]),
            target_name=str(row["target"]),
            source_md5=str(row["md5_checksum"]),
        )

    timeseries = json.loads(TIMESERIES_MANIFEST.read_text(encoding="utf-8"))
    for key, row in timeseries.items():
        target = str(row["target"])
        missing_checksum = checksums.get(target) is None
        registry[key] = Dataset(
            key=key,
            group="timeseries",
            target=target,
            source_url=str(row["url"]),
            sha256=checksums.get(target),
            license=LICENSE_NOTE,
            source_type="url",
            source_id=str(row["url"]),
            compression=row.get("compression"),
            unavailable_reason=(
                "no pinned checksum was available in the removed snapshot"
                if missing_checksum
                else None
            ),
        )

    return registry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(
    registry: dict[str, Dataset],
    groups: Iterable[str],
    names: Iterable[str],
) -> list[Dataset]:
    selected: dict[str, Dataset] = {}
    requested_groups = list(groups)
    requested_names = list(names)
    valid_groups = {entry.group for entry in registry.values()}
    for group in requested_groups:
        if group not in valid_groups:
            raise ValueError(
                f"unknown group {group!r}; available: {', '.join(sorted(valid_groups))}"
            )

    # Explicit dataset names narrow a group selection instead of being unioned
    # with every member of that group.
    for name in requested_names:
        if name not in registry:
            raise ValueError(
                f"unknown dataset {name!r}; use the list command to inspect keys"
            )
        entry = registry[name]
        if requested_groups and entry.group not in requested_groups:
            raise ValueError(
                f"dataset {name!r} belongs to group {entry.group!r}, "
                f"not {', '.join(sorted(requested_groups))}"
            )
        selected[name] = entry
    if requested_names:
        return [selected[key] for key in sorted(selected)]

    for group in requested_groups:
        for entry in registry.values():
            if entry.group == group:
                selected[entry.key] = entry
    return [selected[key] for key in sorted(selected)]


def resolved_target(entry: Dataset, output_root: Path) -> Path:
    if entry.target is None:
        raise ValueError(f"{entry.key}: no target path is registered")
    target = (output_root / entry.target).resolve()
    root = output_root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"{entry.key}: target escapes output root: {target}")
    return target


def checksum_sidecar(target: Path) -> Path:
    return target.with_name(f"{target.name}.sha256")


def write_checksum_sidecar(target: Path, digest: str) -> None:
    checksum_sidecar(target).write_text(
        f"{digest}  {target.name}\n",
        encoding="utf-8",
    )


def sidecar_checksum(target: Path) -> str | None:
    sidecar = checksum_sidecar(target)
    if not sidecar.is_file():
        return None
    fields = sidecar.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise RuntimeError(f"invalid checksum sidecar: {sidecar}")
    return fields[0]


def validate_source_pinned_csv(entry: Dataset, target: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "CSV validation requires pandas from requirements.txt"
        ) from exc
    frame = pd.read_csv(target)
    if frame.empty or frame.shape[1] < 2:
        raise RuntimeError(f"{entry.key}: local CSV is empty or has no feature columns")
    if entry.target_name not in frame.columns:
        raise RuntimeError(
            f"{entry.key}: expected target column {entry.target_name!r} "
            f"is absent from {target}"
        )
    validate_frame_shape(entry, frame)


def validate_frame_shape(entry: Dataset, frame: object) -> None:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) != 2:
        raise RuntimeError(f"{entry.key}: generated object is not a two-dimensional table")
    rows, columns = int(shape[0]), int(shape[1])
    if entry.expected_rows is not None and rows != entry.expected_rows:
        raise RuntimeError(
            f"{entry.key}: expected {entry.expected_rows} cleaned rows, got {rows}"
        )
    if entry.expected_features is not None and columns - 1 != entry.expected_features:
        raise RuntimeError(
            f"{entry.key}: expected {entry.expected_features} features, "
            f"got {columns - 1}"
        )


def canonical_openml_frame(
    features: object,
    target: object,
    *,
    output_target_name: str,
) -> object:
    """Build the complete-case table consumed by the submitted experiments."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "OpenML preprocessing requires pandas from requirements.txt"
        ) from exc

    frame = pd.DataFrame(features).copy()
    if output_target_name in frame.columns:
        raise RuntimeError(
            f"target column {output_target_name!r} collides with an input feature"
        )
    frame[output_target_name] = pd.Series(
        target,
        index=frame.index,
        name=output_target_name,
    )
    # The retained experiment snapshots are complete-case tables. This explicit
    # transform explains the manifest's raw_rows/rows distinction and ensures
    # that a fresh fixed-ID download follows the same protocol.
    return frame.dropna(axis=0, how="any").reset_index(drop=True)


def canonical_csv_bytes(frame: object) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def openml_payload(entry: Dataset) -> bytes:
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise RuntimeError(
            "OpenML downloads require scikit-learn and pandas from requirements.txt"
        ) from exc

    if entry.source_md5:
        request = urllib.request.Request(
            entry.source_url or "",
            headers={"User-Agent": "CrossGB-dataset-downloader/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            source_payload = response.read()
        actual_md5 = hashlib.md5(source_payload, usedforsecurity=False).hexdigest()
        if actual_md5 != entry.source_md5:
            raise RuntimeError(
                f"{entry.key}: OpenML source MD5 mismatch: "
                f"expected {entry.source_md5}, got {actual_md5}"
            )

    bunch = fetch_openml(data_id=int(entry.source_id or ""), as_frame=True, parser="auto")
    if bunch.data is None or bunch.target is None:
        raise RuntimeError(f"{entry.key}: OpenML returned no supervised dataframe")
    target_name = entry.target_name
    if not target_name:
        raise RuntimeError(f"{entry.key}: the registry does not define a target column")
    frame = canonical_openml_frame(
        bunch.data,
        bunch.target,
        output_target_name=target_name,
    )
    validate_frame_shape(entry, frame)
    return canonical_csv_bytes(frame)


def generator_payload(entry: Dataset) -> bytes:
    if entry.generator != "sklearn.make_friedman1":
        raise RuntimeError(f"{entry.key}: unsupported generator {entry.generator!r}")
    try:
        import pandas as pd
        from sklearn.datasets import make_friedman1
    except ImportError as exc:
        raise RuntimeError(
            "Synthetic dataset generation requires scikit-learn and pandas "
            "from requirements.txt"
        ) from exc

    if entry.expected_rows is None or entry.expected_features is None:
        raise RuntimeError(f"{entry.key}: generator shape is not pinned")
    features, target = make_friedman1(
        n_samples=entry.expected_rows,
        n_features=entry.expected_features,
        noise=1.0,
        random_state=entry.generator_seed,
    )
    frame = pd.DataFrame(
        features,
        columns=[f"x{index}" for index in range(entry.expected_features)],
    )
    frame[entry.target_name or "target"] = target
    validate_frame_shape(entry, frame)
    return canonical_csv_bytes(frame)


def url_payload(entry: Dataset) -> bytes:
    request = urllib.request.Request(
        entry.source_url or "", headers={"User-Agent": "CrossGB-dataset-downloader/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if entry.compression == "gzip":
        payload = gzip.decompress(payload)
    return payload


def download_one(entry: Dataset, output_root: Path, force: bool) -> Path:
    if not entry.downloadable:
        raise RuntimeError(f"{entry.key}: {entry.unavailable_reason or 'not downloadable'}")
    target = resolved_target(entry, output_root)
    if target.exists() and not force:
        expected = sidecar_checksum(target) or entry.sha256
        if expected is None and entry.source_md5 is not None:
            validate_source_pinned_csv(entry, target)
            print(f"valid-source-pinned-csv\t{entry.key}\t{target}")
            return target
        actual = sha256_file(target)
        if actual == expected:
            print(f"verified existing\t{entry.key}\t{target}")
            return target
        raise RuntimeError(
            f"{entry.key}: existing target checksum mismatch; use --force to replace it"
        )

    if entry.source_type == "openml":
        payload = openml_payload(entry)
    elif entry.source_type == "generator":
        payload = generator_payload(entry)
    else:
        payload = url_payload(entry)
    actual = hashlib.sha256(payload).hexdigest()
    if (
        entry.sha256 is not None
        and entry.source_type != "openml"
        and entry.source_md5 is None
        and actual != entry.sha256
    ):
        raise RuntimeError(
            f"{entry.key}: downloaded checksum mismatch: expected {entry.sha256}, got {actual}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        temporary.replace(target)
        write_checksum_sidecar(target, actual)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"downloaded\t{entry.key}\t{target}\t{len(payload)} bytes")
    return target


def command_group(registry: dict[str, Dataset], as_json: bool) -> int:
    groups: dict[str, dict[str, int]] = {}
    for entry in registry.values():
        record = groups.setdefault(entry.group, {"total": 0, "downloadable": 0})
        record["total"] += 1
        record["downloadable"] += int(entry.downloadable)
    if as_json:
        print(json.dumps(groups, indent=2, sort_keys=True))
    else:
        for group, record in sorted(groups.items()):
            print(f"{group}\t{record['downloadable']}/{record['total']} downloadable")
    return 0


def command_list(entries: list[Dataset], as_json: bool) -> int:
    rows = [
        {
            "key": entry.key,
            "group": entry.group,
            "downloadable": entry.downloadable,
            "source_type": entry.source_type,
            "source_id": entry.source_id,
            "target": entry.target,
            "sha256": entry.sha256,
            "source_md5": entry.source_md5,
            "license": entry.license,
            "unavailable_reason": entry.unavailable_reason,
        }
        for entry in entries
    ]
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            status = "downloadable" if row["downloadable"] else "unavailable"
            print(f"{row['group']}\t{row['key']}\t{status}\t{row['target'] or '-'}")
    return 0


def command_verify(
    entries: list[Dataset], output_root: Path, allow_missing: bool
) -> int:
    missing_failures = 0
    integrity_failures = 0
    for entry in entries:
        if entry.target is None:
            print(f"unavailable\t{entry.key}\t{entry.unavailable_reason}")
            missing_failures += 1
            continue
        target = resolved_target(entry, output_root)
        if not target.exists():
            print(f"missing\t{entry.key}\t{target}")
            missing_failures += 1
            continue
        expected = sidecar_checksum(target) or entry.sha256
        if expected is None and entry.source_md5 is not None:
            try:
                validate_source_pinned_csv(entry, target)
            except RuntimeError as exc:
                print(f"invalid\t{entry.key}\t{exc}")
                integrity_failures += 1
                continue
            print(f"valid-source-pinned-csv\t{entry.key}\t{target}")
            continue
        if expected is None:
            print(f"unverified\t{entry.key}\tno pinned checksum")
            integrity_failures += 1
            continue
        actual = sha256_file(target)
        if actual != expected:
            print(f"mismatch\t{entry.key}\texpected={expected}\tactual={actual}")
            integrity_failures += 1
            continue
        print(f"ok\t{entry.key}\t{target}")
    if integrity_failures:
        return 1
    if missing_failures and not allow_missing:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    group_parser = subparsers.add_parser("group", help="List registry groups.")
    group_parser.add_argument("--json", action="store_true")

    for command in ("list", "download", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--group", action="append", default=[])
        subparser.add_argument("--dataset", action="append", default=[])
        if command == "list":
            subparser.add_argument("--json", action="store_true")
        else:
            subparser.add_argument("--output-root", type=Path, default=REPO_ROOT)
        if command == "download":
            subparser.add_argument("--force", action="store_true")
        if command == "verify":
            subparser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry()
    if args.command == "group":
        return command_group(registry, args.json)

    entries = select(registry, args.group, args.dataset)
    if not entries:
        entries = [registry[key] for key in sorted(registry)]
    if args.command == "list":
        return command_list(entries, args.json)
    if args.command == "verify":
        return command_verify(entries, args.output_root, args.allow_missing)

    failures = 0
    explicitly_requested = set(args.dataset)
    for entry in entries:
        if not entry.downloadable and entry.key not in explicitly_requested:
            print(f"skipped-unavailable\t{entry.key}\t{entry.unavailable_reason}")
            continue
        try:
            download_one(entry, args.output_root, args.force)
        except Exception as exc:
            failures += 1
            print(f"failed\t{entry.key}\t{exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
