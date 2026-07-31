from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_PATH = REPO_ROOT / "scripts" / "download_datasets.py"
SPEC = importlib.util.spec_from_file_location("crossgb_dataset_downloader", DOWNLOADER_PATH)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


PAPER_AND_LAUNCHER_OPENML_FILES = {
    "regression__Moneyball.csv",
    "regression__energy_efficiency.csv",
    "regression__boston.csv",
    "regression__space_ga.csv",
    "regression__friedman1.csv",
    "regression__abalone.csv",
    "regression__kin8nm.csv",
    "regression__naval_propulsion_plant.csv",
    "regression__cpu_activity.csv",
    "binary__haberman.csv",
    "binary__banknote-authentication.csv",
    "binary__breast-w.csv",
    "binary__ilpd.csv",
    "binary__phoneme.csv",
    "binary__credit-approval.csv",
    "binary__mozilla4.csv",
    "binary__pc1.csv",
    "binary__kc1.csv",
    "binary__electricity.csv",
    "multiclass__balance-scale.csv",
    "multiclass__vowel.csv",
    "multiclass__vehicle.csv",
    "multiclass__segment.csv",
    "multiclass__mfeat-karhunen.csv",
    "multiclass__satimage.csv",
    "multiclass__mfeat-fourier.csv",
    "multiclass__optdigits.csv",
}


def test_authoritative_registry_covers_paper_and_launcher_openml_files() -> None:
    registry = downloader.load_registry()
    registered = {
        Path(entry.target).name
        for entry in registry.values()
        if entry.group == "openml" and entry.target is not None
    }
    assert registered == PAPER_AND_LAUNCHER_OPENML_FILES


def test_legacy_openml_helper_uses_fixed_ids_not_active_versions() -> None:
    helper = (
        REPO_ROOT
        / "exp"
        / "tabular_extra_random300"
        / "scripts"
        / "download_openml_datasets.py"
    ).read_text(encoding="utf-8")

    assert "fetch_openml(" in helper
    assert "data_id=entry.openml_id" in helper
    assert 'version="active"' not in helper


def test_openml_preprocessing_is_deterministic_and_drops_incomplete_rows() -> None:
    features = pd.DataFrame(
        {
            "number": [1.0, None, 3.0, 4.0],
            "category": pd.Series(["b", "a", None, "c"], dtype="category"),
        }
    )
    target = pd.Series(["yes", "no", "yes", None], name="source_target")

    first = downloader.canonical_openml_frame(
        features,
        target,
        output_target_name="target",
    )
    second = downloader.canonical_openml_frame(
        features,
        target,
        output_target_name="target",
    )

    assert list(first.columns) == ["number", "category", "target"]
    assert first.to_dict(orient="records") == [
        {"number": 1.0, "category": "b", "target": "yes"}
    ]
    assert downloader.canonical_csv_bytes(first) == downloader.canonical_csv_bytes(second)


def test_friedman1_generator_matches_pinned_snapshot() -> None:
    entry = downloader.load_registry()["openml:friedman1"]
    payload = downloader.generator_payload(entry)

    assert hashlib.sha256(payload).hexdigest() == entry.sha256


def test_fresh_openml_csv_uses_local_sidecar_not_historical_csv_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = downloader.Dataset(
        key="openml:test",
        group="openml",
        target="data/test.csv",
        source_url="https://openml.org/data/v1/download/1/test.arff",
        sha256="0" * 64,
        license="test",
        source_type="openml",
        source_id="1",
        target_name="target",
        expected_rows=1,
        expected_features=1,
    )
    payload = b"feature,target\n1,2\n"
    monkeypatch.setattr(downloader, "openml_payload", lambda _entry: payload)

    target = downloader.download_one(entry, tmp_path, force=True)

    assert target.read_bytes() == payload
    assert downloader.sidecar_checksum(target) == hashlib.sha256(payload).hexdigest()


def test_allow_missing_does_not_hide_checksum_mismatch(tmp_path: Path) -> None:
    entry = downloader.Dataset(
        key="test:corrupt",
        group="test",
        target="data/corrupt.csv",
        source_url="https://example.invalid/corrupt.csv",
        sha256="0" * 64,
        license="test",
        source_type="url",
    )
    target = tmp_path / "data" / "corrupt.csv"
    target.parent.mkdir(parents=True)
    target.write_text("feature,target\n1,2\n", encoding="utf-8")

    assert downloader.command_verify([entry], tmp_path, allow_missing=True) == 1


def test_allow_missing_ignores_only_absent_files(tmp_path: Path) -> None:
    entry = downloader.Dataset(
        key="test:missing",
        group="test",
        target="data/missing.csv",
        source_url="https://example.invalid/missing.csv",
        sha256="0" * 64,
        license="test",
        source_type="url",
    )

    assert downloader.command_verify([entry], tmp_path, allow_missing=True) == 0
