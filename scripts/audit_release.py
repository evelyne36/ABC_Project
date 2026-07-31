#!/usr/bin/env python3
"""Audit the CrossGB source tree for common submission-package mistakes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_READMES = (
    "README.md",
    "CrossGB/README.md",
    "exp/random300/README.md",
    "exp/tabular_extra_random300/README.md",
    "exp/random300_baselinemodel/README.md",
    "exp/synthetic_crossgb/README.md",
    "exp/synthetic_baselinemodel/README.md",
    "exp/crossgb_ablation/README.md",
    "exp/timeseries_extra_random300/README.md",
)
GENERATED_DIR_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "logs",
    "nohup_logs",
    "results",
    "tmp_mplconfig",
    ".codex",
    ".omx",
    ".claude",
    ".vscode",
    ".venv",
}
BINARY_SUFFIXES = {".a", ".dll", ".dylib", ".o", ".pyc", ".so"}
TEXT_SUFFIXES = {
    ".cc",
    ".cpp",
    ".h",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}


def release_files() -> list[Path]:
    """Return the actual archive contents, independent of an enclosing Git tree."""

    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_required_readmes(errors: list[str]) -> None:
    for name in REQUIRED_READMES:
        if not (ROOT / name).is_file():
            errors.append(f"missing required README: {name}")


def check_generated_artifacts(files: list[Path], errors: list[str]) -> None:
    for path in files:
        rel = path.relative_to(ROOT)
        if path.name == ".DS_Store" or path.name.startswith("._"):
            errors.append(f"OS metadata is still included: {rel.as_posix()}")
        if any(part in GENERATED_DIR_NAMES for part in rel.parts):
            errors.append(f"generated artifact is still included: {rel.as_posix()}")
        if rel.parts[0].startswith("docs_"):
            errors.append(f"historical docs_* directory is still included: {rel.as_posix()}")
        if (
            len(rel.parts) >= 3
            and rel.parts[0] == "exp"
            and path.suffix.lower() == ".md"
            and path.name != "README.md"
        ):
            errors.append(f"non-canonical experiment Markdown is still included: {rel.as_posix()}")
        if path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"compiled/binary artifact is still included: {rel.as_posix()}")
        if rel.parts[0] == "dataset":
            errors.append(f"dataset file is still included: {rel.as_posix()}")
        if len(rel.parts) >= 3 and rel.parts[0] == "exp" and rel.parts[2] == "data":
            errors.append(f"experiment dataset file is still included: {rel.as_posix()}")


def check_active_text(files: list[Path], errors: list[str]) -> None:
    stale_root = re.compile(r"\bbeforereview/")
    private_project_marker = "check_" + "codereview"
    personal_path = re.compile(
        rf"(?:/(?:Users|home)/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\|{private_project_marker})",
        re.IGNORECASE,
    )
    missing_import = re.compile(r"^\s*(?:from|import)\s+random600\b", re.MULTILINE)
    forbidden_crossgb_names = (
        "meta" + "gb",
        "Meta" + "GB",
        "Meta" + "gb",
        "META" + "GB",
        "m" + "gb",
    )
    for path in files:
        rel = relative(path)
        if any(term in rel for term in forbidden_crossgb_names):
            errors.append(f"noncanonical CrossGB name in path: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if stale_root.search(text):
            errors.append(f"obsolete beforereview path in active file: {rel}")
        if personal_path.search(text):
            errors.append(f"personal absolute path or private project name in active file: {rel}")
        if missing_import.search(text):
            errors.append(f"missing random600 package import in active file: {rel}")
        if any(term in text for term in forbidden_crossgb_names):
            errors.append(f"noncanonical CrossGB name in active file: {rel}")


def check_json(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {relative(path)}: {exc}")


def main() -> int:
    files = release_files()
    errors: list[str] = []
    if not files:
        errors.append("release audit found no files")
    check_required_readmes(errors)
    check_generated_artifacts(files, errors)
    check_active_text(files, errors)
    check_json(files, errors)

    if errors:
        print("CrossGB release audit: FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"CrossGB release audit: OK ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
