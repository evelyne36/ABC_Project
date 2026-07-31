from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^([A-Za-z0-9_.-]+)")


def requirement_names(path: Path, seen: set[Path] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return set()
    seen.add(path)

    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r "):
            names.update(requirement_names(path.parent / line[3:].strip(), seen))
            continue
        match = NAME.match(line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


class RequirementsManifestTests(unittest.TestCase):
    def test_root_manifest_is_single_install_surface(self) -> None:
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertFalse(
            any(line.strip().startswith(("-r ", "--requirement ")) for line in lines),
            "root requirements.txt must list dependencies directly",
        )

    def test_root_manifest_covers_every_experiment_manifest(self) -> None:
        root_names = requirement_names(ROOT / "requirements.txt")
        child_names: set[str] = set()
        for manifest in (ROOT / "exp").glob("*/requirements*.txt"):
            child_names.update(requirement_names(manifest))
        self.assertEqual(
            child_names - root_names,
            set(),
            "root requirements.txt must install every submitted experiment dependency",
        )

    def test_root_manifest_does_not_install_pypi_xgboost(self) -> None:
        self.assertNotIn("xgboost", requirement_names(ROOT / "requirements.txt"))


if __name__ == "__main__":
    unittest.main()
