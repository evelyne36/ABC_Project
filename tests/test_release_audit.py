from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_release.py"
SPEC = importlib.util.spec_from_file_location("audit_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_release)


class ReleaseAuditTests(unittest.TestCase):
    def test_release_files_scans_archive_contents_including_ignored_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[3]) as directory:
            root = Path(directory)
            (root / "README.md").write_text("# Test\n", encoding="utf-8")
            (root / ".DS_Store").write_bytes(b"metadata")
            with mock.patch.object(audit_release, "ROOT", root):
                files = {
                    path.relative_to(root).as_posix()
                    for path in audit_release.release_files()
                }
        self.assertEqual(files, {".DS_Store", "README.md"})

    def test_generated_artifact_check_rejects_ds_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / ".DS_Store"
            metadata.write_bytes(b"metadata")
            errors: list[str] = []
            with mock.patch.object(audit_release, "ROOT", root):
                audit_release.check_generated_artifacts([metadata], errors)
        self.assertTrue(any(".DS_Store" in error for error in errors))

    def test_active_text_rejects_personal_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "README.md"
            document.write_text(
                "Copy from /" + "Users/reviewer/private/source before running.\n",
                encoding="utf-8",
            )
            errors: list[str] = []
            with mock.patch.object(audit_release, "ROOT", root):
                audit_release.check_active_text([document], errors)
        self.assertTrue(any("personal absolute path" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
