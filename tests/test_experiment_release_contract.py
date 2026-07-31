from __future__ import annotations

import ast
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def argument_default(path: Path, option: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"No literal default for {option} in {path}")


class ExperimentReleaseContractTests(unittest.TestCase):
    def test_submitted_tabular_runners_default_to_onehot(self) -> None:
        scripts = [
            ROOT / "exp/random300/scripts/exp1_random300_all.py",
            ROOT / "exp/random300/scripts/run_random300_server.py",
            ROOT / "exp/tabular_extra_random300/scripts/exp1_random300_all.py",
            ROOT / "exp/tabular_extra_random300/scripts/run_random300_server.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                self.assertEqual(argument_default(script, "--categorical-handling"), "onehot")
        full_wrapper = (
            ROOT / "exp/random300/scripts/run_random300_full.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'add_default(defaulted, "--categorical-handling", ["onehot"])',
            full_wrapper,
        )

    def test_shell_launchers_default_to_onehot(self) -> None:
        scripts = [
            ROOT / "exp/tabular_extra_random300/run_linux_full.sh",
            ROOT / "exp/tabular_extra_random300/run_linux_comprehensive.sh",
        ]
        pattern = re.compile(r'HANDLING="\$\{HANDLING:-onehot\}"')
        for script in scripts:
            with self.subTest(script=script):
                self.assertRegex(script.read_text(encoding="utf-8"), pattern)

    def test_synthetic_main_dimensions_are_10_and_20(self) -> None:
        tree = ast.parse(
            (ROOT / "exp/synthetic_crossgb/protocols.py").read_text(encoding="utf-8")
        )
        dimensions = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "DIMENSIONS"
                for target in node.targets
            ):
                dimensions = ast.literal_eval(node.value)
                break
        self.assertEqual(dimensions, [10, 20])

    def test_baseline_regression_panel_contains_qsar(self) -> None:
        path = ROOT / "exp/random300_baselinemodel/utils/schedule.py"
        spec = importlib.util.spec_from_file_location("baseline_release_schedule", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn("QSAR_fish_toxicity.csv", module.DEFAULT_TASK_DATASETS["regression"])
        self.assertEqual(len(module.DEFAULT_TASK_DATASETS["regression"]), 10)
        self.assertEqual(len(module.DEFAULT_TASK_DATASETS["binary"]), 10)

        with tempfile.TemporaryDirectory() as temporary:
            primary = Path(temporary) / "primary"
            fallback = Path(temporary) / "fallback"
            primary.mkdir()
            fallback.mkdir()
            qsar = fallback / "QSAR_fish_toxicity.csv"
            qsar.touch()
            args = SimpleNamespace(
                regression_dir=str(primary),
                regression_fallback_dirs=[str(fallback)],
                binary_dir=str(primary),
                binary_fallback_dirs=[],
                multiclass_dir=str(primary),
                multiclass_fallback_dirs=[],
            )
            self.assertEqual(
                module.resolve_dataset_path(args, "regression", qsar.name), qsar.resolve()
            )

    def test_build_scripts_support_an_official_checkout_override(self) -> None:
        for name in (
            "preflight_crossgb.sh",
            "build_crossgb.sh",
            "verify_crossgb.sh",
            "run_targeted_tests.sh",
        ):
            source = (ROOT / "CrossGB/scripts" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("CROSSGB_XGBOOST_ROOT", source)


if __name__ == "__main__":
    unittest.main()
