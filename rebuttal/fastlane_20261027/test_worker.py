"""Cheap worker-contract tests; no third-party estimator is substituted in runs."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

PATH = Path(__file__).with_name("worker.py")
SPEC = importlib.util.spec_from_file_location("fastlane_worker", PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data.npz"
        self.params = self.root / "params.json"
        self.params.write_text("{}")
        self.arrays = dict(X=np.arange(24, dtype=np.float32).reshape(8, 3),
                           y=np.arange(8, dtype=float), train=np.array([0, 1, 2, 3]),
                           val=np.array([4, 5]), test=np.array([6, 7]))
        self.save()

    def save(self):
        np.savez(self.data, **self.arrays)

    def args(self, final=False):
        argv = ["--root", str(self.root), "--data", str(self.data), "--method", "XGB",
                "--params", str(self.params), "--output", str(self.root / "output"),
                "--seed", "13"]
        return worker.parser().parse_args(argv + (["--evaluate-test"] if final else []))

    def test_search_does_not_validate_or_return_test_targets(self):
        self.arrays["y"][-2:] = np.nan
        self.save()
        _, targets, _, _ = worker.load_data(self.data)
        self.assertEqual(set(targets), {"train", "val"})
        with self.assertRaisesRegex(ValueError, "test targets"):
            worker.load_data(self.data, True)

    def test_separate_test_target_member_is_not_opened_in_search(self):
        del self.arrays["y"]
        self.arrays.update(y_train=np.arange(4.), y_val=np.arange(4., 6.),
                           y_test=np.array([object(), object()], dtype=object))
        self.save()
        worker.load_data(self.data)
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            worker.load_data(self.data, True)

    def test_disjoint_nonempty_integer_split_contract(self):
        for invalid in (np.array([3, 5]), np.array([], dtype=int), np.array([4., 5.]),
                        np.array([4, 4]), np.array([4, 8])):
            with self.subTest(invalid=invalid):
                self.arrays["val"] = invalid
                self.save()
                with self.assertRaises(ValueError):
                    worker.load_data(self.data)

    def test_classification_encoding_is_explicit(self):
        self.arrays.update(task_type=np.array("binary"), y=np.array([0, 1] * 4))
        self.save()
        self.assertEqual(worker.load_data(self.data)[3], "binary")
        self.arrays["y"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "encoded"):
            worker.load_data(self.data)

    def test_classification_accuracy_uses_probability_argmax(self):
        # Accuracy has no new dependency. Existing sklearn metrics are stubbed
        # so this contract check also runs in the minimal NumPy-only runtime.
        existing_metrics = types.ModuleType("sklearn.metrics")
        existing_metrics.log_loss = mock.Mock(return_value=0.123)
        existing_metrics.roc_auc_score = mock.Mock(return_value=0.8)
        with mock.patch.dict(sys.modules, {"sklearn.metrics": existing_metrics}):
            binary = worker.metrics("binary", np.array([0, 1, 1, 0]),
                                    np.array([[.8, .2], [.1, .9], [.7, .3], [.6, .4]]))
            self.assertEqual(binary["accuracy"], .75)
            self.assertEqual(binary["logloss"], .123)
            self.assertEqual(binary["auc"], .8)
            multi = worker.metrics("multiclass", np.array([0, 2, 1]),
                                   np.array([[.7, .2, .1], [.1, .8, .1], [.1, .8, .1]]))
            self.assertAlmostEqual(multi["accuracy"], 2 / 3)
            self.assertIn("logloss", multi)
            self.assertNotIn("auc", multi)

    def test_cli_defaults_and_all_methods(self):
        self.assertFalse(self.args().evaluate_test)
        self.assertFalse(self.args().save_model)
        self.assertEqual(len(worker.METHODS), 7)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")

    def test_missing_runtime_becomes_failure_json(self):
        command = [sys.executable, str(PATH), "--root", str(self.root), "--data", str(self.data),
                   "--method", "XGB", "--params", str(self.params), "--output",
                   str(self.root / "failed"), "--seed", "13"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        record = json.loads((self.root / "failed/result.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_stage"], "make_model")
        self.assertIn("Missing rebuttal_execution", record["error"])
        self.assertFalse(record["test_labels_used"])
        self.assertNotIn("test", record["target_hashes"])

    def test_mocked_fit_contract_timing_and_saved_predictions(self):
        # The mock verifies orchestration only. Real native fitting is a Linux
        # deployment smoke check, because this host has no controlled Linux lib.
        class Estimator:
            def fit(inner, X, y):
                np.testing.assert_array_equal(X, self.arrays["X"][:4])
                np.testing.assert_array_equal(y, self.arrays["y"][:4])
                return inner

            def predict(inner, X):
                return X[:, 0] / 3

        threadpool = types.SimpleNamespace(threadpool_limits=lambda **_: contextlib.nullcontext())
        for final in (False, True):
            with self.subTest(final=final), mock.patch.object(
                    worker, "make_model", return_value=(Estimator(), {"mock": True}, {})), \
                    mock.patch.object(worker, "model_info", return_value={}), \
                    mock.patch.dict(sys.modules, {"threadpoolctl": threadpool}):
                self.assertEqual(worker.run(self.args(final)), 0)
                record = json.loads((self.root / "output/result.json").read_text())
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["test_labels_used"], final)
                self.assertEqual(record["val_rmse"], 0)
                splits = {"train", "val", "test"} if final else {"train", "val"}
                self.assertEqual(set(record["metrics"]), splits)
                for phase in ["fit"] + ["predict_" + split for split in splits]:
                    self.assertGreaterEqual(record["timings"][phase + "_wall_sec"], 0)
                    self.assertGreaterEqual(record["timings"][phase + "_cpu_sec"], 0)
                with np.load(self.root / "output/predictions.npz") as predictions:
                    self.assertEqual(set(predictions.files), splits | {s + "_indices" for s in splits})

    def test_failed_fit_keeps_wall_cpu_time(self):
        class BrokenEstimator:
            def fit(inner, X, y):
                raise RuntimeError("synthetic-fit-error")
        threadpool = types.SimpleNamespace(threadpool_limits=lambda **_: contextlib.nullcontext())
        with mock.patch.object(worker, "make_model", return_value=(BrokenEstimator(), {}, {})), \
                mock.patch.dict(sys.modules, {"threadpoolctl": threadpool}):
            self.assertEqual(worker.run(self.args()), 1)
        record = json.loads((self.root / "output/result.json").read_text())
        self.assertEqual(record["failure_stage"], "fit")
        self.assertIn("fit_wall_sec", record["timings"])
        self.assertIn("fit_cpu_sec", record["timings"])
        self.assertEqual(record["error"], "synthetic-fit-error")

    def test_nonfinite_estimator_metadata_serializes_without_mutation(self):
        params = {"missing": np.nan, "nested": [np.float64(np.inf), {"sentinel": -np.inf}]}
        record = {"status": "ok", "estimator_params": params, "metrics": {"val": {"rmse": 1.0}}}
        worker.write_result(self.root, record)
        saved = json.loads((self.root / "result.json").read_text())
        self.assertEqual(saved["estimator_params"],
                         {"missing": "NaN", "nested": ["+Infinity", {"sentinel": "-Infinity"}]})
        self.assertTrue(np.isnan(params["missing"]))
        self.assertTrue(np.isinf(params["nested"][0]))
        self.assertEqual(saved["metrics"]["val"]["rmse"], 1.0)
        record["metrics"]["val"]["rmse"] = np.nan
        with self.assertRaises(ValueError):
            worker.write_result(self.root, record)


if __name__ == "__main__":
    unittest.main()
