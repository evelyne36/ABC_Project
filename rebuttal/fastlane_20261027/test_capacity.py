"""Small deterministic tests requiring numpy only; no native library or data."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("capacity", Path(__file__).with_name("capacity.py"))
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def leaf_tree(feature=0, value=0.0):
    return {"left_children": [-1], "split_conditions": [value], "metagb_leaf_features": [feature]}


def fixture():
    X = np.array([[1., 1., 0.], [2., -1., 1.], [3., 2., -1.], [4., -2., 2.],
                  [5., 1., -2.], [6., -1., 3.]])
    y = X @ np.array([2., -1., .5]) + 3.
    splits = {"train": np.array([0, 1, 2, 3]), "val": np.array([4]), "test": np.array([5])}
    routes = {s: np.zeros((len(rows), 2), dtype=np.int64) for s, rows in splits.items()}
    return X, y, splits, routes, [leaf_tree(), leaf_tree()]


class CapacityTest(unittest.TestCase):
    def test_stored_native_rounds_products_and_accumulates_each_tree_in_float32(self):
        X = np.array([[1.23456789], [np.nan]], dtype=np.float64)
        trees = [leaf_tree(0, .123456789)] * 1024 + [leaf_tree(1, .1)]
        routes = np.zeros((len(X), len(trees)), dtype=np.int64)
        product = np.float32(.123456789) * np.float32(X[0, 0])
        expected = np.float32(.3)
        for _ in range(1024):
            expected = np.float32(expected + product)
        expected = np.float32(expected + np.float32(.1))
        actual = capacity.predict_stored_native(X, routes, trees, .3)
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(actual, [expected, np.float32(np.float32(.3) + np.float32(.1))])
        float64 = capacity.predict_frozen(X[:1], routes[:1], capacity.anchor_entries(trees, 1), .3)
        self.assertGreater(abs(float(actual[0]) - float64[0]), 1e-5)
        self.assertNotEqual(actual[0], np.float32(float64[0]))

    def test_main_gates_native_replay_and_retains_float64_diagnostic(self):
        trees = [leaf_tree(1, .1)] * 1024
        X = np.zeros((3, 1))
        routes = np.zeros((1, len(trees)), dtype=np.int64)
        native = capacity.predict_stored_native(X[:1], routes, trees, 0.)
        learner = {"objective": {"name": "reg:squarederror"},
                   "gradient_booster": {"name": "gbtree", "model": {"trees": trees}},
                   "learner_model_param": {"num_feature": "1", "base_score": "0"}}
        booster = mock.Mock()
        booster.save_raw.return_value = json.dumps({"learner": learner})
        booster.predict.return_value = native
        xgb = mock.Mock(__version__="test", Booster=mock.Mock(return_value=booster))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "data.npz", X=X, y=np.zeros(3), train=[0], val=[1], test=[2])
            (root / "result.json").write_text(json.dumps({"params": {"metagb_lambda_l2": 1., "eta": .1}}))
            (root / "model.json").write_text("{}")
            args = ["--root", directory, "--data", str(root / "data.npz"),
                    "--model-dir", directory, "--output", str(root / "output")]
            with mock.patch.object(capacity, "import_controlled_xgb", return_value=xgb), \
                    mock.patch.object(capacity, "crossgb_leaf_routes", return_value=routes), \
                    mock.patch("builtins.print"):
                self.assertEqual(capacity.main(args), 0)
                report = json.loads((root / "output" / "capacity.json").read_text())
                self.assertEqual(report["replay"]["tolerance_abs"], 1e-5)
                self.assertEqual(report["replay"]["max_abs_difference"], dict.fromkeys(capacity.SPLITS, 0.))
                self.assertGreater(report["replay"]["float64_max_abs_difference"]["train"], 1e-5)
                self.assertEqual(report["metrics"]["stored_anchor"]["rmse"]["train"], float(native[0]))
                booster.predict.return_value = native.astype(np.float64) + 2e-5
                with self.assertRaisesRegex(RuntimeError, "absolute tolerance 1e-5"):
                    capacity.main(args)

    def test_crossgb_routing_threshold_equality_and_default_child(self):
        tree = {"left_children": [1, -1, -1], "right_children": [2, -1, -1],
                "split_conditions": [1., 3., 7.], "split_indices": [0, 0, 0],
                "default_left": [True, False, False], "split_type": [0, 0, 0]}
        above = np.nextafter(np.float32(1.), np.float32(2.))
        X = np.array([[0.], [1.], [above], [np.nan]], dtype=np.float32)
        np.testing.assert_array_equal(capacity.crossgb_leaf_routes(X, [tree])[:, 0], [1, 1, 2, 1])
        # Exact threshold equality must use left (generic pred_leaf's < uses right).
        self.assertEqual(capacity.crossgb_leaf_routes(X[1:2], [tree])[0, 0], 1)
        tree["default_left"][0] = False
        self.assertEqual(capacity.crossgb_leaf_routes(X[3:], [tree])[0, 0], 2)
        tree["split_type"][0] = 1
        with self.assertRaises(ValueError):
            capacity.crossgb_leaf_routes(X, [tree])

    def test_worker_model_and_parameter_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.json").write_text("{}")
            metadata = {"effective_params": {"metagb_lambda_l2": 2., "learning_rate": .1},
                        "estimator_params": {"metagb_lambda_l2": 3., "learning_rate": .2}}
            (root / "result.json").write_text(json.dumps(metadata))
            result, ridge, eta, model, sources = capacity.load_saved_metadata(root)
            self.assertEqual((ridge, eta, model.name), (2., .1, "model.json"))
            self.assertEqual(sources, {"ridge": "effective_params.metagb_lambda_l2", "eta": "effective_params.learning_rate"})
            metadata["effective_params"] = {"metagb_lambda_l2": None}
            (root / "result.json").write_text(json.dumps(metadata))
            _, ridge, eta, _, sources = capacity.load_saved_metadata(root)
            self.assertEqual((ridge, eta), (3., .2))
            self.assertEqual(sources["ridge"], "estimator_params.metagb_lambda_l2")
            (root / "model.ubj").write_bytes(b"test")
            (root / "result.json").write_text(json.dumps({"params": {"metagb_lambda_l2": .5, "eta": .3}}))
            _, ridge, eta, model, sources = capacity.load_saved_metadata(root)
            self.assertEqual((ridge, eta, model.name), (.5, .3, "model.ubj"))
            self.assertEqual(sources["eta"], "params.eta")

    def test_each_variant_updates_own_residual(self):
        X, y, splits, routes, trees = fixture()
        report, pred = capacity.run_probe(X, y, splits, routes, trees, 0., .5, .4)
        train = splits["train"]
        for name in capacity.VARIANTS[1:]:
            entries = report["leaf_models"][name]
            first, second = entries[0][0], entries[1][0]
            first_pred = capacity.design(X[train], first["features"], first["affine"]) @ first["theta"]
            expected, _ = capacity.solve_leaf(X[train], y[train] - first_pred, second["features"], .5, second["affine"])
            np.testing.assert_allclose(second["theta"], .4 * expected, atol=1e-12)
            self.assertFalse(np.allclose(first["theta"], second["theta"]))
            np.testing.assert_allclose(pred[name]["train"], first_pred + capacity.design(X[train], second["features"], second["affine"]) @ second["theta"])

    def test_heldout_labels_do_not_change_fit_or_predictions(self):
        X, y, splits, routes, trees = fixture()
        original, predictions = capacity.run_probe(X, y, splits, routes, trees, 0., .5, .4)
        changed = y.copy()
        changed[np.r_[splits["val"], splits["test"]]] += 1e6
        altered, after = capacity.run_probe(X, changed, splits, routes, trees, 0., .5, .4)
        self.assertEqual(original["leaf_models"], altered["leaf_models"])
        for name in capacity.VARIANTS:
            for split in capacity.SPLITS:
                np.testing.assert_array_equal(predictions[name][split], after[name][split])
        self.assertNotEqual(original["metrics"]["affine_k3"]["rmse"]["test"], altered["metrics"]["affine_k3"]["rmse"]["test"])

    def test_empty_heldout_leaf_still_refits(self):
        # Leaf 1 has training rows but no val/test rows: it must still be fit.
        X, y, splits, _, _ = fixture()
        tree = {"left_children": [1, -1, -1], "split_conditions": [0., 999., 0.],
                "metagb_leaf_features": [0, 0, 0]}
        routes = {"train": np.array([[1], [1], [2], [2]]), "val": np.array([[2]]), "test": np.array([[2]])}
        report, _ = capacity.run_probe(X, y, splits, routes, [tree], 0., .5, .4)
        for name in capacity.VARIANTS[1:]:
            entry = report["leaf_models"][name][0][0]
            self.assertEqual(entry["leaf"], 1)
            self.assertEqual(entry["policy"], "training_only_refit")
            self.assertNotEqual(entry["theta"][0], 999.)
        # Move held-out rows to other leaf: training fits remain identical.
        other = copy.deepcopy(routes)
        other["val"][:] = 1
        other["test"][:] = 1
        changed, _ = capacity.run_probe(X, y, splits, other, [tree], 0., .5, .4)
        self.assertEqual(report["leaf_models"], changed["leaf_models"])

    def test_unpenalized_intercept_and_exact_k_counts(self):
        X = np.zeros((4, 3))
        theta, _ = capacity.solve_leaf(X, np.full(4, 7.), [0, 1], 1000., True)
        np.testing.assert_allclose(theta, [0., 0., 7.])
        X, y, splits, routes, trees = fixture()
        report, _ = capacity.run_probe(X, y, splits, routes, trees, 0., .5, .4)
        counts = report["metrics"]["affine_k3"]["counts"]
        self.assertEqual(counts["nonbias_slope_count"], 6)
        self.assertEqual(counts["affine_intercept_count"], 2)
        self.assertEqual(counts["total_scalar_parameters_including_base"], 9)
        self.assertEqual(counts["actual_k_histogram"], {"3": 2})

    def test_one_training_row_fits_and_unseen_training_leaf_falls_back(self):
        X, y, splits, _, _ = fixture()
        splits = {"train": np.array([0]), "val": np.array([4]), "test": np.array([5])}
        tree = {"left_children": [1, -1, -1], "split_conditions": [0., 999., 2.], "metagb_leaf_features": [0, 0, 0]}
        routes = {"train": np.array([[1]]), "val": np.array([[2]]), "test": np.array([[2]])}
        report, _ = capacity.run_probe(X, y, splits, routes, [tree], 0., .5, .4)
        fitted, fallback = report["leaf_models"]["affine_k3"][0]
        self.assertEqual(fitted["policy"], "training_only_refit")
        self.assertEqual(fallback["policy"], "no_training_rows_keep_stored_slope")
        self.assertEqual(fallback["theta"], [2., 0.])
        self.assertTrue(fallback["affine"])

    def test_constant_policy_and_stored_replay(self):
        X, y, splits, routes, trees = fixture()
        trees[0] = leaf_tree(3, 1.25)
        trees[1] = leaf_tree(1, -.2)
        report, pred = capacity.run_probe(X, y, splits, routes, trees, 2., .5, .4)
        for name in capacity.VARIANTS:
            entry = report["leaf_models"][name][0][0]
            self.assertEqual(entry["kind"], "constant")
            self.assertEqual(entry["theta"], [1.25])
            self.assertEqual(entry["k"], 0)
        for split in capacity.SPLITS:
            np.testing.assert_allclose(pred["stored_anchor"][split], 3.25 - .2 * X[splits[split], 1])

    def test_invalid_inputs_fail_closed(self):
        X, y, splits, routes, trees = fixture()
        with self.assertRaises(ValueError):
            capacity.run_probe(X, y, splits, routes, trees, 0., 0., .4)
        bad = dict(splits, val=splits["train"][:1])
        with self.assertRaises(ValueError):
            capacity.run_probe(X, y, bad, routes, trees, 0., .5, .4)
        bad_routes = copy.deepcopy(routes)
        bad_routes["test"][0, 0] = 999
        with self.assertRaises(ValueError):
            capacity.run_probe(X, y, splits, bad_routes, trees, 0., .5, .4)


if __name__ == "__main__":
    unittest.main()
