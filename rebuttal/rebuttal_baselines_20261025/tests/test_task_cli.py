"""An absent/incorrect result or use of test labels must fail these tests."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_single_task_writes_reproducible_predictions_and_no_test_label_fit(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 2))
    y = 3 * X[:, 0] - X[:, 1]
    path = tmp_path / "input.npz"
    np.savez(path, X_train=X[:60], y_train=y[:60], X_predict=X[60:], y_evaluate=y[60:])
    output = tmp_path / "result"
    result = subprocess.run([sys.executable, str(ROOT/"scripts/run_task.py"),
        "--method", "linxgboost", "--input", str(path), "--output", str(output),
        "--params", '{"n_estimators":2,"max_depth":1,"lbda":0.1}'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output/"result.json").read_text())
    pred = np.load(output/"predictions.npz")["prediction"]
    assert pred.shape == (20,) and np.isfinite(pred).all()
    assert report["metadata"]["test_labels_passed_to_fit"] is False
    assert abs(report["rmse"] - np.sqrt(np.mean((pred-y[60:])**2))) < 1e-12
    assert report["source"]["revision"] == "0166fb3682a0e534c54aea0559186fee26192d6f"
    # Existing outputs are protected against accidental overwrite.
    again = subprocess.run([sys.executable, str(ROOT/"scripts/run_task.py"), "--method", "linxgboost",
        "--input", str(path), "--output", str(output)], capture_output=True, text=True)
    assert again.returncode != 0
