#!/usr/bin/env python3
"""Verify canonical CrossGB model serialization and loading."""

import json
import tempfile
from pathlib import Path

import numpy as np
import xgboost as xgb


def main() -> None:
    rng = np.random.RandomState(1994)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    y = np.where(X[:, 0] <= 0, X[:, 1], -X[:, 1]).astype(np.float32)
    model = xgb.XGBRegressor(
        n_estimators=3,
        max_depth=2,
        tree_method="hist",
        enable_crossgb=True,
        crossgb_lambda_l2=0.0,
        base_score=0.0,
        n_jobs=1,
        random_state=1,
    )
    model.fit(X, y)
    expected = model.predict(X)

    with tempfile.TemporaryDirectory(prefix="crossgb-roundtrip-") as tmp:
        canonical_path = Path(tmp) / "crossgb.json"
        model.save_model(canonical_path)
        payload = json.loads(canonical_path.read_text())
        trees = payload["learner"]["gradient_booster"]["model"]["trees"]
        for tree in trees:
            if "crossgb_leaf_features" not in tree:
                raise RuntimeError("Canonical CrossGB serialization field is missing.")

        loaded = xgb.XGBRegressor()
        loaded.load_model(canonical_path)
        np.testing.assert_allclose(expected, loaded.predict(X))
    print("CrossGB canonical round trip: OK")


if __name__ == "__main__":
    main()
