#!/usr/bin/env python3
"""Minimal runtime proof for the canonical CrossGB interface."""

import json

import numpy as np
import xgboost as xgb


def main() -> None:
    rng = np.random.RandomState(2026)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    y = np.where(X[:, 0] <= 0, X[:, 1], -X[:, 1]).astype(np.float32)
    model = xgb.XGBRegressor(
        n_estimators=2,
        max_depth=2,
        tree_method="hist",
        device="cpu",
        enable_crossgb=True,
        crossgb_lambda_l2=0.0,
        base_score=0.0,
        n_jobs=1,
        random_state=1,
    )
    model.fit(X, y)
    pred = model.predict(X[:8])
    if pred.shape != (8,) or not np.isfinite(pred).all():
        raise RuntimeError("CrossGB smoke prediction failed.")

    config = json.loads(model.get_booster().save_config())
    booster = config["learner"]["gradient_booster"]
    if booster["updater"][0]["name"] != "grow_crossgb_histmaker":
        raise RuntimeError("Canonical CrossGB updater was not selected.")
    print("CrossGB smoke: OK")


if __name__ == "__main__":
    main()
