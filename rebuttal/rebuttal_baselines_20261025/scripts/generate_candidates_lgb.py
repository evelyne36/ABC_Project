#!/usr/bin/env python3
"""Freeze the LightGBM search space (constant and linear_tree share it); no fitting.

Structured exactly like `scripts/generate_candidates.py` (same RNG, same
`pick`/`log` helpers, same "author starting configuration first" rule) so the
LightGBM line is searched with the same style of budget as the other
neighbours.  `linear_lambda` is ignored by the constant variant and dropped
there by `lgb_ext/adapter.py`.

This space is this rebuttal's own frozen follow-up space.  It is NOT an
author-paper exact search reproduction, and 300 configurations is only
comparable to the other neighbours because they use the same count.
"""
import argparse
import json
from pathlib import Path

import numpy as np

AUTHOR_START = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_child_samples": 100,
    "reg_lambda": 0.01,
    "reg_alpha": 0.01,
    "colsample_bytree": 1.0,
    "linear_lambda": 0.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    rng = np.random.default_rng(args.seed)

    def pick(values):
        return values[int(rng.integers(len(values)))]

    def log(a, b):
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))

    rows = [dict(AUTHOR_START)]
    while len(rows) < args.count:
        row = {
            "n_estimators": pick([10, 20, 50, 100, 200, 400, 800]),
            "learning_rate": log(0.005, 0.3),
            "num_leaves": pick([4, 8, 16, 31, 63, 127, 255]),
            "min_child_samples": pick([2, 5, 10, 20, 50, 100]),
            "reg_lambda": log(1e-4, 100),
            "reg_alpha": 0.0 if rng.random() < 0.5 else log(1e-4, 1),
            "colsample_bytree": pick([0.6, 0.8, 1.0]),
            "linear_lambda": 0.0 if rng.random() < 0.1 else log(1e-6, 10),
        }
        if row not in rows:
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(rows, output, indent=2)
    print(json.dumps({"methods": ["lgb_constant", "lgb_linear"], "count": len(rows),
                      "search_seed": args.seed, "model_fits": 0,
                      "space_scope": "rebuttal follow-up LightGBM space; not an author-paper exact search reproduction"}))


if __name__ == "__main__":
    main()
