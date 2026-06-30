from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mgb_datacollection_by0629.preflight_adapter_0629.scripts.score_mechanism_alignment_0629 import (
    build_scores,
    score_band,
)


def test_mechanism_score_ranks_structural_match_above_negative_control():
    rows = pd.DataFrame(
        [
            {
                "dataset_key": "positive",
                "family": "synthetic_0629",
                "projected_gradient_lift_mean": 7.0,
                "local_linearity_lift": 1.7,
                "decoupled_leaf_fraction": 0.75,
                "predictive_score_concentration_mean": 0.6,
                "local_abs_corr_max_mean": 0.55,
                "global_abs_corr_max": 0.30,
                "runtime_pressure_log": 0.2,
                "raw_rows": 3000,
                "processed_features": 10,
            },
            {
                "dataset_key": "negative",
                "family": "synthetic_0629",
                "projected_gradient_lift_mean": 0.8,
                "local_linearity_lift": 0.95,
                "decoupled_leaf_fraction": 0.20,
                "predictive_score_concentration_mean": 0.30,
                "local_abs_corr_max_mean": 0.12,
                "global_abs_corr_max": 0.85,
                "runtime_pressure_log": 0.2,
                "raw_rows": 3000,
                "processed_features": 10,
            },
        ]
    )

    scores = build_scores(rows)

    positive = scores.loc[scores["dataset_key"].eq("positive")].iloc[0]
    negative = scores.loc[scores["dataset_key"].eq("negative")].iloc[0]
    assert positive["mechanism_alignment_score"] > negative["mechanism_alignment_score"]
    assert positive["mechanism_alignment_score"] >= 0.6
    assert negative["mechanism_alignment_score"] < 0.35


def test_score_band_uses_coarse_interpretation_thresholds():
    assert score_band(0.20, "synthetic") == "LOW_MGB_FIT"
    assert score_band(0.45, "synthetic") == "MEDIUM_MGB_FIT"
    assert score_band(0.70, "synthetic") == "HIGH_MGB_FIT"
    assert score_band(0.45, "tabarena") == "MEDIUM_MGB_FIT"
    assert score_band(0.55, "tabarena") == "HIGH_MGB_FIT"
    assert score_band(0.55, "synthetic") == "MEDIUM_MGB_FIT"
