from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300.utils.candidates import candidate_sets as random300_candidate_sets
from random300.utils.candidates import load_search_space as load_random300_search_space
from random600.utils.candidates import candidate_sets as random600_candidate_sets
from random600.utils.candidates import load_search_space as load_random600_search_space
from random600.utils.runner import build_estimator


MODELS = ["xgb_cpu", "current_metagb", "current_metagb_pred0"]
COMMON_KEYS = [
    "learning_rate",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "n_estimators",
    "max_bin",
]


def _candidate_sets(loader, generator, suite: str, count: int, seed: int = 42):
    search_space = loader(ROOT / suite / "configs" / "default_search_space.json")
    candidates, warnings = generator(
        search_space=search_space,
        max_candidates=count,
        seed=seed,
        config_path=ROOT / "beforereview" / "exp12" / "configs_xgboost.json",
        models=MODELS,
    )
    return candidates, warnings


@pytest.mark.parametrize(
    ("loader", "generator", "suite"),
    [
        (load_random300_search_space, random300_candidate_sets, "random300"),
        (load_random600_search_space, random600_candidate_sets, "random600"),
    ],
)
def test_candidates_are_reproducible_and_paired(loader, generator, suite):
    first, warnings = _candidate_sets(loader, generator, suite, 300)
    second, _ = _candidate_sets(loader, generator, suite, 300)

    assert first == second
    assert warnings == ["xgb config file is ignored; candidates are generated from the shared search space"]

    for xgb, metagb, pred0 in zip(
        first["xgb_cpu"],
        first["current_metagb"],
        first["current_metagb_pred0"],
    ):
        assert xgb["backbone_id"] == metagb["backbone_id"] == pred0["backbone_id"]
        for key in COMMON_KEYS:
            assert xgb[key] == metagb[key] == pred0[key]
        assert xgb["reg_lambda"] == metagb["metagb_lambda_l2"]
        assert xgb["reg_lambda"] == pred0["metagb_lambda_l2"]
        assert pred0["pred_feature_fraction"] == 0.0
        assert pred0["metagb_score_agg"] == 0


@pytest.mark.parametrize(
    ("loader", "generator", "suite"),
    [
        (load_random300_search_space, random300_candidate_sets, "random300"),
        (load_random600_search_space, random600_candidate_sets, "random600"),
    ],
)
def test_random300_is_balanced_prefix_of_random600(loader, generator, suite):
    candidates_300, _ = _candidate_sets(loader, generator, suite, 300)
    candidates_600, _ = _candidate_sets(loader, generator, suite, 600)

    for model in MODELS:
        assert candidates_300[model] == candidates_600[model][:300]

    pred_counts_300 = Counter(c["pred_feature_fraction"] for c in candidates_300["current_metagb"])
    pred_counts_600 = Counter(c["pred_feature_fraction"] for c in candidates_600["current_metagb"])
    score_counts_300 = Counter(c["metagb_score_agg"] for c in candidates_300["current_metagb"])
    score_counts_600 = Counter(c["metagb_score_agg"] for c in candidates_600["current_metagb"])
    combo_counts_300 = Counter(
        (c["pred_feature_fraction"], c["metagb_score_agg"])
        for c in candidates_300["current_metagb"]
    )
    combo_counts_600 = Counter(
        (c["pred_feature_fraction"], c["metagb_score_agg"])
        for c in candidates_600["current_metagb"]
    )

    assert pred_counts_300 == {round(i * 0.1, 1): 30 for i in range(1, 11)}
    assert pred_counts_600 == {round(i * 0.1, 1): 60 for i in range(1, 11)}
    assert score_counts_300 == {0: 100, 1: 100, 2: 100}
    assert score_counts_600 == {0: 200, 1: 200, 2: 200}
    assert set(combo_counts_300.values()) == {10}
    assert set(combo_counts_600.values()) == {20}
    assert len(combo_counts_300) == len(combo_counts_600) == 30


def test_random300_and_random600_suites_generate_identical_candidates():
    random300, _ = _candidate_sets(
        load_random300_search_space,
        random300_candidate_sets,
        "random300",
        600,
    )
    random600, _ = _candidate_sets(
        load_random600_search_space,
        random600_candidate_sets,
        "random600",
        600,
    )
    assert random300 == random600


def test_different_seed_changes_shared_backbones():
    first, _ = _candidate_sets(
        load_random600_search_space,
        random600_candidate_sets,
        "random600",
        300,
        seed=42,
    )
    second, _ = _candidate_sets(
        load_random600_search_space,
        random600_candidate_sets,
        "random600",
        300,
        seed=43,
    )
    assert first != second


def test_metagb_candidate_count_must_cover_complete_joint_cycles():
    search_space = load_random600_search_space(
        ROOT / "random600" / "configs" / "default_search_space.json"
    )
    with pytest.raises(ValueError, match="divisible by 30"):
        random600_candidate_sets(
            search_space=search_space,
            max_candidates=300 - 1,
            seed=42,
            config_path=None,
            models=["current_metagb"],
        )


class _CapturedEstimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeXGBoost:
    XGBClassifier = _CapturedEstimator
    XGBRegressor = _CapturedEstimator


def test_estimator_parameters_follow_comparison_contract():
    candidates, _ = _candidate_sets(
        load_random600_search_space,
        random600_candidate_sets,
        "random600",
        300,
    )
    args = SimpleNamespace(
        binary_objective="binary:logistic",
        multiclass_objective="multi:softprob",
        seed=42,
        n_jobs=1,
        xgb_reg_lambda=1.0,
        metagb_hist_subtract=False,
        metagb_score_agg=0,
        metagb_lambda_l2=1.0,
        metagb_pred_value_type="raw",
    )

    xgb_params = build_estimator(
        _FakeXGBoost,
        "xgb_cpu",
        candidates["xgb_cpu"][2],
        args,
        "regression",
        0,
    ).kwargs
    metagb_params = build_estimator(
        _FakeXGBoost,
        "current_metagb",
        candidates["current_metagb"][2],
        args,
        "regression",
        0,
    ).kwargs

    assert xgb_params["min_child_weight"] == 1.0
    assert xgb_params["reg_alpha"] == 0.0
    assert xgb_params["grow_policy"] == "depthwise"
    assert xgb_params["colsample_bylevel"] == 1.0
    assert xgb_params["colsample_bynode"] == 1.0
    assert xgb_params["reg_lambda"] == metagb_params["metagb_lambda_l2"]
    assert metagb_params["metagb_score_agg"] == 2
    assert metagb_params["metagb_min_data_in_leaf"] == 1
    assert metagb_params["metagb_cond_pred_share"] is False
    assert metagb_params["metagb_leaf_only_use_bias"] is False
