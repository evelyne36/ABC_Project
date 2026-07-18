from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300_baselinemodel.utils.candidates import BASELINE_MODEL_ORDER, candidate_sets
from random300_baselinemodel.utils.candidates import DEFAULT_SEARCH_SPACE as BASELINE_SEARCH_SPACE
from random300_baselinemodel.utils.schedule import create_ordered_tasks


def test_baseline_candidate_sets_are_reproducible_and_cover_requested_models():
    first = candidate_sets(models=BASELINE_MODEL_ORDER, max_candidates=5, seed=42)
    second = candidate_sets(models=BASELINE_MODEL_ORDER, max_candidates=5, seed=42)

    assert first == second
    assert list(first) == BASELINE_MODEL_ORDER
    assert all(len(candidates) == 5 for candidates in first.values())
    assert first["linear"][0]["candidate_id"] == "linear_default"
    assert first["torchmlp"][0]["candidate_id"] == "torchmlp_default"


def test_all_baseline_models_can_generate_random300_candidates():
    candidates = candidate_sets(models=BASELINE_MODEL_ORDER, max_candidates=300, seed=42)

    assert list(candidates) == BASELINE_MODEL_ORDER
    for model, model_candidates in candidates.items():
        assert len(model_candidates) == 300
        candidate_ids = [candidate["candidate_id"] for candidate in model_candidates]
        assert len(set(candidate_ids)) == 300
        assert candidate_ids[0] == f"{model}_default"


def test_boosting_spaces_align_with_metagb_xgb_shared_backbone():
    xgb_values = {
        "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.1},
        "n_estimators": {"type": "choice", "values": [200, 400, 600, 800]},
        "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
        "lambda_l2": {"type": "loguniform", "low": 0.0001, "high": 5.0},
    }

    lightgbm = BASELINE_SEARCH_SPACE["lightgbm"]["space"]
    catboost = BASELINE_SEARCH_SPACE["catboost"]["space"]
    assert lightgbm["learning_rate"] == xgb_values["learning_rate"]
    assert lightgbm["n_estimators"] == xgb_values["n_estimators"]
    assert lightgbm["subsample"] == xgb_values["subsample"]
    assert lightgbm["colsample_bytree"] == xgb_values["colsample_bytree"]
    assert lightgbm["reg_lambda"] == xgb_values["lambda_l2"]

    assert catboost["learning_rate"] == xgb_values["learning_rate"]
    assert catboost["iterations"] == xgb_values["n_estimators"]
    assert catboost["l2_leaf_reg"] == xgb_values["lambda_l2"]
    assert catboost["depth"] == {"type": "logint", "low": 4, "high": 10}


def test_tasks_are_model_first_then_categorical_then_light_datasets_then_folds():
    args = SimpleNamespace(
        task_types=["regression", "binary"],
        categorical_handlings=["onehot", "drop"],
        models=["catboost", "linear"],
        regression_datasets=["regression__cpu_activity.csv", "regression__Moneyball.csv"],
        binary_datasets=["binary__electricity.csv", "binary__haberman.csv"],
        skip_datasets=[],
        skip_regression_datasets=[],
        skip_binary_datasets=[],
        skip_multiclass_datasets=[],
        run_all_outer_folds=True,
        mode="full",
        outer_folds=2,
        outer_fold_index=0,
    )
    tasks = create_ordered_tasks(args, {"linear": [{}], "catboost": [{}]})

    first_model = [task for task in tasks if task["model"] == "catboost"]
    second_model = [task for task in tasks if task["model"] == "linear"]
    assert tasks == first_model + second_model

    catboost_keys = [
        (task["categorical_handling"], task["task_type"], task["dataset"], task["outer_index"])
        for task in first_model
    ]
    assert catboost_keys[:4] == [
        ("drop", "binary", "binary__haberman.csv", 0),
        ("drop", "binary", "binary__haberman.csv", 1),
        ("drop", "regression", "regression__Moneyball.csv", 0),
        ("drop", "regression", "regression__Moneyball.csv", 1),
    ]
    assert all(key[0] == "drop" for key in catboost_keys[:8])
    assert all(key[0] == "onehot" for key in catboost_keys[8:])
