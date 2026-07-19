from __future__ import annotations

from typing import Any, Dict, List


EXPLICIT_COMMON_PARAMS = [
    "learning_rate",
    "n_estimators",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "max_bin",
    "tree_method",
    "device",
    "objective",
    "random_state",
    "n_jobs",
    "verbosity",
]

EXPLICIT_XGB_PARAMS = ["reg_lambda"]
EXPLICIT_XGB_FIXED_PARAMS = [
    "min_child_weight",
    "reg_alpha",
    "grow_policy",
    "colsample_bylevel",
    "colsample_bynode",
]

EXPLICIT_METAGB_PARAMS = [
    "enable_metagb",
    "metagb_pred_colsample_bytree",
    "metagb_lambda_l2",
    "metagb_score_agg",
    "metagb_hist_subtract",
    "metagb_min_data_in_leaf",
    "metagb_cond_pred_share",
    "metagb_leaf_only_use_bias",
]

DEFAULTED_XGB_PARAMS = [
    "gamma/min_split_loss",
    "max_delta_step",
    "max_leaves",
    "sampling_method",
    "scale_pos_weight",
    "base_score",
    "max_cat_to_onehot",
    "max_cat_threshold",
]

DEFAULTED_METAGB_PARAMS = [
    "metagb_split_sample",
    "metagb_min_gain_to_split",
]


def build_parameter_audit(args: Any, search_space: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    searched_shared = sorted(search_space.get("shared", {}).keys())
    searched_xgb = sorted(search_space.get("xgb_cpu", {}).keys())
    searched_metagb = sorted(search_space.get("current_metagb", {}).keys())
    return {
        "explicitly_set": {
            "common": EXPLICIT_COMMON_PARAMS,
            "xgb_cpu": EXPLICIT_XGB_PARAMS + EXPLICIT_XGB_FIXED_PARAMS,
            "current_metagb": EXPLICIT_METAGB_PARAMS,
            "classification_extra": ["eval_metric", "num_class for multiclass"],
        },
        "searched_parameters": {
            "shared": searched_shared,
            "xgb_cpu": searched_xgb,
            "current_metagb": searched_metagb + ["pred_feature_fraction", "metagb_score_agg"],
        },
        "kept_as_xgboost_defaults": DEFAULTED_XGB_PARAMS,
        "kept_as_metagb_defaults_or_fixed": DEFAULTED_METAGB_PARAMS,
        "fixed_values": {
            "tree_method": "hist",
            "device": "cpu",
            "regression_objective": "reg:squarederror",
            "binary_objective": args.binary_objective,
            "multiclass_objective": args.multiclass_objective,
            "categorical_handling": args.categorical_handling,
            "metagb_hist_subtract": bool(args.metagb_hist_subtract),
            "xgb_min_child_weight": 1.0,
            "xgb_reg_alpha": 0.0,
            "xgb_grow_policy": "depthwise",
            "xgb_colsample_bylevel": 1.0,
            "xgb_colsample_bynode": 1.0,
            "metagb_min_data_in_leaf": 1,
            "metagb_cond_pred_share": False,
            "metagb_leaf_only_use_bias": False,
        },
        "parameter_mapping": {
            "shared_lambda_l2": {
                "xgb_cpu": "reg_lambda",
                "current_metagb": "metagb_lambda_l2",
            },
            "shared_colsample_bytree": "condition-feature sampling once per tree",
            "metagb_joint_search": "10 pred_feature_fraction values x 3 score_agg values",
        },
        "discussion_notes": [
            "scale_pos_weight is not searched for binary classification; consider it if class imbalance becomes a paper concern.",
            "reg_alpha, grow_policy, colsample_bylevel, and colsample_bynode are fixed for XGBoost to match the agreed comparison contract.",
            "metagb_min_data_in_leaf=1 is a permissive fixed row-count constraint; it is not claimed to be semantically equivalent to classification min_child_weight.",
            "MetaGB internal defaults are recorded but not expanded into the random search budget in this run.",
            "score_agg values 0, 1, and 2 are searched jointly and evenly with pred_feature_fraction.",
        ],
        "warnings": warnings,
    }
