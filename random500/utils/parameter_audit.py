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

EXPLICIT_METAGB_PARAMS = [
    "enable_metagb",
    "metagb_pred_colsample_bytree",
    "metagb_lambda_l2",
    "metagb_score_agg",
    "metagb_hist_subtract",
]

DEFAULTED_XGB_PARAMS = [
    "gamma/min_split_loss",
    "reg_alpha",
    "max_delta_step",
    "colsample_bylevel",
    "colsample_bynode",
    "grow_policy",
    "max_leaves",
    "sampling_method",
    "scale_pos_weight",
    "base_score",
    "max_cat_to_onehot",
    "max_cat_threshold",
]

DEFAULTED_METAGB_PARAMS = [
    "metagb_cond_pred_share",
    "metagb_leaf_only_use_bias",
    "metagb_split_sample",
    "metagb_min_gain_to_split",
    "metagb_min_data_in_leaf",
]


def build_parameter_audit(args: Any, search_space: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    searched_shared = sorted(search_space.get("shared", {}).keys())
    searched_xgb = sorted(search_space.get("xgb_cpu", {}).keys())
    searched_metagb = sorted(search_space.get("current_metagb", {}).keys())
    return {
        "explicitly_set": {
            "common": EXPLICIT_COMMON_PARAMS,
            "xgb_cpu": EXPLICIT_XGB_PARAMS,
            "current_metagb": EXPLICIT_METAGB_PARAMS,
            "classification_extra": ["eval_metric", "num_class for multiclass"],
        },
        "searched_parameters": {
            "shared": searched_shared,
            "xgb_cpu": searched_xgb,
            "current_metagb": searched_metagb + ["pred_feature_fraction"],
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
        },
        "discussion_notes": [
            "scale_pos_weight is not searched for binary classification; consider it if class imbalance becomes a paper concern.",
            "reg_alpha/gamma/colsample_bylevel/colsample_bynode remain default, so the current experiment isolates the main Exp1-style parameter family.",
            "MetaGB internal defaults are recorded but not expanded into the random search budget in this run.",
        ],
        "warnings": warnings,
    }
