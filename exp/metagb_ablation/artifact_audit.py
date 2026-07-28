from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from metagb_ablation.contracts import SHARED_BACKBONE_KEYS
from metagb_ablation.protocols import VariantSpec


def _parse_params(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_completeness_audit(
    *,
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[VariantSpec],
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    backbones: pd.DataFrame,
    split_manifest: Mapping[str, Any],
    expected_identity: Mapping[str, str],
    shards: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    issues = []
    expected_task_ids = {str(task["task_id"]) for task in tasks}
    expected_variant_ids = {variant.variant_id for variant in variants}
    expected_fold_rows = len(expected_task_ids) * len(expected_variant_ids)
    if not expected_task_ids:
        issues.append("no_tasks_selected")

    if expected_task_ids:
        for artifact_name, frame in (
            ("fold_results", fold_results),
            ("predictions", predictions),
            ("backbones", backbones),
        ):
            if frame.empty:
                continue
            for field, expected_value in expected_identity.items():
                if field not in frame:
                    issues.append(f"missing_identity:{artifact_name}:{field}")
                    continue
                actual_values = set(frame[field].astype(str))
                if actual_values != {str(expected_value)}:
                    issues.append(
                        f"identity_mismatch:{artifact_name}:{field}:"
                        f"expected={expected_value}:actual={sorted(actual_values)}"
                    )
        if shards:
            shard_task_ids = {str(shard.get("task_id", "")) for shard in shards}
            if shard_task_ids != expected_task_ids:
                issues.append("shard_task_set_mismatch")
            for shard in shards:
                task_id = str(shard.get("task_id", ""))
                for field, expected_value in expected_identity.items():
                    if str(shard.get(field, "")) != str(expected_value):
                        issues.append(
                            f"identity_mismatch:shard:{task_id}:{field}"
                        )
        task_lookup = {str(task["task_id"]): task for task in tasks}
        metadata_fields = (
            "unit_id",
            "domain",
            "task_type",
            "dataset",
            "scenario",
            "dimension",
            "data_seed",
            "cv_repeat",
            "outer_fold",
        )
        for artifact_name, frame in (
            ("fold_results", fold_results),
            ("predictions", predictions),
            ("backbones", backbones),
        ):
            if frame.empty or "task_id" not in frame:
                continue
            for task_id, task_frame in frame.groupby("task_id"):
                task = task_lookup.get(str(task_id))
                if task is None:
                    continue
                for field in metadata_fields:
                    task_expected_value: Any = (
                        int(task["outer_index"]) + 1
                        if field == "outer_fold"
                        else task.get(field, "")
                    )
                    if field not in task_frame:
                        issues.append(
                            f"missing_task_metadata:{artifact_name}:{task_id}:{field}"
                        )
                    elif set(task_frame[field].astype(str)) != {
                        str(task_expected_value)
                    }:
                        issues.append(
                            f"task_metadata_mismatch:"
                            f"{artifact_name}:{task_id}:{field}"
                        )

    actual_task_ids = (
        set(fold_results["task_id"].astype(str))
        if "task_id" in fold_results
        else set()
    )
    if actual_task_ids != expected_task_ids:
        issues.append(
            "task_id_set_mismatch:"
            f"missing={sorted(expected_task_ids - actual_task_ids)}:"
            f"extra={sorted(actual_task_ids - expected_task_ids)}"
        )
    if not fold_results.empty:
        duplicate_fold_keys = int(
            fold_results.duplicated(
                subset=["task_id", "variant_id"],
                keep=False,
            ).sum()
        )
        if duplicate_fold_keys:
            issues.append(f"duplicate_task_variant_rows={duplicate_fold_keys}")
        for task_id, frame in fold_results.groupby("task_id"):
            actual_variants = set(frame["variant_id"].astype(str))
            if actual_variants != expected_variant_ids:
                issues.append(
                    f"variant_set_mismatch:{task_id}:"
                    f"missing={sorted(expected_variant_ids - actual_variants)}:"
                    f"extra={sorted(actual_variants - expected_variant_ids)}"
                )
            if frame["outer_train_hash"].nunique(dropna=False) != 1:
                issues.append(f"outer_train_hash_mismatch:{task_id}")
            if frame["outer_test_hash"].nunique(dropna=False) != 1:
                issues.append(f"outer_test_hash_mismatch:{task_id}")
            if frame["inner_manifest_hash"].nunique(dropna=False) != 1:
                issues.append(f"inner_manifest_hash_mismatch:{task_id}")
            if (frame["status"] != "ok").any():
                failed = frame.loc[
                    frame["status"] != "ok",
                    "variant_id",
                ].tolist()
                issues.append(f"failed_variants:{task_id}:{failed}")

            xgb_rows = frame[frame["variant_id"] == "xgb"]
            if len(xgb_rows) != 1:
                issues.append(f"xgb_row_count:{task_id}:{len(xgb_rows)}")
                continue
            xgb_params = _parse_params(xgb_rows.iloc[0]["params_json"])
            for _, row in frame[frame["variant_id"] != "xgb"].iterrows():
                params = _parse_params(row["params_json"])
                for key in SHARED_BACKBONE_KEYS:
                    if params.get(key) != xgb_params.get(key):
                        issues.append(
                            f"public_param_mismatch:"
                            f"{task_id}:{row['variant_id']}:{key}"
                        )
                xgb_lambda = float(
                    xgb_params.get(
                        "reg_lambda",
                        xgb_params.get("lambda_l2", 1.0),
                    )
                )
                metagb_lambda = float(
                    params.get("metagb_lambda_l2", float("nan"))
                )
                if not np.isfinite(metagb_lambda) or xgb_lambda != metagb_lambda:
                    issues.append(
                        f"lambda_mismatch:{task_id}:{row['variant_id']}"
                    )

    manifest_entries = {
        str(entry["task_id"]): entry
        for entry in split_manifest.get("entries", [])
    }
    if set(manifest_entries) != expected_task_ids:
        issues.append("split_manifest_task_set_mismatch")
    if not fold_results.empty:
        for task_id, frame in fold_results.groupby("task_id"):
            manifest = manifest_entries.get(str(task_id))
            if manifest is None:
                continue
            first = frame.iloc[0]
            for field in (
                "outer_train_hash",
                "outer_test_hash",
                "inner_manifest_hash",
            ):
                if str(first[field]) != str(manifest[field]):
                    issues.append(
                        f"split_manifest_value_mismatch:{task_id}:{field}"
                    )

    if backbones.empty:
        if expected_task_ids:
            issues.append("missing_backbone_manifest")
    else:
        duplicate_backbones = int(
            backbones.duplicated(subset=["task_id"], keep=False).sum()
        )
        if duplicate_backbones:
            issues.append(f"duplicate_backbone_rows={duplicate_backbones}")
        if set(backbones["task_id"].astype(str)) != expected_task_ids:
            issues.append("backbone_task_set_mismatch")
        if not fold_results.empty:
            xgb_hashes = (
                fold_results[fold_results["variant_id"] == "xgb"]
                .set_index("task_id")["params_hash"]
                .to_dict()
            )
            for _, row in backbones.iterrows():
                task_id = str(row["task_id"])
                expected_backbone_hash = str(row.get("params_hash", ""))
                if str(row.get("backbone_hash", "")) != expected_backbone_hash:
                    issues.append(f"backbone_self_hash_mismatch:{task_id}")
                if expected_backbone_hash != str(xgb_hashes.get(task_id, "")):
                    issues.append(
                        f"backbone_params_hash_mismatch:{row['task_id']}"
                    )
                task_folds = fold_results[fold_results["task_id"] == task_id]
                if (
                    "backbone_hash" not in task_folds
                    or set(task_folds["backbone_hash"].astype(str))
                    != {expected_backbone_hash}
                ):
                    issues.append(f"fold_backbone_hash_mismatch:{task_id}")
                task_predictions = (
                    predictions[predictions["task_id"].astype(str) == task_id]
                    if "task_id" in predictions
                    else pd.DataFrame()
                )
                if (
                    task_predictions.empty
                    or "backbone_hash" not in task_predictions
                    or set(task_predictions["backbone_hash"].astype(str))
                    != {expected_backbone_hash}
                ):
                    issues.append(f"prediction_backbone_hash_mismatch:{task_id}")
                matching_shards = [
                    shard
                    for shard in shards
                    if str(shard.get("task_id", "")) == task_id
                ]
                if shards and (
                    len(matching_shards) != 1
                    or str(matching_shards[0].get("backbone_hash", ""))
                    != expected_backbone_hash
                ):
                    issues.append(f"shard_backbone_hash_mismatch:{task_id}")

    expected_prediction_count = (
        int(fold_results["n_test"].sum())
        if not fold_results.empty and "n_test" in fold_results
        else 0
    )
    if predictions.empty:
        if expected_prediction_count:
            issues.append("missing_predictions")
    else:
        duplicate_prediction_keys = int(
            predictions.duplicated(
                subset=["task_id", "variant_id", "sample_id"],
                keep=False,
            ).sum()
        )
        if duplicate_prediction_keys:
            issues.append(
                f"duplicate_prediction_keys={duplicate_prediction_keys}"
            )
        prediction_task_ids = set(predictions["task_id"].astype(str))
        if prediction_task_ids != expected_task_ids:
            issues.append("prediction_task_set_mismatch")
        for task_id, task_predictions in predictions.groupby("task_id"):
            task_fold = fold_results[fold_results["task_id"] == task_id]
            if task_fold.empty:
                continue
            expected_n_test = int(task_fold["n_test"].iloc[0])
            reference_samples: Optional[set[int]] = None
            for variant_id in expected_variant_ids:
                frame = task_predictions[
                    task_predictions["variant_id"] == variant_id
                ]
                if len(frame) != expected_n_test:
                    issues.append(
                        f"prediction_count_mismatch:{task_id}:{variant_id}:"
                        f"{len(frame)}!={expected_n_test}"
                    )
                sample_ids = set(frame["sample_id"].astype(int))
                if reference_samples is None:
                    reference_samples = sample_ids
                elif sample_ids != reference_samples:
                    issues.append(
                        f"prediction_sample_set_mismatch:{task_id}:{variant_id}"
                    )
            y_consistency = task_predictions.groupby("sample_id")[
                "y_true"
            ].nunique()
            if (y_consistency > 1).any():
                issues.append(f"y_true_mismatch:{task_id}")
            if not np.isfinite(
                task_predictions[["y_true", "y_pred"]].to_numpy(dtype=float)
            ).all():
                issues.append(f"nonfinite_predictions:{task_id}")
            task_type = str(task_fold["task_type"].iloc[0])
            if task_type != "regression":
                expected_classes = int(task_fold["n_classes"].iloc[0])
                for value in task_predictions["probabilities_json"]:
                    try:
                        proba = np.asarray(json.loads(value), dtype=float)
                    except (TypeError, json.JSONDecodeError):
                        issues.append(f"invalid_probability_json:{task_id}")
                        break
                    if (
                        proba.shape != (expected_classes,)
                        or not np.isfinite(proba).all()
                        or not np.isclose(
                            proba.sum(),
                            1.0,
                            rtol=1e-5,
                            atol=1e-5,
                        )
                    ):
                        issues.append(f"invalid_probability_vector:{task_id}")
                        break

    return {
        "passed": not issues,
        "issues": issues,
        "expected_tasks": len(expected_task_ids),
        "actual_tasks": len(actual_task_ids),
        "variant_count": len(expected_variant_ids),
        "expected_fold_rows": expected_fold_rows,
        "actual_fold_rows": int(len(fold_results)),
        "successful_fold_rows": (
            int((fold_results["status"] == "ok").sum())
            if "status" in fold_results
            else 0
        ),
        "expected_predictions": expected_prediction_count,
        "actual_predictions": int(len(predictions)),
        "backbone_rows": int(len(backbones)),
    }
