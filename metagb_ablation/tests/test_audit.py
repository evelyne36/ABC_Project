from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from metagb_ablation.io_utils import canonical_hash, canonical_json, write_json
from metagb_ablation.modeling import metagb_params_from_backbone
from metagb_ablation.protocols import build_variants
from metagb_ablation.run_ablation import (
    build_completeness_audit,
    ensure_execution_compatible,
)

EXPECTED_IDENTITY = {
    "run_id": "run",
    "config_hash": "config",
    "execution_hash": "execution",
    "variant_hash": "variants",
}


def _valid_artifacts():
    variants = build_variants()
    task = {
        "task_id": "t1",
        "unit_id": "u1",
        "domain": "synthetic",
        "task_type": "regression",
        "dataset": "",
        "scenario": "regime_slope",
        "dimension": 10,
        "data_seed": 101,
        "cv_repeat": 0,
        "outer_index": 0,
    }
    task_metadata = {
        "unit_id": "u1",
        "domain": "synthetic",
        "task_type": "regression",
        "dataset": "",
        "scenario": "regime_slope",
        "dimension": 10,
        "data_seed": 101,
        "cv_repeat": 0,
        "outer_fold": 1,
    }
    backbone = {
        "candidate_id": "xgb_0001",
        "backbone_id": "shared_backbone_0001",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_estimators": 200,
        "max_bin": 255,
        "reg_lambda": 0.25,
    }
    fold_rows = []
    prediction_rows = []
    for variant in variants:
        params = (
            dict(backbone)
            if variant.variant_id == "xgb"
            else metagb_params_from_backbone(backbone, variant)
        )
        fold_rows.append(
            {
                **EXPECTED_IDENTITY,
                **task_metadata,
                "backbone_hash": canonical_hash(backbone),
                "task_id": "t1",
                "variant_id": variant.variant_id,
                "outer_train_hash": "train",
                "outer_test_hash": "test",
                "inner_manifest_hash": "inner",
                "status": "ok",
                "params_json": canonical_json(params),
                "params_hash": canonical_hash(params),
                "n_test": 2,
                "task_type": "regression",
                "n_classes": 0,
            }
        )
        for sample_id in (0, 1):
            prediction_rows.append(
                {
                    **EXPECTED_IDENTITY,
                    **task_metadata,
                    "backbone_hash": canonical_hash(backbone),
                    "task_id": "t1",
                    "variant_id": variant.variant_id,
                    "sample_id": sample_id,
                    "y_true": float(sample_id),
                    "y_pred": float(sample_id),
                    "probabilities_json": "",
                }
            )
    backbone_rows = [
        {
            **EXPECTED_IDENTITY,
            **task_metadata,
            "backbone_hash": canonical_hash(backbone),
            "task_id": "t1",
            "params_hash": canonical_hash(backbone),
            "params_json": canonical_json(backbone),
        }
    ]
    split_manifest = {
        "entries": [
            {
                "task_id": "t1",
                "outer_train_hash": "train",
                "outer_test_hash": "test",
                "inner_manifest_hash": "inner",
            }
        ]
    }
    return (
        [task],
        variants,
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(backbone_rows),
        split_manifest,
    )


def _audit(artifacts):
    tasks, variants, folds, predictions, backbones, split_manifest = artifacts
    return build_completeness_audit(
        tasks=tasks,
        variants=variants,
        fold_results=folds,
        predictions=predictions,
        backbones=backbones,
        split_manifest=split_manifest,
        expected_identity=EXPECTED_IDENTITY,
    )


def test_complete_artifact_set_passes_strict_audit() -> None:
    audit = _audit(_valid_artifacts())
    assert audit["passed"]
    assert audit["issues"] == []


def test_audit_rejects_variant_substitution_and_prediction_duplicates() -> None:
    artifacts = list(_valid_artifacts())
    folds = artifacts[2].copy()
    folds.loc[folds["variant_id"] == "full_p10_sa2", "variant_id"] = "unknown"
    artifacts[2] = folds
    audit = _audit(tuple(artifacts))
    assert not audit["passed"]
    assert any("variant_set_mismatch" in issue for issue in audit["issues"])

    artifacts = list(_valid_artifacts())
    predictions = artifacts[3]
    artifacts[3] = pd.concat(
        [predictions, predictions.iloc[[0]]],
        ignore_index=True,
    )
    audit = _audit(tuple(artifacts))
    assert not audit["passed"]
    assert any("duplicate_prediction_keys" in issue for issue in audit["issues"])


def test_zero_task_audit_is_explicitly_not_complete() -> None:
    audit = build_completeness_audit(
        tasks=[],
        variants=build_variants(),
        fold_results=pd.DataFrame(),
        predictions=pd.DataFrame(),
        backbones=pd.DataFrame(),
        split_manifest={"entries": []},
        expected_identity=EXPECTED_IDENTITY,
    )
    assert not audit["passed"]
    assert "no_tasks_selected" in audit["issues"]


def test_audit_rejects_corrupted_execution_identity() -> None:
    artifacts = list(_valid_artifacts())
    folds = artifacts[2].copy()
    folds.loc[0, "execution_hash"] = "CORRUPT"
    artifacts[2] = folds
    audit = _audit(tuple(artifacts))
    assert not audit["passed"]
    assert any(
        issue.startswith("identity_mismatch:fold_results:execution_hash")
        for issue in audit["issues"]
    )


def test_audit_rejects_prediction_task_metadata_mismatch() -> None:
    artifacts = list(_valid_artifacts())
    predictions = artifacts[3].copy()
    predictions.loc[0, "unit_id"] = "wrong-unit"
    artifacts[3] = predictions
    audit = _audit(tuple(artifacts))
    assert not audit["passed"]
    assert any(
        issue.startswith("task_metadata_mismatch:predictions:t1:unit_id")
        for issue in audit["issues"]
    )


def test_execution_manifest_rejects_changed_backbone_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution_manifest.json"
    first = {
        "synthetic_backbones_hash": "a",
        "execution_hash": canonical_hash({"backbone": "a"}),
    }
    write_json(path, first)
    ensure_execution_compatible(path, first)
    changed = {
        "synthetic_backbones_hash": "b",
        "execution_hash": canonical_hash({"backbone": "b"}),
    }
    with pytest.raises(ValueError, match="Execution identity changed"):
        ensure_execution_compatible(path, changed)
