from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from metagb_ablation.analyze_existing import analyze_existing_run
from metagb_ablation.io_utils import canonical_hash, write_csv, write_json
from metagb_ablation.tests.test_audit import (
    EXPECTED_IDENTITY,
    _valid_artifacts,
)


def _write_existing_run(path: Path, *, include_shard: bool = True) -> None:
    tasks, variants, folds, predictions, backbones, split_manifest = (
        _valid_artifacts()
    )
    variant_lookup = {variant.variant_id: variant for variant in variants}
    variant_hash = canonical_hash([variant.to_dict() for variant in variants])
    identity = {**EXPECTED_IDENTITY, "variant_hash": variant_hash}
    folds = folds.copy()
    predictions = predictions.copy()
    backbones = backbones.copy()
    folds["variant_hash"] = variant_hash
    predictions["variant_hash"] = variant_hash
    backbones["variant_hash"] = variant_hash
    folds["primary_metric"] = "rmse"
    folds["test_primary_metric"] = 0.0
    folds["fit_sec"] = 1.0
    folds["predict_sec"] = 0.1
    for row_index, row in folds.iterrows():
        variant = variant_lookup[str(row["variant_id"])]
        folds.loc[row_index, "family"] = variant.family
        folds.loc[row_index, "leaf_mode"] = variant.leaf_mode
        folds.loc[row_index, "pred_fraction"] = variant.pred_fraction
        folds.loc[row_index, "score_agg"] = variant.score_agg
        folds.loc[row_index, "score_agg_label"] = variant.score_agg_label
    path.mkdir()
    (path / "tasks").mkdir()
    write_json(
        path / "protocol.json",
        {
            "protocol": {
                "name": "smoke",
                "confirmatory": False,
            },
            "config_hash": EXPECTED_IDENTITY["config_hash"],
            "variant_hash": variant_hash,
            "variant_count": len(variants),
            "bootstrap_seed": 42,
        },
    )
    write_json(
        path / "execution_manifest.json",
        {
            "protocol_config_hash": EXPECTED_IDENTITY["config_hash"],
            "variant_hash": variant_hash,
            "execution_hash": EXPECTED_IDENTITY["execution_hash"],
            "xgboost_python_package": "/path/that/does/not/exist",
        },
    )
    write_csv(
        path / "variant_manifest.csv",
        [variant.to_dict() for variant in variants],
    )
    write_csv(path / "task_plan.csv", tasks)
    write_json(path / "split_manifest.json", split_manifest)
    if include_shard:
        write_json(
            path / "tasks" / "t1.json",
            {
                **identity,
                "task_id": "t1",
                "task_hash": "stored-task-hash",
                "backbone_hash": backbones.iloc[0]["backbone_hash"],
                "fold_rows": folds.to_dict(orient="records"),
                "predictions": predictions.to_dict(orient="records"),
                "backbone_row": backbones.iloc[0].to_dict(),
            },
        )


def test_reanalysis_rebuilds_complete_tables_without_training_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_existing_run(source)

    sys.modules.pop("xgboost", None)
    result = analyze_existing_run(source, output_dir=output)

    assert result.audit["passed"]
    assert result.metadata["training_executed"] is False
    assert result.metadata["current_environment_identity_checked"] is False
    assert "xgboost" not in sys.modules
    assert (output / "ANALYSIS_EVIDENCE.md").is_file()
    assert not (output / "predictions.parquet").exists()
    assert len(pd.read_csv(output / "unit_results.csv")) == 16
    assert len(pd.read_csv(output / "main_effects.csv")) == 4


def test_reanalysis_hard_gates_statistics_when_shard_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_existing_run(source, include_shard=False)

    result = analyze_existing_run(source, output_dir=output)

    assert not result.audit["passed"]
    assert "shard_task_set_mismatch" not in result.audit["issues"]
    assert any(
        issue.startswith("task_id_set_mismatch")
        for issue in result.audit["issues"]
    )
    assert pd.read_csv(output / "unit_results.csv").empty
    assert not (output / "ANALYSIS_EVIDENCE.md").exists()


def test_reanalysis_refuses_to_write_inside_source_or_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_existing_run(source)

    with pytest.raises(ValueError, match="outside"):
        analyze_existing_run(source, output_dir=source / "reanalysis")

    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        analyze_existing_run(source, output_dir=output)


def test_failed_overwrite_removes_stale_success_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_existing_run(source)
    analyze_existing_run(source, output_dir=output)
    assert (output / "ANALYSIS_EVIDENCE.md").is_file()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    (source / "tasks" / "t1.json").unlink()
    result = analyze_existing_run(source, output_dir=output, overwrite=True)

    assert not result.audit["passed"]
    assert not (output / "ANALYSIS_EVIDENCE.md").exists()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"
