from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from crossgb_ablation.artifact_audit import build_completeness_audit
from crossgb_ablation.io_utils import (
    canonical_hash,
    file_sha256,
    write_csv,
    write_json,
    write_parquet,
)
from crossgb_ablation.protocols import VariantSpec
from crossgb_ablation.statistics import (
    main_ablation_tables,
    pred_sensitivity_tables,
    scoreagg_tables,
    unit_level_results,
)


MANAGED_OUTPUT_NAMES = (
    "fold_results.csv",
    "backbone_manifest.csv",
    "completeness_audit.json",
    "predictions.parquet",
    "unit_results.csv",
    "main_ablation.csv",
    "main_effects.csv",
    "pred_sensitivity.csv",
    "pred_effects.csv",
    "scoreagg_ablation.csv",
    "scoreagg_effects.csv",
    "ANALYSIS_EVIDENCE.md",
    "reanalysis_metadata.json",
)


@dataclass(frozen=True)
class ReanalysisResult:
    output_dir: Path
    audit: Dict[str, Any]
    metadata: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild CrossGB ablation audits and statistical tables from existing "
            "task shards without loading XGBoost or running training."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Completed or interrupted run directory containing tasks/*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination for rebuilt artifacts "
            "(default: sibling <result-dir>_reanalysis)."
        ),
    )
    parser.add_argument(
        "--write-predictions-parquet",
        action="store_true",
        help=(
            "Also write reconstructed predictions.parquet. This is optional and "
            "requires a pandas parquet engine such as PyArrow."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing reanalysis directory.",
    )
    return parser


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "" or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "" or pd.isna(value):
        return None
    return int(float(value))


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "" or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid optional boolean value: {value!r}")


def _required_bool(value: Any) -> bool:
    parsed = _optional_bool(value)
    if parsed is None:
        raise ValueError("Required boolean value is empty")
    return parsed


def _load_variants(path: Path) -> List[VariantSpec]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "variant_id",
        "family",
        "leaf_mode",
        "pred_fraction",
        "score_agg",
        "score_agg_label",
        "leaf_only_use_bias",
        "in_main_table",
        "in_pred_sensitivity",
        "in_scoreagg_ablation",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Variant manifest is missing columns: {sorted(missing)}")
    variants = [
        VariantSpec(
            variant_id=str(row["variant_id"]),
            family=str(row["family"]),
            leaf_mode=str(row["leaf_mode"]),
            pred_fraction=_optional_float(row["pred_fraction"]),
            score_agg=_optional_int(row["score_agg"]),
            score_agg_label=str(row["score_agg_label"]),
            leaf_only_use_bias=_optional_bool(row["leaf_only_use_bias"]),
            in_main_table=_required_bool(row["in_main_table"]),
            in_pred_sensitivity=_required_bool(row["in_pred_sensitivity"]),
            in_scoreagg_ablation=_required_bool(row["in_scoreagg_ablation"]),
        )
        for _, row in frame.iterrows()
    ]
    identifiers = [variant.variant_id for variant in variants]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Variant manifest contains duplicate variant identifiers")
    return variants


def _load_task_plan(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    if "task_id" not in frame:
        raise ValueError("Task plan is missing task_id")
    if frame["task_id"].duplicated().any():
        raise ValueError("Task plan contains duplicate task_id values")
    return frame.to_dict(orient="records")


def _collect_shards(
    shard_dir: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Missing task shard directory: {shard_dir}")
    fold_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    backbone_rows: List[Dict[str, Any]] = []
    shards: List[Dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for path in sorted(shard_dir.glob("*.json")):
        payload = _read_json(path)
        task_id = str(payload.get("task_id", ""))
        if not task_id:
            raise ValueError(f"Task shard has no task_id: {path}")
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate task shard for task_id={task_id}")
        seen_task_ids.add(task_id)
        shards.append(
            {
                key: payload.get(key)
                for key in (
                    "run_id",
                    "task_id",
                    "task_hash",
                    "config_hash",
                    "variant_hash",
                    "execution_hash",
                    "backbone_hash",
                )
            }
        )
        shard_folds = payload.get("fold_rows", [])
        shard_predictions = payload.get("predictions", [])
        if shard_folds:
            fold_frames.append(pd.DataFrame(shard_folds))
        if shard_predictions:
            prediction_frames.append(pd.DataFrame(shard_predictions))
        backbone = payload.get("backbone_row")
        if backbone:
            backbone_rows.append(backbone)
    return (
        (
            pd.concat(fold_frames, ignore_index=True)
            if fold_frames
            else pd.DataFrame()
        ),
        (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        ),
        pd.DataFrame(backbone_rows),
        shards,
    )


def _stored_run_id(
    result_dir: Path,
    shards: Sequence[Mapping[str, Any]],
) -> str:
    run_ids = {str(shard.get("run_id", "")) for shard in shards}
    run_ids.discard("")
    if len(run_ids) > 1:
        raise ValueError(f"Task shards contain multiple run IDs: {sorted(run_ids)}")
    if run_ids:
        return next(iter(run_ids))
    metadata_path = result_dir / "run_metadata.json"
    if metadata_path.is_file():
        run_id = str(_read_json(metadata_path).get("run_id", ""))
        if run_id:
            return run_id
    return result_dir.name


def _validate_manifest_identities(
    protocol: Mapping[str, Any],
    execution: Mapping[str, Any],
    variants: Sequence[VariantSpec],
    split_manifest: Mapping[str, Any],
) -> None:
    config_hash = str(protocol.get("config_hash", ""))
    variant_hash = str(protocol.get("variant_hash", ""))
    execution_hash = str(execution.get("execution_hash", ""))
    missing = [
        name
        for name, value in (
            ("protocol.config_hash", config_hash),
            ("protocol.variant_hash", variant_hash),
            ("execution_manifest.execution_hash", execution_hash),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Stored manifests are missing identities: {missing}")
    if str(execution.get("protocol_config_hash", "")) != config_hash:
        raise ValueError("Execution manifest does not match protocol config_hash")
    if str(execution.get("variant_hash", "")) != variant_hash:
        raise ValueError("Execution manifest does not match protocol variant_hash")
    expected_count = int(protocol.get("variant_count", -1))
    if expected_count != len(variants):
        raise ValueError(
            "Variant manifest row count does not match protocol: "
            f"{len(variants)} != {expected_count}"
        )
    actual_variant_hash = canonical_hash(
        [variant.to_dict() for variant in variants]
    )
    if actual_variant_hash != variant_hash:
        raise ValueError("Variant manifest content does not match variant_hash")
    entries = split_manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Split manifest entries must be a list")
    actual_split_hash = canonical_hash(entries)
    stored_split_hash = str(split_manifest.get("manifest_hash", ""))
    if stored_split_hash and actual_split_hash != stored_split_hash:
        raise ValueError("Split manifest content does not match manifest_hash")
    protocol_split_hash = str(protocol.get("split_manifest_hash", ""))
    if protocol_split_hash and actual_split_hash != protocol_split_hash:
        raise ValueError("Split manifest does not match protocol split_manifest_hash")


def _evidence_label(row: Mapping[str, Any]) -> str:
    paired_units = int(row.get("n_paired_units", 0))
    if paired_units < 2:
        return "单位不足"
    low = float(row.get("signed_ci_low", float("nan")))
    high = float(row.get("signed_ci_high", float("nan")))
    if math.isfinite(low) and low > 0.0:
        return "CI正向"
    if math.isfinite(high) and high < 0.0:
        return "CI负向"
    return "CI跨零"


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.6g}"


def _scenario_roles(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for task in tasks:
        scenario = str(task.get("scenario", ""))
        if not scenario:
            continue
        role = str(task.get("scenario_role", "")) or "unspecified"
        previous = roles.setdefault(scenario, role)
        if previous != role:
            raise ValueError(
                f"Scenario has inconsistent roles: {scenario}: {previous}, {role}"
            )
    return roles


def _effect_summary_rows(
    effects: pd.DataFrame,
    *,
    role_map: Mapping[str, str],
    group_columns: Sequence[str],
) -> List[List[str]]:
    if effects.empty:
        return []
    frame = effects.copy()
    frame["scenario_role"] = frame["scenario"].map(role_map).fillna("real/other")
    frame["evidence"] = [
        _evidence_label(row) for row in frame.to_dict(orient="records")
    ]
    rows: List[List[str]] = []
    columns = [*group_columns, "scenario_role"]
    for key, group in frame.groupby(columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        labels = group["evidence"].value_counts()
        relative = group["mean_relative_improvement"].to_numpy(dtype=float)
        relative = relative[np.isfinite(relative)]
        rows.append(
            [
                *(str(value) for value in key_values),
                str(int(labels.get("CI正向", 0))),
                str(int(labels.get("CI负向", 0))),
                str(int(labels.get("CI跨零", 0))),
                str(int(labels.get("单位不足", 0))),
                _format_number(np.median(relative) if relative.size else np.nan),
            ]
        )
    return rows


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_无可用行_"
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _write_evidence_report(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    execution: Mapping[str, Any],
    audit: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    main_effects: pd.DataFrame,
    pred_effects: pd.DataFrame,
    scoreagg_effects: pd.DataFrame,
) -> None:
    role_map = _scenario_roles(tasks)
    main_summary = _effect_summary_rows(
        main_effects,
        role_map=role_map,
        group_columns=["effect"],
    )
    pred_summary = _effect_summary_rows(
        pred_effects,
        role_map=role_map,
        group_columns=["effect", "pred_fraction"],
    )
    scoreagg_summary = _effect_summary_rows(
        scoreagg_effects,
        role_map=role_map,
        group_columns=["effect"],
    )

    detailed_rows: List[List[str]] = []
    if not main_effects.empty:
        ordered = main_effects.sort_values(
            ["scenario", "dimension", "effect"],
            kind="stable",
        )
        for row in ordered.to_dict(orient="records"):
            detailed_rows.append(
                [
                    str(row.get("scenario") or row.get("dataset") or ""),
                    str(row.get("dimension", "")),
                    role_map.get(str(row.get("scenario", "")), "real/other"),
                    str(row.get("effect", "")),
                    str(row.get("n_paired_units", "")),
                    _format_number(row.get("mean_signed_improvement")),
                    (
                        f"[{_format_number(row.get('signed_ci_low'))}, "
                        f"{_format_number(row.get('signed_ci_high'))}]"
                    ),
                    _format_number(row.get("mean_relative_improvement")),
                    _format_number(row.get("win_rate")),
                    _evidence_label(row),
                ]
            )

    protocol_body = protocol.get("protocol", {})
    confirmatory = (
        protocol_body.get("confirmatory")
        if isinstance(protocol_body, Mapping)
        else None
    )
    content = f"""# CrossGB Existing-Run Reanalysis Evidence

> 本报告从已保存的 `tasks/*.json` 离线重建；没有加载 XGBoost，也没有执行训练。
> “CI正向/负向”仅表示本次 paired bootstrap CI 是否完全位于 0 的一侧；
> paired unit 少于 2 时标记为“单位不足”。这些标签不是自动生成的普适论文结论。

## 1. 身份与完整性

- Protocol config hash: `{protocol.get("config_hash", "")}`
- Execution hash: `{execution.get("execution_hash", "")}`
- Variant hash: `{protocol.get("variant_hash", "")}`
- Confirmatory: `{confirmatory}`
- Audit passed: `{audit.get("passed")}`
- Tasks: `{audit.get("actual_tasks")}/{audit.get("expected_tasks")}`
- Fold rows: `{audit.get("actual_fold_rows")}/{audit.get("expected_fold_rows")}`
- Predictions: `{audit.get("actual_predictions")}/{audit.get("expected_predictions")}`

## 2. 主消融证据计数

正的 signed improvement 表示 target 优于 baseline。不要跨回归 RMSE 与分类
logloss 直接比较绝对 effect 大小；跨场景汇总使用相对改善的中位数。

{_markdown_table(
    [
        "effect",
        "scenario role",
        "CI正向 cells",
        "CI负向 cells",
        "CI跨零 cells",
        "单位不足 cells",
        "median relative improvement",
    ],
    main_summary,
)}

## 3. pred_fraction 剂量证据

{_markdown_table(
    [
        "effect",
        "pred_fraction",
        "scenario role",
        "CI正向 cells",
        "CI负向 cells",
        "CI跨零 cells",
        "单位不足 cells",
        "median relative improvement",
    ],
    pred_summary,
)}

## 4. score_agg 证据

{_markdown_table(
    [
        "effect",
        "scenario role",
        "CI正向 cells",
        "CI负向 cells",
        "CI跨零 cells",
        "单位不足 cells",
        "median relative improvement",
    ],
    scoreagg_summary,
)}

## 5. 主消融逐 cell 明细

{_markdown_table(
    [
        "scenario/dataset",
        "dimension",
        "role",
        "effect",
        "paired units",
        "signed improvement",
        "95% CI",
        "relative improvement",
        "win rate",
        "evidence",
    ],
    detailed_rows,
)}

## 6. 写结论时的边界

- `split_vs_bias` 对应 predictive-aware split 的独立贡献；
- `leaf_vs_split` 对应 predictive leaf 的增量贡献；
- `total_vs_bias` 对应完整机制相对 Bias 的总贡献；
- `full_vs_xgb` 是 fixed-backbone 诊断对比，不等同于各自调优后的最终性能；
- 必须单独检查 negative controls；若其改善与机制匹配场景同量级，应收缩机制归因；
- paired unit 是 seed/repeat，不能把 outer folds 当作 iid 重复。
"""
    path.write_text(content, encoding="utf-8")


def analyze_existing_run(
    result_dir: Path,
    *,
    output_dir: Optional[Path] = None,
    write_predictions_parquet: bool = False,
    overwrite: bool = False,
) -> ReanalysisResult:
    started = time.perf_counter()
    result_dir = result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else result_dir.parent / f"{result_dir.name}_reanalysis"
    )
    if destination == result_dir or destination.is_relative_to(result_dir):
        raise ValueError(
            "Reanalysis output directory must be outside the source result directory"
        )
    if write_predictions_parquet and not (
        importlib.util.find_spec("pyarrow")
        or importlib.util.find_spec("fastparquet")
    ):
        raise RuntimeError(
            "--write-predictions-parquet requires pyarrow or fastparquet; "
            "omit the flag to run analysis without a parquet dependency"
        )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Reanalysis directory is not empty: {destination}; use --overwrite"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in MANAGED_OUTPUT_NAMES:
            managed_path = destination / name
            if managed_path.is_file():
                managed_path.unlink()

    protocol = _read_json(result_dir / "protocol.json")
    execution = _read_json(result_dir / "execution_manifest.json")
    variants = _load_variants(result_dir / "variant_manifest.csv")
    tasks = _load_task_plan(result_dir / "task_plan.csv")
    split_manifest = _read_json(result_dir / "split_manifest.json")
    _validate_manifest_identities(
        protocol,
        execution,
        variants,
        split_manifest,
    )

    fold_df, prediction_df, backbone_df, shards = _collect_shards(
        result_dir / "tasks"
    )
    run_id = _stored_run_id(result_dir, shards)
    expected_identity = {
        "run_id": run_id,
        "config_hash": str(protocol["config_hash"]),
        "execution_hash": str(execution["execution_hash"]),
        "variant_hash": str(protocol["variant_hash"]),
    }
    audit = build_completeness_audit(
        tasks=tasks,
        variants=variants,
        fold_results=fold_df,
        predictions=prediction_df,
        backbones=backbone_df,
        split_manifest=split_manifest,
        expected_identity=expected_identity,
        shards=shards,
    )
    original_audit_path = result_dir / "completeness_audit.json"
    if original_audit_path.is_file():
        original_audit = _read_json(original_audit_path)
        audit["candidate_warnings"] = original_audit.get(
            "candidate_warnings",
            [],
        )
    write_csv(destination / "fold_results.csv", fold_df)
    write_csv(destination / "backbone_manifest.csv", backbone_df)
    write_json(destination / "completeness_audit.json", audit)
    if write_predictions_parquet:
        write_parquet(destination / "predictions.parquet", prediction_df)

    if audit["passed"]:
        unit_df = unit_level_results(fold_df, prediction_df)
    else:
        unit_df = unit_level_results(pd.DataFrame(), pd.DataFrame())
    seed = int(protocol.get("bootstrap_seed", 42))
    main_df, main_effect_df = main_ablation_tables(unit_df, seed=seed)
    pred_df, pred_effect_df = pred_sensitivity_tables(unit_df, seed=seed)
    scoreagg_df, scoreagg_effect_df = scoreagg_tables(unit_df, seed=seed)
    tables = {
        "unit_results.csv": unit_df,
        "main_ablation.csv": main_df,
        "main_effects.csv": main_effect_df,
        "pred_sensitivity.csv": pred_df,
        "pred_effects.csv": pred_effect_df,
        "scoreagg_ablation.csv": scoreagg_df,
        "scoreagg_effects.csv": scoreagg_effect_df,
    }
    for name, frame in tables.items():
        write_csv(destination / name, frame)

    if audit["passed"]:
        _write_evidence_report(
            destination / "ANALYSIS_EVIDENCE.md",
            protocol=protocol,
            execution=execution,
            audit=audit,
            tasks=tasks,
            main_effects=main_effect_df,
            pred_effects=pred_effect_df,
            scoreagg_effects=scoreagg_effect_df,
        )

    metadata = {
        "run_id": run_id,
        "source_result_dir": str(result_dir),
        "output_dir": str(destination),
        "analysis_only": True,
        "training_executed": False,
        "current_environment_identity_checked": False,
        "stored_config_hash": protocol["config_hash"],
        "stored_execution_hash": execution["execution_hash"],
        "stored_variant_hash": protocol["variant_hash"],
        "bootstrap_seed": seed,
        "analyzer_source_sha256": file_sha256(Path(__file__).resolve()),
        "statistics_source_sha256": file_sha256(
            Path(__file__).resolve().with_name("statistics.py")
        ),
        "completed": bool(audit["passed"]),
        "elapsed_sec": time.perf_counter() - started,
        "task_shards": len(shards),
        "fold_result_rows": len(fold_df),
        "prediction_rows_in_memory": len(prediction_df),
        "predictions_parquet_written": bool(write_predictions_parquet),
        "unit_result_rows": len(unit_df),
    }
    write_json(destination / "reanalysis_metadata.json", metadata)
    return ReanalysisResult(
        output_dir=destination,
        audit=audit,
        metadata=metadata,
    )


def main() -> int:
    args = build_parser().parse_args()
    result = analyze_existing_run(
        args.result_dir,
        output_dir=args.output_dir,
        write_predictions_parquet=args.write_predictions_parquet,
        overwrite=args.overwrite,
    )
    print(
        f"output_dir={result.output_dir} "
        f"completed={result.audit['passed']} "
        f"tasks={result.audit['actual_tasks']}/{result.audit['expected_tasks']} "
        f"fold_rows={result.audit['actual_fold_rows']} "
        f"predictions={result.audit['actual_predictions']}"
    )
    if not result.audit["passed"]:
        print(f"audit_issues={result.audit['issues']}")
        return 2
    print(f"evidence_report={result.output_dir / 'ANALYSIS_EVIDENCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
