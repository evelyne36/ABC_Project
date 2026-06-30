#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build MetaGB-vs-XGB analysis tables from task logs and JSON artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "mgb_datacollection_by0629"
DERIVED = OUT_ROOT / "derived"
TABLES = OUT_ROOT / "tables"


LOG_CONFIG_RE = re.compile(
    r"config task_type=(?P<task_type>\S+) dataset=(?P<dataset>\S+) "
    r"model=(?P<model>\S+) outer_fold=(?P<outer_fold>\d+)/(?P<outer_folds>\d+) "
    r"inner_folds=(?P<inner_folds>\d+) candidates=(?P<candidates>\d+) "
    r"objective=(?P<objective>\S+) primary_metric=(?P<primary_metric>\S+) "
    r"direction=(?P<direction>\S+) rows=(?P<rows>\d+)"
    r"(?: original_rows=(?P<original_rows>\d+))? .*?features=(?P<features>\d+) "
    r"classes=(?P<classes>\d+)"
)
LOG_SELECTED_RE = re.compile(
    r"selected candidate_index=(?P<selected_candidate_index>\d+) "
    r"candidate_id=(?P<selected_candidate_id>\S+) "
    r"val_(?P<val_metric>\S+)=(?P<val_primary_metric>[-+eE0-9.]+) "
    r"val_std=(?P<val_primary_metric_std>[-+eE0-9.]+)"
)
LOG_FINISH_RE = re.compile(
    r"finish best=(?P<best>\d+) test_(?P<test_metric>\S+)="
    r"(?P<test_primary_metric>[-+eE0-9.]+) "
    r"final_train_sec=(?P<final_train_time_sec>[-+eE0-9.]+) "
    r"final_predict_sec=(?P<final_predict_time_sec>[-+eE0-9.]+) "
    r"elapsed_sec=(?P<task_elapsed_sec>[-+eE0-9.]+) status=(?P<status>\S+)"
)


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def clean_dataset_name(task_type: str | None, dataset: str | None) -> str | None:
    if dataset is None:
        return None
    value = dataset
    if task_type and value.startswith(f"{task_type}__"):
        value = value[len(task_type) + 2 :]
    return value


def infer_context(path: Path) -> dict[str, str]:
    text = path.as_posix()
    context = {
        "family": "unknown",
        "protocol": "unknown",
        "encoding": "unknown",
        "collection": "unknown",
    }
    if "/datacollection0624/" in text:
        context["family"] = "tabarena_main_0624"
        context["collection"] = "datacollection0624"
        context["protocol"] = "random600" if "random600" in text else "random300"
        if "_onehot_" in text or "/binary_onehot_" in text or "/multiclass_onehot_" in text:
            context["encoding"] = "onehot"
        elif "_drop_" in text or "/binary_drop_" in text or "/multiclass_drop_" in text or "/regression_drop_" in text:
            context["encoding"] = "drop"
    elif "/mgb_datacollection_by0629/synthetic_main" in text:
        context["family"] = "synthetic_0629"
        context["collection"] = "mgb_datacollection_by0629"
        context["protocol"] = "synthetic_random300_single_seed"
        context["encoding"] = "synthetic"
    elif "/mgb_datacollection_by0629/tabarena30_" in text:
        context["family"] = "tabarena30_0629"
        context["collection"] = "mgb_datacollection_by0629"
        context["protocol"] = "xgb_best_mgb30"
        context["encoding"] = "onehot" if "tabarena30_onehot" in text else "drop"
    elif "/mgb_datacollection_by0629/tabular_otherdata_" in text:
        context["family"] = "tabular_otherdata_0629"
        context["collection"] = "mgb_datacollection_by0629"
        context["protocol"] = "random300"
        context["encoding"] = "onehot" if "tabular_otherdata_onehot" in text else "drop"
    return context


def parse_log(path: Path) -> dict[str, Any] | None:
    lines = path.read_text(errors="replace").splitlines()
    config: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    finish: dict[str, Any] = {}
    for line in lines:
        if not config:
            m = LOG_CONFIG_RE.search(line)
            if m:
                config = m.groupdict()
                continue
        m = LOG_SELECTED_RE.search(line)
        if m:
            selected = m.groupdict()
            continue
        m = LOG_FINISH_RE.search(line)
        if m:
            finish = m.groupdict()
    if not config:
        return None

    context = infer_context(path)
    task_type = config.get("task_type")
    dataset = clean_dataset_name(task_type, config.get("dataset"))
    row: dict[str, Any] = {
        **context,
        "source_type": "log",
        "source_path": str(path.relative_to(ROOT)),
        "mtime": path.stat().st_mtime,
        "task_type": task_type,
        "dataset": dataset,
        "model": config.get("model"),
        "outer_fold": int(config["outer_fold"]),
        "outer_folds": int(config["outer_folds"]),
        "inner_folds": int(config["inner_folds"]),
        "candidate_count": int(config["candidates"]),
        "objective": config.get("objective"),
        "primary_metric": config.get("primary_metric"),
        "primary_metric_direction": config.get("direction"),
        "processed_rows": int(config["rows"]),
        "raw_rows": int(config["original_rows"]) if config.get("original_rows") else None,
        "processed_features": int(config["features"]),
        "classes": int(config["classes"]),
        "status": finish.get("status", "incomplete"),
    }
    if selected:
        row.update(
            {
                "selected_candidate_index": int(selected["selected_candidate_index"]),
                "selected_candidate_id": selected["selected_candidate_id"],
                "val_primary_metric": finite_or_none(selected["val_primary_metric"]),
                "val_primary_metric_std": finite_or_none(selected["val_primary_metric_std"]),
            }
        )
    if finish:
        metric = finish["test_metric"]
        value = finite_or_none(finish["test_primary_metric"])
        row.update(
            {
                "test_primary_metric": value,
                f"test_{metric}": value,
                "final_train_time_sec": finite_or_none(finish["final_train_time_sec"]),
                "final_predict_time_sec": finite_or_none(finish["final_predict_time_sec"]),
                "task_elapsed_sec": finite_or_none(finish["task_elapsed_sec"]),
            }
        )
    return row


def parse_synthetic_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    fold = data.get("fold_row")
    if not isinstance(fold, dict):
        return None
    context = infer_context(path)
    candidate_rows = data.get("candidate_rows") or []
    selected_id = fold.get("selected_candidate_id")
    selected = next((r for r in candidate_rows if r.get("candidate_id") == selected_id), {})
    inner_fit = sum(finite_or_none(r.get("inner_fit_sec")) or 0.0 for r in candidate_rows)
    inner_pred = sum(finite_or_none(r.get("inner_predict_sec")) or 0.0 for r in candidate_rows)
    primary_metric = fold.get("primary_metric")
    direction = "maximize" if primary_metric == "auc" else "minimize"
    row: dict[str, Any] = {
        **context,
        "source_type": "synthetic_json",
        "source_path": str(path.relative_to(ROOT)),
        "mtime": path.stat().st_mtime,
        "task_type": fold.get("task_type"),
        "dataset": fold.get("scenario"),
        "synthetic_family": fold.get("family"),
        "synthetic_case": fold.get("case") or fold.get("base_scenario"),
        "synthetic_dimension": fold.get("dimension"),
        "model": fold.get("model"),
        "outer_fold": int(fold["outer_fold"]),
        "outer_folds": 5,
        "inner_folds": 4,
        "candidate_count": len(candidate_rows),
        "objective": None,
        "primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "processed_rows": 3000,
        "raw_rows": 3000,
        "processed_features": fold.get("dimension"),
        "classes": 0 if fold.get("task_type") == "regression" else None,
        "selected_candidate_id": selected_id,
        "selected_backbone_id": fold.get("selected_backbone_id"),
        "selected_candidate_index": selected.get("candidate_index"),
        "val_primary_metric": finite_or_none(selected.get("val_primary_metric")),
        "test_primary_metric": finite_or_none(fold.get("test_primary_metric")),
        "test_rmse": finite_or_none(fold.get("test_rmse")),
        "test_auc": finite_or_none(fold.get("test_auc")),
        "test_logloss": finite_or_none(fold.get("test_logloss")),
        "final_train_time_sec": finite_or_none(fold.get("fit_sec")),
        "final_predict_time_sec": finite_or_none(fold.get("predict_sec")),
        "inner_train_time_sec": inner_fit,
        "inner_predict_time_sec": inner_pred,
        "task_elapsed_sec": inner_fit + inner_pred + (finite_or_none(fold.get("fit_sec")) or 0.0) + (finite_or_none(fold.get("predict_sec")) or 0.0),
        "status": "ok",
    }
    return row


def parse_summary_csv(path: Path) -> list[dict[str, Any]]:
    context = infer_context(path)
    rows: list[dict[str, Any]] = []
    try:
        df = pd.read_csv(path)
    except Exception:
        return rows
    for _, rec in df.iterrows():
        task_type = rec.get("task_type")
        dataset = clean_dataset_name(task_type, rec.get("dataset"))
        row = {
            **context,
            "source_type": "summary_csv",
            "source_path": str(path.relative_to(ROOT)),
            "mtime": path.stat().st_mtime,
            "task_type": task_type,
            "dataset": dataset,
            "model": rec.get("model"),
            "outer_fold": int(rec.get("outer_fold")),
            "outer_folds": int(rec.get("outer_folds", 5)),
            "inner_folds": int(rec.get("inner_folds", 4)),
            "candidate_count": int(rec.get("candidate_count", 0)),
            "objective": rec.get("objective"),
            "primary_metric": rec.get("primary_metric"),
            "primary_metric_direction": rec.get("primary_metric_direction"),
            "selected_candidate_index": rec.get("selected_candidate_index"),
            "selected_candidate_id": rec.get("selected_candidate_id"),
            "selected_backbone_id": rec.get("selected_backbone_id"),
            "val_primary_metric": finite_or_none(rec.get("val_primary_metric_mean")),
            "val_primary_metric_std": finite_or_none(rec.get("val_primary_metric_std")),
            "test_primary_metric": finite_or_none(rec.get("test_primary_metric")),
            "test_rmse": finite_or_none(rec.get("test_rmse")),
            "test_auc": finite_or_none(rec.get("test_auc")),
            "test_logloss": finite_or_none(rec.get("test_logloss")),
            "test_accuracy": finite_or_none(rec.get("test_accuracy")),
            "final_train_time_sec": finite_or_none(rec.get("final_train_time_sec")),
            "final_predict_time_sec": finite_or_none(rec.get("final_predict_time_sec")),
            "inner_train_time_sec": finite_or_none(rec.get("inner_train_time_sec")),
            "inner_predict_time_sec": finite_or_none(rec.get("inner_predict_time_sec")),
            "task_elapsed_sec": finite_or_none(rec.get("task_elapsed_sec")),
            "status": rec.get("status"),
        }
        rows.append(row)
    return rows


def collect_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for log_path in list((ROOT / "datacollection0624").rglob("*.log")) + list(OUT_ROOT.rglob("*.log")):
        parsed = parse_log(log_path)
        if parsed:
            rows.append(parsed)
    for json_path in (OUT_ROOT / "synthetic_main102050_random300_seed42" / "tasks").glob("*.json"):
        parsed = parse_synthetic_json(json_path)
        if parsed:
            rows.append(parsed)
    for summary_path in (ROOT / "datacollection0624").rglob("summary.csv"):
        rows.extend(parse_summary_csv(summary_path))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["status_rank"] = (df["status"] == "ok").astype(int)
    key_cols = ["family", "protocol", "encoding", "task_type", "dataset", "model", "outer_fold"]
    df = (
        df.sort_values(["status_rank", "mtime"], ascending=[False, False])
        .drop_duplicates(key_cols, keep="first")
        .drop(columns=["status_rank"])
        .reset_index(drop=True)
    )
    return df


def compare_values(mgb: float, xgb: float, direction: str) -> tuple[str, float]:
    if xgb is None or mgb is None or not math.isfinite(xgb) or not math.isfinite(mgb):
        return "missing", math.nan
    if direction == "maximize":
        improvement = mgb - xgb
        tie = abs(improvement) <= 1e-4
        if tie:
            return "tie", improvement
        return ("mgb_win" if improvement > 0 else "xgb_win", improvement)
    improvement = (xgb - mgb) / abs(xgb) if xgb else xgb - mgb
    tie = abs(improvement) <= 0.001
    if tie:
        return "tie", improvement
    return ("mgb_win" if improvement > 0 else "xgb_win", improvement)


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[(df["status"] == "ok") & df["test_primary_metric"].notna()].copy()
    idx_cols = ["family", "protocol", "encoding", "task_type", "dataset", "outer_fold"]
    pairs = []
    for key, group in ok.groupby(idx_cols, dropna=False):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        if "xgb_cpu" not in by_model or "current_metagb" not in by_model:
            continue
        xgb = by_model["xgb_cpu"]
        mgb = by_model["current_metagb"]
        direction = mgb.get("primary_metric_direction") or xgb.get("primary_metric_direction")
        test_winner, test_improvement = compare_values(
            float(mgb["test_primary_metric"]), float(xgb["test_primary_metric"]), direction
        )
        cv_winner, cv_improvement = compare_values(
            finite_or_none(mgb.get("val_primary_metric")),
            finite_or_none(xgb.get("val_primary_metric")),
            direction,
        )
        row = dict(zip(idx_cols, key))
        row.update(
            {
                "primary_metric": mgb.get("primary_metric") or xgb.get("primary_metric"),
                "direction": direction,
                "xgb_test": xgb.get("test_primary_metric"),
                "mgb_test": mgb.get("test_primary_metric"),
                "test_winner": test_winner,
                "test_improvement": test_improvement,
                "xgb_val": xgb.get("val_primary_metric"),
                "mgb_val": mgb.get("val_primary_metric"),
                "cv_winner": cv_winner,
                "cv_improvement": cv_improvement,
                "xgb_elapsed_sec": xgb.get("task_elapsed_sec"),
                "mgb_elapsed_sec": mgb.get("task_elapsed_sec"),
                "runtime_ratio": (
                    float(mgb.get("task_elapsed_sec")) / float(xgb.get("task_elapsed_sec"))
                    if finite_or_none(mgb.get("task_elapsed_sec")) and finite_or_none(xgb.get("task_elapsed_sec"))
                    else math.nan
                ),
                "rows": mgb.get("processed_rows") if pd.notna(mgb.get("processed_rows")) else xgb.get("processed_rows"),
                "features": mgb.get("processed_features") if pd.notna(mgb.get("processed_features")) else xgb.get("processed_features"),
                "synthetic_family": mgb.get("synthetic_family"),
                "synthetic_case": mgb.get("synthetic_case"),
                "synthetic_dimension": mgb.get("synthetic_dimension"),
                "xgb_source": xgb.get("source_path"),
                "mgb_source": mgb.get("source_path"),
            }
        )
        pairs.append(row)
    return pd.DataFrame(pairs)


def summarize_datasets(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    group_cols = ["family", "protocol", "encoding", "task_type", "dataset"]
    rows = []
    for key, group in pairs.groupby(group_cols, dropna=False):
        direction = group["direction"].dropna().iloc[0]
        xgb_mean = group["xgb_test"].mean()
        mgb_mean = group["mgb_test"].mean()
        winner, improvement = compare_values(float(mgb_mean), float(xgb_mean), direction)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "primary_metric": group["primary_metric"].dropna().iloc[0],
                "direction": direction,
                "paired_folds": len(group),
                "rows": group["rows"].dropna().max() if group["rows"].notna().any() else math.nan,
                "features": group["features"].dropna().max() if group["features"].notna().any() else math.nan,
                "pressure_proxy": (
                    group["rows"].dropna().max() * (group["features"].dropna().max() ** 2)
                    if group["rows"].notna().any() and group["features"].notna().any()
                    else math.nan
                ),
                "xgb_test_mean": xgb_mean,
                "mgb_test_mean": mgb_mean,
                "test_improvement": improvement,
                "dataset_winner": winner,
                "fold_mgb_win": int((group["test_winner"] == "mgb_win").sum()),
                "fold_tie": int((group["test_winner"] == "tie").sum()),
                "fold_xgb_win": int((group["test_winner"] == "xgb_win").sum()),
                "xgb_elapsed_sec_mean": group["xgb_elapsed_sec"].mean(),
                "mgb_elapsed_sec_mean": group["mgb_elapsed_sec"].mean(),
                "runtime_ratio_mean": group["runtime_ratio"].mean(),
                "cv_test_agree_folds": int((group["cv_winner"] == group["test_winner"]).sum()),
                "cv_mgb_test_xgb_folds": int(((group["cv_winner"] == "mgb_win") & (group["test_winner"] == "xgb_win")).sum()),
                "cv_xgb_test_mgb_folds": int(((group["cv_winner"] == "xgb_win") & (group["test_winner"] == "mgb_win")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "encoding", "task_type", "dataset"]).reset_index(drop=True)


def summarize_groups(dataset_df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["family", "protocol", "encoding", "task_type"]
    rows = []
    for key, group in dataset_df.groupby(group_cols, dropna=False):
        pair_group = pairs
        for col, value in zip(group_cols, key):
            pair_group = pair_group[pair_group[col] == value]
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "datasets": len(group),
                "paired_folds": int(group["paired_folds"].sum()),
                "dataset_mgb_win": int((group["dataset_winner"] == "mgb_win").sum()),
                "dataset_tie": int((group["dataset_winner"] == "tie").sum()),
                "dataset_xgb_win": int((group["dataset_winner"] == "xgb_win").sum()),
                "fold_mgb_win": int((pair_group["test_winner"] == "mgb_win").sum()),
                "fold_tie": int((pair_group["test_winner"] == "tie").sum()),
                "fold_xgb_win": int((pair_group["test_winner"] == "xgb_win").sum()),
                "median_runtime_ratio": pair_group["runtime_ratio"].median(),
                "mean_runtime_ratio": pair_group["runtime_ratio"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["family", "protocol", "encoding", "task_type", "dataset"]
    for key, group in df.groupby(group_cols, dropna=False):
        ok = group[group["status"] == "ok"]
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "xgb_ok_folds": int(((ok["model"] == "xgb_cpu")).sum()),
                "mgb_ok_folds": int(((ok["model"] == "current_metagb")).sum()),
                "xgb_records": int((group["model"] == "xgb_cpu").sum()),
                "mgb_records": int((group["model"] == "current_metagb").sum()),
                "rows": group["processed_rows"].dropna().max() if group["processed_rows"].notna().any() else math.nan,
                "features": group["processed_features"].dropna().max() if group["processed_features"].notna().any() else math.nan,
                "status_values": ",".join(sorted(set(str(x) for x in group["status"].dropna()))),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_status_estimates() -> pd.DataFrame:
    rows = []
    for status_path in OUT_ROOT.rglob("task_status.json"):
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            continue
        run_dir = status_path.parent
        elapsed = []
        for log_path in run_dir.rglob("*.log"):
            parsed = parse_log(log_path)
            if parsed and parsed.get("status") == "ok" and finite_or_none(parsed.get("task_elapsed_sec")):
                elapsed.append(float(parsed["task_elapsed_sec"]))
        median_elapsed = float(pd.Series(elapsed).median()) if elapsed else math.nan
        total = status.get("total", 0)
        completed = status.get("completed", 0)
        failed = status.get("failed", 0)
        remaining = max(int(total) - int(completed) - int(failed), 0)
        rows.append(
            {
                "run_dir": str(run_dir.relative_to(ROOT)),
                "run_id": status.get("run_id"),
                "total_tasks": total,
                "completed": completed,
                "failed": failed,
                "remaining": remaining,
                "median_ok_elapsed_hours": median_elapsed / 3600 if math.isfinite(median_elapsed) else math.nan,
                "remaining_serial_task_hours": remaining * median_elapsed / 3600 if math.isfinite(median_elapsed) else math.nan,
                "updated_at": status.get("updated_at"),
            }
        )
    return pd.DataFrame(rows).sort_values("run_dir").reset_index(drop=True)


def build_synthetic_mechanism(dataset_df: pd.DataFrame) -> pd.DataFrame:
    syn = dataset_df[dataset_df["family"] == "synthetic_0629"].copy()
    if syn.empty:
        return syn
    parts = syn["dataset"].str.extract(r"(?P<case>.+)__dim(?P<dimension>\d+)$")
    syn["synthetic_case"] = parts["case"]
    syn["dimension"] = pd.to_numeric(parts["dimension"], errors="coerce")
    type_map = {
        "xor_2way": "A_interaction",
        "xor_classification": "A_interaction",
        "xor_3way": "A_interaction",
        "multiplicative": "A_interaction",
        "multiplicative_2way": "A_interaction",
        "conditional_multiplicative": "A_interaction",
        "global_linear": "B_linear_in_leaves",
        "regime_slope": "B_linear_in_leaves",
        "regime_slope_2": "B_linear_in_leaves",
        "regime_slope_multi": "B_linear_in_leaves",
        "feature_switch": "B_linear_in_leaves",
        "regime_logit": "B_linear_in_leaves",
        "categorical_regime_slope": "C_conditional_predictive",
        "categorical_feature_switch": "C_conditional_predictive",
        "mixed_gate_regime": "C_conditional_predictive",
        "sparse_multiplicative": "D_sparse_interaction",
        "sparse_regime_slope": "D_sparse_interaction",
        "sparse_feature_switch": "D_sparse_interaction",
        "pure_step_control": "E_negative_control",
        "curved_control": "E_negative_control",
        "additive_nonlinear_control": "E_negative_control",
        "noise_regime_slope": "F_noise_pressure",
    }
    syn["mechanism_type"] = syn["synthetic_case"].map(type_map).fillna("unknown")
    return syn[
        [
            "mechanism_type",
            "synthetic_case",
            "dimension",
            "paired_folds",
            "primary_metric",
            "xgb_test_mean",
            "mgb_test_mean",
            "test_improvement",
            "dataset_winner",
            "runtime_ratio_mean",
        ]
    ].sort_values(["mechanism_type", "synthetic_case", "dimension"])


def fmt_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return ""
    if abs(x) >= 1000:
        return f"{x:,.1f}"
    if abs(x) >= 10:
        return f"{x:.3f}"
    return f"{x:.6g}"


def to_md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = list(view.columns)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        vals = [fmt_number(row[col]) for col in headers]
        vals = [v.replace("|", "\\|") for v in vals]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_outputs(df: pd.DataFrame, pairs: pd.DataFrame, dataset_df: pd.DataFrame, group_df: pd.DataFrame, coverage: pd.DataFrame, status_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(DERIVED / "fold_results.csv", index=False)
    pairs.to_csv(DERIVED / "paired_fold_results.csv", index=False)
    dataset_df.to_csv(DERIVED / "dataset_level_mgb_vs_xgb.csv", index=False)
    group_df.to_csv(DERIVED / "win_tie_loss_by_group.csv", index=False)
    coverage.to_csv(DERIVED / "coverage_status.csv", index=False)
    status_df.to_csv(DERIVED / "remaining_time_estimates.csv", index=False)
    synthetic_df.to_csv(DERIVED / "synthetic_mechanism_table.csv", index=False)

    (TABLES / "table_coverage_status.md").write_text(
        "# Coverage Status\n\n"
        + to_md_table(
            coverage,
            ["family", "protocol", "encoding", "task_type", "dataset", "xgb_ok_folds", "mgb_ok_folds", "rows", "features", "status_values"],
        )
        + "\n",
        encoding="utf-8",
    )
    (TABLES / "table_win_tie_loss_by_group.md").write_text(
        "# Win/Tie/Loss by Group\n\n"
        + to_md_table(
            group_df,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "datasets",
                "paired_folds",
                "dataset_mgb_win",
                "dataset_tie",
                "dataset_xgb_win",
                "fold_mgb_win",
                "fold_tie",
                "fold_xgb_win",
                "median_runtime_ratio",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (TABLES / "table_dataset_level_mgb_vs_xgb.md").write_text(
        "# Dataset-Level MGB vs XGB\n\n"
        + to_md_table(
            dataset_df,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "dataset",
                "paired_folds",
                "rows",
                "features",
                "primary_metric",
                "xgb_test_mean",
                "mgb_test_mean",
                "test_improvement",
                "dataset_winner",
                "runtime_ratio_mean",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = dataset_df.sort_values(["runtime_ratio_mean"], ascending=False)
    (TABLES / "table_runtime_quality_tradeoff.md").write_text(
        "# Runtime-Quality Tradeoff\n\n"
        + to_md_table(
            runtime,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "dataset",
                "dataset_winner",
                "test_improvement",
                "xgb_elapsed_sec_mean",
                "mgb_elapsed_sec_mean",
                "runtime_ratio_mean",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cv_gap = dataset_df.sort_values(["cv_mgb_test_xgb_folds", "cv_xgb_test_mgb_folds"], ascending=False)
    (TABLES / "table_cv_test_gap.md").write_text(
        "# CV/Test Gap\n\n"
        + to_md_table(
            cv_gap,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "dataset",
                "paired_folds",
                "cv_test_agree_folds",
                "cv_mgb_test_xgb_folds",
                "cv_xgb_test_mgb_folds",
                "dataset_winner",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (TABLES / "table_synthetic_mechanism.md").write_text(
        "# Synthetic Mechanism Table\n\n"
        + to_md_table(
            synthetic_df,
            [
                "mechanism_type",
                "synthetic_case",
                "dimension",
                "paired_folds",
                "primary_metric",
                "xgb_test_mean",
                "mgb_test_mean",
                "test_improvement",
                "dataset_winner",
                "runtime_ratio_mean",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (TABLES / "table_remaining_time_estimates.md").write_text(
        "# Remaining Time Estimates\n\n"
        + to_md_table(
            status_df,
            [
                "run_dir",
                "run_id",
                "total_tasks",
                "completed",
                "failed",
                "remaining",
                "median_ok_elapsed_hours",
                "remaining_serial_task_hours",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    write_analysis_md(group_df, dataset_df, pairs, coverage, status_df, synthetic_df)


def wtl_text(row: pd.Series) -> str:
    return f"{int(row['dataset_mgb_win'])}/{int(row['dataset_tie'])}/{int(row['dataset_xgb_win'])}"


def write_analysis_md(group_df: pd.DataFrame, dataset_df: pd.DataFrame, pairs: pd.DataFrame, coverage: pd.DataFrame, status_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    total_datasets = len(dataset_df)
    total_pairs = len(pairs)
    total_mgb = int((dataset_df["dataset_winner"] == "mgb_win").sum()) if not dataset_df.empty else 0
    total_tie = int((dataset_df["dataset_winner"] == "tie").sum()) if not dataset_df.empty else 0
    total_xgb = int((dataset_df["dataset_winner"] == "xgb_win").sum()) if not dataset_df.empty else 0

    best_mgb = dataset_df[dataset_df["dataset_winner"] == "mgb_win"].sort_values("test_improvement", ascending=False).head(10)
    best_xgb = dataset_df[dataset_df["dataset_winner"] == "xgb_win"].sort_values("test_improvement", ascending=True).head(10)
    synthetic_type = (
        synthetic_df.groupby("mechanism_type")["dataset_winner"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        if not synthetic_df.empty
        else pd.DataFrame()
    )
    for col in ["mgb_win", "tie", "xgb_win"]:
        if col not in synthetic_type.columns:
            synthetic_type[col] = 0

    coverage_summary_rows = []
    for family, group in coverage.groupby("family", dropna=False):
        full = ((group["xgb_ok_folds"] >= 5) & (group["mgb_ok_folds"] >= 5)).sum()
        partial = ((group["xgb_ok_folds"] > 0) & (group["mgb_ok_folds"] > 0) & ~((group["xgb_ok_folds"] >= 5) & (group["mgb_ok_folds"] >= 5))).sum()
        one_sided = ((group["xgb_ok_folds"] == 0) ^ (group["mgb_ok_folds"] == 0)).sum()
        coverage_summary_rows.append(
            {
                "family": family,
                "dataset_cells": len(group),
                "full_5fold_paired": int(full),
                "partial_paired": int(partial),
                "one_sided_only": int(one_sided),
            }
        )
    coverage_summary = pd.DataFrame(coverage_summary_rows)

    def group_line(family: str, protocol: str | None, encoding: str, task_type: str) -> str:
        subset = group_df[(group_df["family"] == family) & (group_df["encoding"] == encoding) & (group_df["task_type"] == task_type)]
        if protocol is not None:
            subset = subset[subset["protocol"] == protocol]
        if subset.empty:
            return "0/0/0"
        row = subset.iloc[0]
        return wtl_text(row)

    tabarena_reg_drop = group_line("tabarena_main_0624", "random600", "drop", "regression")
    tabarena_reg_onehot = group_line("tabarena_main_0624", "random600", "onehot", "regression")
    tabarena_bin_drop = group_line("tabarena_main_0624", "random300", "drop", "binary")
    tabarena_bin_onehot = group_line("tabarena_main_0624", "random300", "onehot", "binary")
    tabarena_multi_drop = group_line("tabarena_main_0624", "random300", "drop", "multiclass")
    tabarena_multi_onehot = group_line("tabarena_main_0624", "random300", "onehot", "multiclass")
    tabarena30_drop = group_line("tabarena30_0629", "xgb_best_mgb30", "drop", "binary")
    tabarena30_onehot_bin = group_line("tabarena30_0629", "xgb_best_mgb30", "onehot", "binary")
    tabarena30_onehot_multi = group_line("tabarena30_0629", "xgb_best_mgb30", "onehot", "multiclass")

    other_total = group_df[group_df["family"] == "tabular_otherdata_0629"]
    other_wtl = (
        f"{int(other_total['dataset_mgb_win'].sum())}/"
        f"{int(other_total['dataset_tie'].sum())}/"
        f"{int(other_total['dataset_xgb_win'].sum())}"
        if not other_total.empty
        else "0/0/0"
    )
    syn_total = group_df[group_df["family"] == "synthetic_0629"]
    syn_wtl = (
        f"{int(syn_total['dataset_mgb_win'].sum())}/"
        f"{int(syn_total['dataset_tie'].sum())}/"
        f"{int(syn_total['dataset_xgb_win'].sum())}"
        if not syn_total.empty
        else "0/0/0"
    )

    lines = [
        "# MGB vs XGB Comprehensive Analysis (0629)",
        "",
        "## Scope",
        "",
        "本文件由 `mgb_datacollection_by0629/scripts/analyze_mgb_vs_xgb.py` 生成，"
        "证据来自 `datacollection0624` 的 TabArena 主体结果以及 `mgb_datacollection_by0629` 中的 TabArena30、tabular other data、synthetic 结果。"
        "比较只纳入 XGB 与 current_metagb 在同一 dataset、outer fold 下均 `status=ok` 的 paired folds。",
        "",
        "口径：AUC 越高越好；RMSE/logloss 越低越好。AUC 绝对差 `<=1e-4` 记为 tie；RMSE/logloss 相对差 `<=0.1%` 记为 tie。",
        "",
        "## Executive Summary",
        "",
        f"- Paired fold 数：{total_pairs}；dataset-level 可比较单元：{total_datasets}。",
        f"- 总体 dataset-level win/tie/loss (MGB/Tie/XGB)：{total_mgb}/{total_tie}/{total_xgb}。",
        "- `tabarena30_0629` 是 fixed-backbone 协议：先取 XGB best backbone，再搜索 `10 pred_fraction x 3 score_agg`，因此应与完整 Random300/600 主表分开解释。",
        "- `synthetic_0629` 是机制证据：它更适合回答 MetaGB 理念是否成立，而不是代表真实 TabArena 总体分布。",
        "",
        "## Coverage Snapshot",
        "",
        to_md_table(coverage_summary, ["family", "dataset_cells", "full_5fold_paired", "partial_paired", "one_sided_only"]),
        "",
        "Coverage reading: one-sided cells are mostly XGB-only or MGB-only unfinished tails, so they are excluded from win/tie/loss. This matters for `tabarena30_0629`, where many one-hot binary tails currently have XGB backbones but no finished MGB30 result yet.",
        "",
        "## Data-Supported Findings",
        "",
        f"**Synthetic mechanism evidence.** Current synthetic paired results give MGB/Tie/XGB = {syn_wtl} at dataset-cell level. The A-D/F mechanism families are strongly positive for MGB, while negative controls are not: this is the cleanest evidence that MGB is doing what it is designed to do rather than simply benefiting from a larger search surface. In concrete terms, conditional multiplicative, regime slope, categorical regime slope, mixed gate, and sparse interaction cases show large RMSE reductions, while additive/curved/step controls do not become uniformly favorable.",
        "",
        f"**TabArena main results are mixed rather than universally positive.** Regression Random600 is {tabarena_reg_drop} under drop and {tabarena_reg_onehot} under onehot; binary Random300 is {tabarena_bin_drop} under drop and {tabarena_bin_onehot} under onehot; multiclass Random300 is {tabarena_multi_drop} under drop and {tabarena_multi_onehot} under onehot. This supports a careful claim: MGB has clear positive pockets, especially some multiclass and selected regression/binary datasets, but the broad real-data average is not a blanket win over XGB.",
        "",
        f"**TabArena30 validates the fixed-backbone tail strategy, but binary gains are small.** Current TabArena30 gives binary drop {tabarena30_drop}, binary onehot {tabarena30_onehot_bin}, and multiclass onehot {tabarena30_onehot_multi}. The median runtime ratio is much lower than full Random300/600 in several groups, because only 30 MGB candidates are searched after choosing the XGB backbone. The multiclass onehot results are currently favorable, while binary results are close and dataset-dependent; many AUC deltas are around 0.001-0.008, so these are table-filling/coverage results more than decisive superiority evidence.",
        "",
        f"**Other tabular data is useful for finding positive pockets but remains mixed overall.** Across current other-tabular cells, MGB/Tie/XGB = {other_wtl}. Strong positives include `vowel.csv`, `kin8nm.csv`, `Moneyball.csv`, `ilpd.csv`, and `phoneme.csv`; strong negatives include `balance-scale.csv`, `boston.csv`, and `energy_efficiency.csv`. This aligns with the working hypothesis that MGB helps when local linear response or conditional/predictive separation is present, but can lose on data where constant-leaf partitioning is already a better inductive bias.",
        "",
        "**Runtime tradeoff.** MGB is consistently more expensive in full Random300/600 settings; median runtime ratios are commonly 10x-60x depending on feature count and task family. TabArena30 reduces this cost by shrinking the MGB-specific search to 30 candidates, with several binary groups around 2x-3x median ratio, but wide/high-pressure data can still be much slower. The right paper framing is therefore not speed, but whether the extra local-linear expressivity buys enough test performance in the mechanism-aligned regimes.",
        "",
        "## Win/Tie/Loss by Family",
        "",
        to_md_table(
            group_df,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "datasets",
                "paired_folds",
                "dataset_mgb_win",
                "dataset_tie",
                "dataset_xgb_win",
                "fold_mgb_win",
                "fold_tie",
                "fold_xgb_win",
                "median_runtime_ratio",
            ],
        ),
        "",
        "## Strongest Positive Cases",
        "",
        to_md_table(
            best_mgb,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "dataset",
                "paired_folds",
                "primary_metric",
                "xgb_test_mean",
                "mgb_test_mean",
                "test_improvement",
                "runtime_ratio_mean",
            ],
        ),
        "",
        "## Strongest Negative Cases",
        "",
        to_md_table(
            best_xgb,
            [
                "family",
                "protocol",
                "encoding",
                "task_type",
                "dataset",
                "paired_folds",
                "primary_metric",
                "xgb_test_mean",
                "mgb_test_mean",
                "test_improvement",
                "runtime_ratio_mean",
            ],
        ),
        "",
        "## Synthetic Mechanism Reading",
        "",
        to_md_table(synthetic_type, ["mechanism_type", "mgb_win", "tie", "xgb_win"]),
        "",
        "Interpretation: synthetic positive cases should be read through the MetaGB design lens: conditional features define regions, while predictive features provide local one-feature linear responses inside leaves. Negative controls are important because they check whether MGB is merely winning everywhere by search noise.",
        "",
        "## Dataset Scale and Runtime",
        "",
        "Dataset rows/features in the tables are the processed training matrix sizes observed in logs or synthetic protocol settings. The pressure proxy is `rows * features^2`, which is the relevant rough cost driver for MetaGB candidate evaluation.",
        "",
        to_md_table(
            dataset_df.sort_values("pressure_proxy", ascending=False).head(30),
            [
                "family",
                "encoding",
                "task_type",
                "dataset",
                "paired_folds",
                "rows",
                "features",
                "pressure_proxy",
                "dataset_winner",
                "runtime_ratio_mean",
            ],
        ),
        "",
        "## CV/Test Gap",
        "",
        "CV/Test gap columns count fold-level disagreements between validation-selected winner and test winner. Large values here mean the model-selection signal is unstable for that dataset, even if the final test mean favors one model.",
        "",
        to_md_table(
            dataset_df.sort_values(["cv_mgb_test_xgb_folds", "cv_xgb_test_mgb_folds"], ascending=False).head(30),
            [
                "family",
                "encoding",
                "task_type",
                "dataset",
                "paired_folds",
                "cv_test_agree_folds",
                "cv_mgb_test_xgb_folds",
                "cv_xgb_test_mgb_folds",
                "dataset_winner",
            ],
        ),
        "",
        "## Remaining Work Estimate",
        "",
        "The estimate below is serial task-hours from observed median completed-task elapsed time in each run directory. Parallel wall time depends on available workers and should be divided by effective worker count only after accounting for slow-tail skew.",
        "",
        to_md_table(
            status_df,
            [
                "run_dir",
                "total_tasks",
                "completed",
                "failed",
                "remaining",
                "median_ok_elapsed_hours",
                "remaining_serial_task_hours",
            ],
        ),
        "",
        "## Evidence Tables",
        "",
        "- `derived/fold_results.csv`: parsed model/fold records.",
        "- `derived/paired_fold_results.csv`: only folds where XGB and MGB both finished.",
        "- `derived/dataset_level_mgb_vs_xgb.csv`: dataset-level aggregation.",
        "- `tables/table_dataset_level_mgb_vs_xgb.md`: full dataset table.",
        "- `tables/table_synthetic_mechanism.md`: synthetic mechanism view.",
        "- `tables/table_runtime_quality_tradeoff.md`: runtime-quality view.",
        "- `tables/table_cv_test_gap.md`: CV/test consistency view.",
    ]
    (OUT_ROOT / "mgb_vs_xgb_comprehensive_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = collect_rows()
    if df.empty:
        raise SystemExit("No parseable rows found.")
    pairs = build_pairs(df)
    dataset_df = summarize_datasets(pairs)
    group_df = summarize_groups(dataset_df, pairs) if not dataset_df.empty else pd.DataFrame()
    coverage = build_coverage(df)
    status_df = build_status_estimates()
    synthetic_df = build_synthetic_mechanism(dataset_df)
    write_outputs(df, pairs, dataset_df, group_df, coverage, status_df, synthetic_df)
    print(f"fold rows: {len(df)}")
    print(f"paired folds: {len(pairs)}")
    print(f"dataset rows: {len(dataset_df)}")
    print(f"outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
