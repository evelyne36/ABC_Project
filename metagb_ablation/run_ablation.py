from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from random300.utils.candidates import (  # noqa: E402
    generate_xgb_candidates,
    load_search_space,
)
from synthetic_metagb.scenarios import SCENARIOS, generate_instance  # noqa: E402

from metagb_ablation.io_utils import (  # noqa: E402
    canonical_hash,
    canonical_json,
    ensure_resume_compatible,
    file_sha256,
    index_hash,
    write_csv,
    write_json,
    write_parquet,
)
from metagb_ablation.modeling import (  # noqa: E402
    SHARED_BACKBONE_KEYS,
    evaluate_variant,
    load_xgboost,
    primary_metric_direction,
    primary_metric_name,
    runtime_args,
    select_xgb_backbone,
    xgboost_runtime_identity,
)
from metagb_ablation.protocols import (  # noqa: E402
    AblationProtocol,
    RealDatasetSpec,
    SyntheticCaseSpec,
    VariantSpec,
    build_variants,
    protocol_by_name,
)
from metagb_ablation.real_data import (  # noqa: E402
    fit_transform_features,
    load_raw_real_dataset,
)
from metagb_ablation.statistics import (  # noqa: E402
    main_ablation_tables,
    pred_sensitivity_tables,
    scoreagg_tables,
    unit_level_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-XGB-backbone MetaGB mechanism ablation. Every frozen MetaGB "
            "configuration receives an outer-test evaluation."
        )
    )
    parser.add_argument("--protocol", choices=["smoke", "formal"], default="smoke")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["real", "synthetic"],
        default=["real", "synthetic"],
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    scoreagg_group = parser.add_mutually_exclusive_group()
    scoreagg_group.add_argument(
        "--include-split-scoreagg",
        dest="include_split_scoreagg",
        action="store_true",
        help="Use the formal 16-variant SplitOnly/Full scoreagg design (default).",
    )
    scoreagg_group.add_argument(
        "--no-split-scoreagg",
        dest="include_split_scoreagg",
        action="store_false",
        help="Use the reduced 14-variant design without SplitOnly mean/median.",
    )
    parser.set_defaults(include_split_scoreagg=True)
    parser.add_argument("--real-datasets", nargs="+", default=None)
    parser.add_argument("--synthetic-cases", nargs="+", default=None)
    parser.add_argument("--dimensions", nargs="+", type=int, default=None)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--real-cv-repeats", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--outer-fold-index", type=int, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=None)
    parser.add_argument("--xgb-candidates", type=int, default=None)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--task-workers", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument(
        "--categorical-handling",
        choices=["drop", "onehot", "ordinal"],
        default="ordinal",
    )
    parser.add_argument(
        "--regression-dir",
        default=str(ROOT / "dataset" / "TabArena_regression"),
    )
    parser.add_argument(
        "--binary-dir",
        default=str(ROOT / "dataset" / "TabArena_binary"),
    )
    parser.add_argument(
        "--multiclass-dir",
        default=str(ROOT / "dataset" / "TabArena_multiclass"),
    )
    parser.add_argument(
        "--xgboost-python-package",
        default=str(ROOT / "beforereview" / "xgboost-master" / "python-package"),
    )
    parser.add_argument(
        "--search-space-json",
        default=str(
            ROOT
            / "metagb_ablation"
            / "configs"
            / "xgb_backbone_search_space.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "metagb_ablation" / "results"),
    )
    return parser


def _normalize_dataset_name(value: str) -> str:
    return value if value.endswith(".csv") else f"{value}.csv"


def resolve_protocol(
    args: argparse.Namespace,
) -> AblationProtocol:
    protocol = protocol_by_name(args.protocol)
    if args.real_datasets:
        requested = {_normalize_dataset_name(value) for value in args.real_datasets}
        selected = tuple(
            item for item in protocol.real_datasets if item.filename in requested
        )
        missing = requested - {item.filename for item in selected}
        if missing:
            available = {
                item.filename: item
                for item in protocol_by_name("formal").real_datasets
            }
            selected = selected + tuple(
                available[name] for name in sorted(missing) if name in available
            )
            unresolved = missing - set(available)
            if unresolved:
                raise ValueError(f"Unknown real diagnostic datasets: {sorted(unresolved)}")
        protocol = replace(protocol, real_datasets=selected)
    if args.synthetic_cases:
        requested_cases = set(args.synthetic_cases)
        available_cases = {
            item.scenario: item
            for item in protocol_by_name("formal").synthetic_cases
        }
        missing = requested_cases - set(available_cases)
        if missing:
            raise ValueError(f"Unknown synthetic diagnostic cases: {sorted(missing)}")
        protocol = replace(
            protocol,
            synthetic_cases=tuple(
                available_cases[name] for name in args.synthetic_cases
            ),
        )
    overrides: Dict[str, Any] = {}
    if args.dimensions:
        overrides["dimensions"] = tuple(args.dimensions)
    if args.data_seeds:
        overrides["synthetic_data_seeds"] = tuple(args.data_seeds)
    if args.real_cv_repeats is not None:
        overrides["real_cv_repeats"] = int(args.real_cv_repeats)
    if args.outer_folds is not None:
        overrides["outer_folds"] = int(args.outer_folds)
    if args.inner_folds is not None:
        overrides["inner_folds"] = int(args.inner_folds)
    if args.synthetic_samples is not None:
        overrides["synthetic_samples"] = int(args.synthetic_samples)
    if args.xgb_candidates is not None:
        overrides["xgb_candidates"] = int(args.xgb_candidates)
    if overrides:
        protocol = replace(protocol, **overrides)
    if protocol.outer_folds < 2 or protocol.inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least two")
    if protocol.xgb_candidates < 1:
        raise ValueError("xgb_candidates must be positive")
    if protocol.real_cv_repeats < 1:
        raise ValueError("real_cv_repeats must be positive")
    return protocol


def _dataset_dir(args: argparse.Namespace, task_type: str) -> Path:
    return Path(getattr(args, f"{task_type}_dir")).expanduser().resolve()


def selected_outer_indices(
    protocol: AblationProtocol,
    outer_fold_index: Optional[int],
) -> Sequence[int]:
    if outer_fold_index is None:
        return tuple(range(protocol.outer_folds))
    if not 0 <= outer_fold_index < protocol.outer_folds:
        raise ValueError(
            f"--outer-fold-index must be in [0, {protocol.outer_folds})"
        )
    return (outer_fold_index,)


def create_task_plan(
    args: argparse.Namespace,
    protocol: AblationProtocol,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    outer_indices = selected_outer_indices(protocol, args.outer_fold_index)
    if "real" in args.domains:
        for dataset in protocol.real_datasets:
            path = _dataset_dir(args, dataset.task_type) / dataset.filename
            if not path.exists():
                raise FileNotFoundError(f"Real dataset not found: {path}")
            for repeat_index in range(protocol.real_cv_repeats):
                split_seed = protocol.cv_seed + repeat_index * 10_007
                unit_id = f"real::{dataset.name}::repeat{repeat_index + 1}"
                for outer_index in outer_indices:
                    tasks.append(
                        {
                            "task_id": (
                                f"real__{dataset.task_type}__{dataset.name}"
                                f"__rep{repeat_index + 1}__out{outer_index + 1}"
                            ),
                            "unit_id": unit_id,
                            "domain": "real",
                            "task_type": dataset.task_type,
                            "dataset": dataset.name,
                            "dataset_path": str(path),
                            "scenario": "",
                            "scenario_role": "real_diagnostic",
                            "dimension": 0,
                            "data_seed": -1,
                            "cv_repeat": repeat_index + 1,
                            "split_seed": split_seed,
                            "outer_index": outer_index,
                        }
                    )
    if "synthetic" in args.domains:
        for case in protocol.synthetic_cases:
            task_type = SCENARIOS[case.scenario].task_type
            for dimension in protocol.dimensions:
                for data_seed in protocol.synthetic_data_seeds:
                    unit_id = (
                        f"synthetic::{case.scenario}::dim{dimension}::seed{data_seed}"
                    )
                    split_seed = protocol.cv_seed + int(data_seed)
                    for outer_index in outer_indices:
                        tasks.append(
                            {
                                "task_id": (
                                    f"synthetic__{case.scenario}__dim{dimension}"
                                    f"__seed{data_seed}__out{outer_index + 1}"
                                ),
                                "unit_id": unit_id,
                                "domain": "synthetic",
                                "task_type": task_type,
                                "dataset": "",
                                "dataset_path": "",
                                "scenario": case.scenario,
                                "scenario_role": case.role,
                                "dimension": int(dimension),
                                "data_seed": int(data_seed),
                                "cv_repeat": 0,
                                "split_seed": split_seed,
                                "outer_index": outer_index,
                            }
                        )
    if args.limit_tasks is not None:
        tasks = tasks[: max(0, int(args.limit_tasks))]
    return tasks


def load_task_data(
    task: Mapping[str, Any],
    protocol: AblationProtocol,
    categorical_handling: str,
) -> Tuple[Any, np.ndarray, int, str]:
    if task["domain"] == "real":
        loaded = load_raw_real_dataset(
            Path(task["dataset_path"]),
            str(task["task_type"]),
        )
        return (
            loaded.features,
            loaded.target,
            int(loaded.n_classes),
            file_sha256(Path(task["dataset_path"])),
        )
    noise = (
        protocol.regression_noise
        if task["task_type"] == "regression"
        else protocol.classification_noise
    )
    X, y = generate_instance(
        str(task["scenario"]),
        int(task["data_seed"]),
        protocol.synthetic_samples,
        noise,
        target_features=int(task["dimension"]),
    )
    data_identity = canonical_hash(
        {
            "scenario": task["scenario"],
            "dimension": task["dimension"],
            "data_seed": task["data_seed"],
            "n_samples": protocol.synthetic_samples,
            "noise": noise,
        }
    )
    n_classes = 0 if task["task_type"] == "regression" else int(len(np.unique(y)))
    return np.asarray(X), np.asarray(y), n_classes, data_identity


def nested_splits(
    *,
    task_type: str,
    X: Any,
    y: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    split_seed: int,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
]:
    outer_splitter = (
        KFold(n_splits=outer_folds, shuffle=True, random_state=split_seed)
        if task_type == "regression"
        else StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=split_seed,
        )
    )
    indices = np.arange(len(y))
    outer = list(
        outer_splitter.split(indices, None if task_type == "regression" else y)
    )
    inner: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for outer_index, (outer_train, _) in enumerate(outer):
        inner_splitter = (
            KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=split_seed + outer_index + 1,
            )
            if task_type == "regression"
            else StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=split_seed + outer_index + 1,
            )
        )
        inner[outer_index] = list(
            inner_splitter.split(
                np.arange(len(outer_train)),
                None if task_type == "regression" else y[outer_train],
            )
        )
    return outer, inner


def split_manifest_entry(
    task: Mapping[str, Any],
    protocol: AblationProtocol,
    categorical_handling: str,
) -> Dict[str, Any]:
    X, y, _, data_hash = load_task_data(task, protocol, categorical_handling)
    outer, inner = nested_splits(
        task_type=str(task["task_type"]),
        X=X,
        y=y,
        outer_folds=protocol.outer_folds,
        inner_folds=protocol.inner_folds,
        split_seed=int(task["split_seed"]),
    )
    outer_train, outer_test = outer[int(task["outer_index"])]
    inner_entries = []
    for inner_index, (relative_train, relative_validation) in enumerate(
        inner[int(task["outer_index"])]
    ):
        absolute_train = outer_train[relative_train]
        absolute_validation = outer_train[relative_validation]
        if set(absolute_train) & set(absolute_validation):
            raise AssertionError("Inner train and validation overlap")
        if set(outer_test) & (
            set(absolute_train) | set(absolute_validation)
        ):
            raise AssertionError("Outer test leaked into inner CV")
        inner_entries.append(
            {
                "inner_fold": inner_index + 1,
                "train_hash": index_hash(absolute_train),
                "validation_hash": index_hash(absolute_validation),
            }
        )
    entry = {
        "task_id": task["task_id"],
        "data_hash": data_hash,
        "split_seed": int(task["split_seed"]),
        "outer_fold": int(task["outer_index"]) + 1,
        "outer_train_hash": index_hash(outer_train),
        "outer_test_hash": index_hash(outer_test),
        "inner": inner_entries,
    }
    entry["inner_manifest_hash"] = canonical_hash(inner_entries)
    return entry


def build_split_manifest(
    tasks: Sequence[Mapping[str, Any]],
    protocol: AblationProtocol,
    categorical_handling: str,
) -> Dict[str, Any]:
    entries = [
        split_manifest_entry(task, protocol, categorical_handling)
        for task in tasks
    ]
    return {
        "entries": entries,
        "manifest_hash": canonical_hash(entries),
        "task_count": len(entries),
    }


def _select_synthetic_backbone_cell(
    *,
    scenario: str,
    dimension: int,
    protocol: AblationProtocol,
    candidates: Sequence[Dict[str, Any]],
    xgboost_python_package: Path,
    n_jobs: int,
    categorical_handling: str,
) -> Tuple[str, Dict[str, Any]]:
    xgb = load_xgboost(xgboost_python_package)
    task_type = SCENARIOS[scenario].task_type
    noise = (
        protocol.regression_noise
        if task_type == "regression"
        else protocol.classification_noise
    )
    X, y = generate_instance(
        scenario,
        protocol.synthetic_backbone_seed,
        protocol.synthetic_samples,
        noise,
        target_features=dimension,
    )
    n_classes = 0 if task_type == "regression" else int(len(np.unique(y)))
    splitter = (
        KFold(
            n_splits=protocol.inner_folds,
            shuffle=True,
            random_state=protocol.cv_seed,
        )
        if task_type == "regression"
        else StratifiedKFold(
            n_splits=protocol.inner_folds,
            shuffle=True,
            random_state=protocol.cv_seed,
        )
    )
    inner = list(
        splitter.split(X, None if task_type == "regression" else y)
    )
    model_args = runtime_args(
        model_seed=protocol.model_seed,
        n_jobs=n_jobs,
        categorical_handling=categorical_handling,
    )
    selection = select_xgb_backbone(
        xgb=xgb,
        candidates=candidates,
        X_outer_train=np.asarray(X),
        y_outer_train=np.asarray(y),
        inner_splits=inner,
        args=model_args,
        task_type=task_type,
        n_classes=n_classes,
    )
    key = f"{scenario}::dim{dimension}"
    return key, {
        "scenario": scenario,
        "dimension": dimension,
        "development_seed": protocol.synthetic_backbone_seed,
        "validation_primary_metric": selection.validation_primary_metric,
        "validation_primary_metric_std": selection.validation_primary_metric_std,
        "validation_fit_sec": selection.validation_fit_sec,
        "validation_predict_sec": selection.validation_predict_sec,
        "selected_candidate_id": selection.candidate_id,
        "selected_candidate_index": selection.candidate_index,
        "params": selection.params,
    }


def select_synthetic_backbones(
    *,
    tasks: Sequence[Mapping[str, Any]],
    protocol: AblationProtocol,
    candidates: Sequence[Dict[str, Any]],
    xgboost_python_package: Path,
    task_workers: int,
    n_jobs: int,
    categorical_handling: str,
) -> Dict[str, Dict[str, Any]]:
    keys = sorted(
        {
            (str(task["scenario"]), int(task["dimension"]))
            for task in tasks
            if task["domain"] == "synthetic"
        }
    )
    if not keys:
        return {}
    selected: Dict[str, Dict[str, Any]] = {}
    worker_count = min(max(1, int(task_workers)), len(keys))
    if worker_count == 1:
        for index, (scenario, dimension) in enumerate(keys, start=1):
            key, payload = _select_synthetic_backbone_cell(
                scenario=scenario,
                dimension=dimension,
                protocol=protocol,
                candidates=candidates,
                xgboost_python_package=xgboost_python_package,
                n_jobs=n_jobs,
                categorical_handling=categorical_handling,
            )
            selected[key] = payload
            print(
                f"[backbone {index}/{len(keys)}] {key}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _select_synthetic_backbone_cell,
                    scenario=scenario,
                    dimension=dimension,
                    protocol=protocol,
                    candidates=candidates,
                    xgboost_python_package=xgboost_python_package,
                    n_jobs=n_jobs,
                    categorical_handling=categorical_handling,
                ): (scenario, dimension)
                for scenario, dimension in keys
            }
            for index, future in enumerate(as_completed(futures), start=1):
                key, payload = future.result()
                selected[key] = payload
                print(
                    f"[backbone {index}/{len(keys)}] {key}",
                    flush=True,
                )
    return {key: selected[key] for key in sorted(selected)}


def _prediction_rows(
    *,
    run_id: str,
    config_hash: str,
    variant_hash: str,
    execution_hash: str,
    backbone_hash: str,
    task: Mapping[str, Any],
    variant: VariantSpec,
    absolute_test_indices: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for position, sample_id in enumerate(absolute_test_indices):
        probability = (
            ""
            if proba is None
            else json.dumps(np.asarray(proba[position]).tolist())
        )
        rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "variant_hash": variant_hash,
                "execution_hash": execution_hash,
                "backbone_hash": backbone_hash,
                "task_id": task["task_id"],
                "unit_id": task["unit_id"],
                "domain": task["domain"],
                "task_type": task["task_type"],
                "dataset": task["dataset"],
                "scenario": task["scenario"],
                "dimension": task["dimension"],
                "data_seed": task["data_seed"],
                "cv_repeat": task["cv_repeat"],
                "outer_fold": int(task["outer_index"]) + 1,
                "sample_id": int(sample_id),
                "variant_id": variant.variant_id,
                "y_true": float(y_true[position]),
                "y_pred": float(pred[position]),
                "probabilities_json": probability,
            }
        )
    return rows


def run_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    task = payload["task"]
    protocol = AblationProtocol(
        real_datasets=tuple(
            RealDatasetSpec(**item) for item in payload["protocol"]["real_datasets"]
        ),
        synthetic_cases=tuple(
            SyntheticCaseSpec(**item)
            for item in payload["protocol"]["synthetic_cases"]
        ),
        **{
            key: value
            for key, value in payload["protocol"].items()
            if key not in {"real_datasets", "synthetic_cases"}
        },
    )
    variants = [VariantSpec(**item) for item in payload["variants"]]
    X, y, n_classes, data_hash = load_task_data(
        task,
        protocol,
        payload["categorical_handling"],
    )
    outer, inner = nested_splits(
        task_type=task["task_type"],
        X=X,
        y=y,
        outer_folds=protocol.outer_folds,
        inner_folds=protocol.inner_folds,
        split_seed=task["split_seed"],
    )
    outer_train, outer_test = outer[int(task["outer_index"])]
    y_train, y_test = y[outer_train], y[outer_test]
    prepared_inner_folds = None
    if task["domain"] == "real":
        raw_outer_train = X.iloc[outer_train]
        raw_outer_test = X.iloc[outer_test]
        X_train, X_test = fit_transform_features(
            raw_outer_train,
            raw_outer_test,
            payload["categorical_handling"],
        )
        prepared_inner_folds = []
        for relative_train, relative_validation in inner[int(task["outer_index"])]:
            inner_X_train, inner_X_validation = fit_transform_features(
                raw_outer_train.iloc[relative_train],
                raw_outer_train.iloc[relative_validation],
                payload["categorical_handling"],
            )
            prepared_inner_folds.append(
                (
                    inner_X_train,
                    y_train[relative_train],
                    inner_X_validation,
                    y_train[relative_validation],
                )
            )
    else:
        X_train, X_test = X[outer_train], X[outer_test]
    xgb = load_xgboost(Path(payload["xgboost_python_package"]))
    model_args = runtime_args(
        model_seed=protocol.model_seed,
        n_jobs=payload["n_jobs"],
        categorical_handling=payload["categorical_handling"],
    )
    if task["domain"] == "synthetic":
        key = f"{task['scenario']}::dim{task['dimension']}"
        synthetic_selection = payload["synthetic_backbones"][key]
        backbone = dict(synthetic_selection["params"])
        selection_payload = dict(synthetic_selection)
        selection_payload["source"] = "synthetic_development_seed"
    else:
        selection = select_xgb_backbone(
            xgb=xgb,
            candidates=payload["xgb_candidates"],
            prepared_inner_folds=prepared_inner_folds,
            args=model_args,
            task_type=task["task_type"],
            n_classes=n_classes,
        )
        backbone = selection.params
        selection_payload = {
            "selected_candidate_id": selection.candidate_id,
            "selected_candidate_index": selection.candidate_index,
            "validation_primary_metric": selection.validation_primary_metric,
            "validation_primary_metric_std": selection.validation_primary_metric_std,
            "validation_fit_sec": selection.validation_fit_sec,
            "validation_predict_sec": selection.validation_predict_sec,
            "source": "real_outer_train_inner_cv",
        }
    backbone_hash = canonical_hash(backbone)
    split_entry = split_manifest_entry(
        task,
        protocol,
        payload["categorical_handling"],
    )
    fold_rows: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    n_features = int(X_train.shape[1])
    for variant in variants:
        started = time.perf_counter()
        try:
            outcome = evaluate_variant(
                xgb=xgb,
                variant=variant,
                backbone=backbone,
                args=model_args,
                task_type=task["task_type"],
                n_classes=n_classes,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            metrics = outcome["metrics"]
            status = "ok"
            error = ""
            predictions.extend(
                _prediction_rows(
                    run_id=payload["run_id"],
                    config_hash=payload["config_hash"],
                    variant_hash=payload["variant_hash"],
                    execution_hash=payload["execution_hash"],
                    backbone_hash=backbone_hash,
                    task=task,
                    variant=variant,
                    absolute_test_indices=outer_test,
                    y_true=y_test,
                    pred=outcome["pred"],
                    proba=outcome["proba"],
                )
            )
            params_json = canonical_json(outcome["params"])
            params_hash = outcome["params_hash"]
            fit_sec = outcome["fit_sec"]
            predict_sec = outcome["predict_sec"]
            primary = outcome["test_primary_metric"]
        except Exception as exc:
            metrics = {}
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            params_json = ""
            params_hash = ""
            fit_sec = float("nan")
            predict_sec = float("nan")
            primary = float("nan")
        pred_fraction = variant.pred_fraction
        expected_candidates = (
            float(pred_fraction * n_features)
            if pred_fraction is not None
            else float("nan")
        )
        zero_candidate_probability = (
            float((1.0 - pred_fraction) ** n_features)
            if pred_fraction is not None
            else float("nan")
        )
        fold_rows.append(
            {
                **task,
                "run_id": payload["run_id"],
                "config_hash": payload["config_hash"],
                "variant_hash": payload["variant_hash"],
                "execution_hash": payload["execution_hash"],
                "backbone_hash": backbone_hash,
                "outer_fold": int(task["outer_index"]) + 1,
                "n_train": int(len(outer_train)),
                "n_test": int(len(outer_test)),
                "n_features": n_features,
                "n_classes": n_classes,
                "model_seed": protocol.model_seed,
                "categorical_handling": payload["categorical_handling"],
                "data_hash": data_hash,
                "outer_train_hash": split_entry["outer_train_hash"],
                "outer_test_hash": split_entry["outer_test_hash"],
                "inner_manifest_hash": split_entry["inner_manifest_hash"],
                "variant_id": variant.variant_id,
                "family": variant.family,
                "leaf_mode": variant.leaf_mode,
                "pred_fraction": pred_fraction,
                "score_agg": variant.score_agg,
                "score_agg_label": variant.score_agg_label,
                "leaf_only_use_bias": variant.leaf_only_use_bias,
                "expected_pred_candidates": expected_candidates,
                "probability_no_pred_candidate": zero_candidate_probability,
                "primary_metric": primary_metric_name(task["task_type"]),
                "primary_metric_direction": primary_metric_direction(
                    task["task_type"]
                ),
                "test_primary_metric": primary,
                "test_rmse": metrics.get("rmse", float("nan")),
                "test_logloss": metrics.get("logloss", float("nan")),
                "test_auc": metrics.get("auc", float("nan")),
                "test_accuracy": metrics.get("accuracy", float("nan")),
                "test_balanced_accuracy": metrics.get(
                    "balanced_accuracy",
                    float("nan"),
                ),
                "fit_sec": fit_sec,
                "predict_sec": predict_sec,
                "elapsed_sec": time.perf_counter() - started,
                "status": status,
                "error": error,
                "params_hash": params_hash,
                "params_json": params_json,
            }
        )
    backbone_row = {
        "run_id": payload["run_id"],
        "config_hash": payload["config_hash"],
        "variant_hash": payload["variant_hash"],
        "execution_hash": payload["execution_hash"],
        "backbone_hash": backbone_hash,
        "task_id": task["task_id"],
        "unit_id": task["unit_id"],
        "domain": task["domain"],
        "task_type": task["task_type"],
        "dataset": task["dataset"],
        "scenario": task["scenario"],
        "dimension": task["dimension"],
        "data_seed": task["data_seed"],
        "cv_repeat": task["cv_repeat"],
        "outer_fold": int(task["outer_index"]) + 1,
        "data_hash": data_hash,
        "outer_train_hash": split_entry["outer_train_hash"],
        "outer_test_hash": split_entry["outer_test_hash"],
        "inner_manifest_hash": split_entry["inner_manifest_hash"],
        "search_space_hash": payload["search_space_hash"],
        "xgboost_source_hash": payload["xgboost_source_hash"],
        **selection_payload,
        "params_hash": backbone_hash,
        "params_json": canonical_json(backbone),
    }
    return {
        "run_id": payload["run_id"],
        "task_id": task["task_id"],
        "task_hash": payload["task_hash"],
        "config_hash": payload["config_hash"],
        "variant_hash": payload["variant_hash"],
        "execution_hash": payload["execution_hash"],
        "backbone_hash": backbone_hash,
        "fold_rows": fold_rows,
        "predictions": predictions,
        "backbone_row": backbone_row,
    }


def _artifact_catalog() -> List[Dict[str, Any]]:
    descriptions = {
        "protocol.json": "Frozen protocol, CLI filters, paths, and config hash.",
        "variant_manifest.csv": "One row per frozen ablation variant.",
        "task_plan.csv": "One row per outer evaluation task.",
        "split_manifest.json": "Outer/inner split hashes and leakage audit.",
        "execution_manifest.json": "Frozen runtime, candidate, and synthetic-backbone identity.",
        "synthetic_backbones.json": "Development-seed XGB backbones for synthetic cells.",
        "backbone_manifest.csv": "Selected XGB public backbone per evaluation task.",
        "tasks/*.json": "Atomic resumable task shards.",
        "fold_results.csv": "Outer-test metrics for every task × variant.",
        "predictions.parquet": "Per-sample outer-test predictions for paired resampling.",
        "unit_results.csv": "Fold-aggregated independent seed/repeat units.",
        "main_ablation.csv": "Four-row main ablation summaries.",
        "main_effects.csv": "Paired split/leaf/total/full-vs-XGB effects.",
        "pred_sensitivity.csv": "Raw SplitOnly/Full dose-response summaries.",
        "pred_effects.csv": "Paired split/leaf/total effect curves.",
        "scoreagg_ablation.csv": (
            "SplitOnly/Full p=1 greedy/mean/median summaries."
        ),
        "scoreagg_effects.csv": "Paired mean/median effects versus greedy.",
        "completeness_audit.json": "Expected-versus-actual rows and hard pass flag.",
        "run_metadata.json": "Runtime, completion state, and artifact row counts.",
        "SYNTHETIC_CONCLUSION.md": (
            "Human-authored Synthetic evidence synthesis required before Real."
        ),
        "synthetic_analysis_approval.json": (
            "Hash-bound approval of the completed Synthetic conclusion."
        ),
    }
    optional_runtime_artifacts = {
        "synthetic_backbones.json",
        "SYNTHETIC_CONCLUSION.md",
        "synthetic_analysis_approval.json",
    }
    return [
        {
            "path": path,
            "description": description,
            "required_for_complete_run": path not in optional_runtime_artifacts,
        }
        for path, description in descriptions.items()
    ]


def _collect_shards(
    shard_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    backbone_rows: List[Dict[str, Any]] = []
    for path in sorted(shard_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_rows.extend(payload.get("fold_rows", []))
        predictions.extend(payload.get("predictions", []))
        if payload.get("backbone_row"):
            backbone_rows.append(payload["backbone_row"])
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(predictions),
        pd.DataFrame(backbone_rows),
    )


def _source_identity() -> Dict[str, Any]:
    paths = [
        ROOT / "metagb_ablation" / "protocols.py",
        ROOT / "metagb_ablation" / "io_utils.py",
        ROOT / "metagb_ablation" / "modeling.py",
        ROOT / "metagb_ablation" / "real_data.py",
        ROOT / "metagb_ablation" / "statistics.py",
        ROOT / "metagb_ablation" / "run_ablation.py",
        ROOT / "random300" / "utils" / "candidates.py",
        ROOT / "random300" / "utils" / "metrics.py",
        ROOT / "random300" / "utils" / "runner.py",
        ROOT / "synthetic_metagb" / "scenarios.py",
        ROOT
        / "beforereview"
        / "xgboost-master"
        / "src"
        / "tree"
        / "updater_metagb_hist.cc",
        ROOT
        / "beforereview"
        / "xgboost-master"
        / "src"
        / "tree"
        / "param.h",
    ]
    files = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in paths
    }
    return {
        "files": files,
        "combined_hash": canonical_hash(files),
    }


def _environment_identity() -> Dict[str, str]:
    package_versions: Dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **package_versions,
    }


def _validate_synthetic_backbones(
    tasks: Sequence[Mapping[str, Any]],
    backbones: Mapping[str, Any],
) -> None:
    expected = {
        f"{task['scenario']}::dim{task['dimension']}"
        for task in tasks
        if task["domain"] == "synthetic"
    }
    actual = set(backbones)
    if actual != expected:
        raise ValueError(
            "Synthetic backbone key mismatch: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    for key, payload in backbones.items():
        if "params" not in payload:
            raise ValueError(f"Synthetic backbone has no params: {key}")
        if int(payload.get("development_seed", -1)) <= 0:
            raise ValueError(f"Synthetic backbone has invalid development seed: {key}")


def ensure_execution_compatible(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("execution_hash") != payload.get("execution_hash"):
            raise ValueError(
                "Execution identity changed within the same run-id; "
                "synthetic backbone, runtime binary, candidates, or n_jobs differ"
            )
        return
    write_json(path, payload)


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
    issues: List[str] = []
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
                failed = frame.loc[frame["status"] != "ok", "variant_id"].tolist()
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
                            f"public_param_mismatch:{task_id}:{row['variant_id']}:{key}"
                        )
                xgb_lambda = float(
                    xgb_params.get("reg_lambda", xgb_params.get("lambda_l2", 1.0))
                )
                metagb_lambda = float(params.get("metagb_lambda_l2", float("nan")))
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
                    issues.append(f"split_manifest_value_mismatch:{task_id}:{field}")

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
                    issues.append(f"backbone_params_hash_mismatch:{row['task_id']}")
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
            y_consistency = task_predictions.groupby("sample_id")["y_true"].nunique()
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
                        or not np.isclose(proba.sum(), 1.0, rtol=1e-5, atol=1e-5)
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


def _execute(
    payloads: Sequence[Dict[str, Any]],
    *,
    shard_dir: Path,
    task_workers: int,
) -> None:
    if not payloads:
        return
    if task_workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            result = run_task(payload)
            write_json(shard_dir / f"{result['task_id']}.json", result)
            print(
                f"[{index}/{len(payloads)}] {result['task_id']}",
                flush=True,
            )
        return
    with ProcessPoolExecutor(
        max_workers=min(task_workers, len(payloads))
    ) as executor:
        future_map = {
            executor.submit(run_task, payload): payload for payload in payloads
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            write_json(shard_dir / f"{result['task_id']}.json", result)
            print(
                f"[{index}/{len(payloads)}] {result['task_id']}",
                flush=True,
            )


def main() -> int:
    args = build_parser().parse_args()
    protocol = resolve_protocol(args)
    variants = build_variants(args.include_split_scoreagg)
    run_id = args.run_id or (
        f"{protocol.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir = Path(args.output_root).expanduser().resolve() / run_id
    shard_dir = results_dir / "tasks"
    results_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    search_space_path = Path(args.search_space_json).expanduser().resolve()
    xgboost_python_package = (
        Path(args.xgboost_python_package).expanduser().resolve()
    )
    tasks = create_task_plan(args, protocol)
    split_manifest = build_split_manifest(
        tasks,
        protocol,
        args.categorical_handling,
    )
    runtime_identity = xgboost_runtime_identity(xgboost_python_package)
    environment_identity = _environment_identity()
    source_identity = _source_identity()
    protocol_payload = {
        "protocol": protocol.to_dict(),
        "domains": list(args.domains),
        "categorical_handling": args.categorical_handling,
        "outer_fold_index": args.outer_fold_index,
        "include_split_scoreagg": bool(args.include_split_scoreagg),
        "variant_count": len(variants),
        "variant_hash": canonical_hash([variant.to_dict() for variant in variants]),
        "search_seed": args.search_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "n_jobs": args.n_jobs,
        "search_space_json": str(search_space_path),
        "search_space_hash": file_sha256(search_space_path),
        "xgboost_python_package": str(xgboost_python_package),
        "xgboost_runtime": runtime_identity,
        "environment_identity": environment_identity,
        "source_identity": source_identity,
        "split_manifest_hash": split_manifest["manifest_hash"],
    }
    protocol_payload["config_hash"] = canonical_hash(protocol_payload)
    protocol_path = results_dir / "protocol.json"
    ensure_resume_compatible(
        protocol_path,
        protocol_payload,
        resume=args.resume,
    )
    if not protocol_path.exists():
        write_json(protocol_path, protocol_payload)
    write_csv(
        results_dir / "variant_manifest.csv",
        [variant.to_dict() for variant in variants],
    )
    write_csv(results_dir / "task_plan.csv", tasks)
    write_json(results_dir / "split_manifest.json", split_manifest)
    write_json(results_dir / "artifact_catalog.json", _artifact_catalog())

    if args.plan_only:
        write_json(
            results_dir / "run_metadata.json",
            {
                "run_id": run_id,
                "plan_only": True,
                "completed": False,
                "task_count": len(tasks),
                "variant_count": len(variants),
                "expected_fold_rows": len(tasks) * len(variants),
                "results_dir": str(results_dir),
            },
        )
        print(f"plan_only=True results_dir={results_dir}")
        print(
            f"tasks={len(tasks)} variants={len(variants)} "
            f"expected_fold_rows={len(tasks) * len(variants)}"
        )
        return 0

    search_space = load_search_space(search_space_path)
    xgb_candidates, warnings = generate_xgb_candidates(
        search_space,
        protocol.xgb_candidates,
        args.search_seed,
        None,
    )
    synthetic_backbone_path = results_dir / "synthetic_backbones.json"
    if synthetic_backbone_path.exists():
        synthetic_backbones = json.loads(
            synthetic_backbone_path.read_text(encoding="utf-8")
        )
    else:
        synthetic_backbones = select_synthetic_backbones(
            tasks=tasks,
            protocol=protocol,
            candidates=xgb_candidates,
            xgboost_python_package=xgboost_python_package,
            task_workers=args.task_workers,
            n_jobs=args.n_jobs,
            categorical_handling=args.categorical_handling,
        )
        if synthetic_backbones:
            write_json(synthetic_backbone_path, synthetic_backbones)
    _validate_synthetic_backbones(tasks, synthetic_backbones)
    execution_payload = {
        "protocol_config_hash": protocol_payload["config_hash"],
        "variant_hash": protocol_payload["variant_hash"],
        "xgb_candidates_hash": canonical_hash(xgb_candidates),
        "synthetic_backbones_hash": canonical_hash(synthetic_backbones),
        "xgboost_runtime": runtime_identity,
        "environment_identity": environment_identity,
        "n_jobs": args.n_jobs,
    }
    execution_payload["execution_hash"] = canonical_hash(execution_payload)
    execution_path = results_dir / "execution_manifest.json"
    ensure_execution_compatible(execution_path, execution_payload)

    common_payload = {
        "run_id": run_id,
        "config_hash": protocol_payload["config_hash"],
        "variant_hash": protocol_payload["variant_hash"],
        "execution_hash": execution_payload["execution_hash"],
        "environment_identity": environment_identity,
        "protocol": asdict(protocol),
        "variants": [variant.to_dict() for variant in variants],
        "categorical_handling": args.categorical_handling,
        "xgboost_python_package": str(xgboost_python_package),
        "n_jobs": args.n_jobs,
        "xgb_candidates": xgb_candidates,
        "synthetic_backbones": synthetic_backbones,
        "search_space_hash": protocol_payload["search_space_hash"],
        "xgboost_source_hash": source_identity["combined_hash"],
    }
    payloads: List[Dict[str, Any]] = []
    for task in tasks:
        task_hash = canonical_hash(
            {
                "execution_hash": execution_payload["execution_hash"],
                "task": task,
            }
        )
        shard_path = shard_dir / f"{task['task_id']}.json"
        if shard_path.exists():
            existing = json.loads(shard_path.read_text(encoding="utf-8"))
            if existing.get("task_hash") != task_hash:
                raise ValueError(
                    f"Shard hash mismatch for {task['task_id']}; "
                    "use a new run-id"
                )
            continue
        payloads.append(
            {
                **common_payload,
                "task": task,
                "task_hash": task_hash,
            }
        )

    started = time.perf_counter()
    _execute(
        payloads,
        shard_dir=shard_dir,
        task_workers=max(1, int(args.task_workers)),
    )
    fold_df, prediction_df, backbone_df = _collect_shards(shard_dir)
    write_csv(results_dir / "fold_results.csv", fold_df)
    write_parquet(results_dir / "predictions.parquet", prediction_df)
    write_csv(results_dir / "backbone_manifest.csv", backbone_df)

    audit = build_completeness_audit(
        tasks=tasks,
        variants=variants,
        fold_results=fold_df,
        predictions=prediction_df,
        backbones=backbone_df,
        split_manifest=split_manifest,
        expected_identity={
            "run_id": run_id,
            "config_hash": str(protocol_payload["config_hash"]),
            "execution_hash": str(execution_payload["execution_hash"]),
            "variant_hash": str(protocol_payload["variant_hash"]),
        },
        shards=[
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(shard_dir.glob("*.json"))
        ],
    )
    audit["candidate_warnings"] = warnings
    write_json(results_dir / "completeness_audit.json", audit)

    if audit["passed"]:
        unit_df = unit_level_results(fold_df, prediction_df)
    else:
        unit_df = unit_level_results(pd.DataFrame(), pd.DataFrame())
    main_df, main_effect_df = main_ablation_tables(
        unit_df,
        seed=args.bootstrap_seed,
    )
    pred_df, pred_effect_df = pred_sensitivity_tables(
        unit_df,
        seed=args.bootstrap_seed,
    )
    scoreagg_df, scoreagg_effect_df = scoreagg_tables(
        unit_df,
        seed=args.bootstrap_seed,
    )
    write_csv(results_dir / "unit_results.csv", unit_df)
    write_csv(results_dir / "main_ablation.csv", main_df)
    write_csv(results_dir / "main_effects.csv", main_effect_df)
    write_csv(results_dir / "pred_sensitivity.csv", pred_df)
    write_csv(results_dir / "pred_effects.csv", pred_effect_df)
    write_csv(results_dir / "scoreagg_ablation.csv", scoreagg_df)
    write_csv(results_dir / "scoreagg_effects.csv", scoreagg_effect_df)
    metadata = {
        "run_id": run_id,
        "protocol": protocol.name,
        "confirmatory": protocol.confirmatory,
        "config_hash": protocol_payload["config_hash"],
        "variant_hash": protocol_payload["variant_hash"],
        "execution_hash": execution_payload["execution_hash"],
        "environment_identity": environment_identity,
        "completed": bool(audit["passed"]),
        "elapsed_this_invocation_sec": time.perf_counter() - started,
        "task_count": len(tasks),
        "variant_count": len(variants),
        "task_workers": args.task_workers,
        "n_jobs": args.n_jobs,
        "fold_result_rows": len(fold_df),
        "prediction_rows": len(prediction_df),
        "unit_result_rows": len(unit_df),
        "results_dir": str(results_dir),
    }
    write_json(results_dir / "run_metadata.json", metadata)
    print(
        f"results_dir={results_dir} completed={audit['passed']} "
        f"fold_rows={len(fold_df)} predictions={len(prediction_df)}"
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
