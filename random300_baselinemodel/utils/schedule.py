from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


DEFAULT_TASK_DATASETS = {
    "regression": [
        "regression__Moneyball.csv",
        "regression__energy_efficiency.csv",
        "regression__boston.csv",
        "regression__space_ga.csv",
        "regression__friedman1.csv",
        "regression__abalone.csv",
        "regression__kin8nm.csv",
        "regression__naval_propulsion_plant.csv",
        "regression__cpu_activity.csv",
    ],
    "binary": [
        "binary__haberman.csv",
        "binary__banknote-authentication.csv",
        "binary__breast-w.csv",
        "binary__ilpd.csv",
        "binary__phoneme.csv",
        "binary__credit-approval.csv",
        "binary__mozilla4.csv",
        "binary__pc1.csv",
        "binary__kc1.csv",
        "binary__electricity.csv",
    ],
    "multiclass": [
        "multiclass__balance-scale.csv",
        "multiclass__vowel.csv",
        "multiclass__vehicle.csv",
        "multiclass__segment.csv",
        "multiclass__mfeat-karhunen.csv",
        "multiclass__satimage.csv",
        "multiclass__mfeat-fourier.csv",
        "multiclass__optdigits.csv",
    ],
}

DATASET_INSTANCE_COUNTS: Dict[Tuple[str, str], int] = {
    ("regression", "regression__Moneyball.csv"): 114,
    ("binary", "binary__haberman.csv"): 306,
    ("binary", "binary__ilpd.csv"): 583,
    ("binary", "binary__credit-approval.csv"): 653,
    ("binary", "binary__breast-w.csv"): 683,
    ("regression", "regression__energy_efficiency.csv"): 768,
    ("multiclass", "multiclass__vehicle.csv"): 846,
    ("multiclass", "multiclass__vowel.csv"): 990,
    ("binary", "binary__banknote-authentication.csv"): 1372,
    ("binary", "binary__kc1.csv"): 2109,
    ("multiclass", "multiclass__segment.csv"): 2310,
    ("regression", "regression__space_ga.csv"): 3107,
    ("multiclass", "multiclass__optdigits.csv"): 5620,
    ("multiclass", "multiclass__satimage.csv"): 6430,
    ("regression", "regression__kin8nm.csv"): 8192,
    ("regression", "regression__cpu_activity.csv"): 8192,
    ("regression", "regression__naval_propulsion_plant.csv"): 11934,
    ("binary", "binary__electricity.csv"): 38474,
    ("regression", "regression__abalone.csv"): 4177,
    ("regression", "regression__boston.csv"): 506,
    ("regression", "regression__friedman1.csv"): 2000,
    ("binary", "binary__mozilla4.csv"): 15545,
    ("binary", "binary__pc1.csv"): 1109,
    ("binary", "binary__phoneme.csv"): 5404,
    ("multiclass", "multiclass__balance-scale.csv"): 625,
    ("multiclass", "multiclass__mfeat-fourier.csv"): 2000,
    ("multiclass", "multiclass__mfeat-karhunen.csv"): 2000,
}

DATASET_PRESSURE_PROXY: Dict[Tuple[str, str], int] = {
    ("binary", "binary__haberman.csv"): 2754,
    ("multiclass", "multiclass__balance-scale.csv"): 10000,
    ("binary", "binary__banknote-authentication.csv"): 21952,
    ("regression", "regression__Moneyball.csv"): 22344,
    ("regression", "regression__energy_efficiency.csv"): 49152,
    ("binary", "binary__breast-w.csv"): 55323,
    ("binary", "binary__ilpd.csv"): 58300,
    ("regression", "regression__boston.csv"): 85514,
    ("regression", "regression__space_ga.csv"): 111852,
    ("multiclass", "multiclass__vowel.csv"): 119790,
    ("binary", "binary__phoneme.csv"): 135100,
    ("binary", "binary__credit-approval.csv"): 146925,
    ("regression", "regression__friedman1.csv"): 200000,
    ("regression", "regression__abalone.csv"): 267328,
    ("multiclass", "multiclass__vehicle.csv"): 274104,
    ("binary", "binary__mozilla4.csv"): 388625,
    ("binary", "binary__pc1.csv"): 489069,
    ("regression", "regression__kin8nm.csv"): 524288,
    ("multiclass", "multiclass__segment.csv"): 833910,
    ("binary", "binary__kc1.csv"): 930069,
    ("binary", "binary__electricity.csv"): 1885226,
    ("regression", "regression__naval_propulsion_plant.csv"): 2339064,
    ("regression", "regression__cpu_activity.csv"): 3612672,
    ("multiclass", "multiclass__mfeat-karhunen.csv"): 8192000,
    ("multiclass", "multiclass__satimage.csv"): 8333280,
    ("multiclass", "multiclass__mfeat-fourier.csv"): 11552000,
    ("multiclass", "multiclass__optdigits.csv"): 23019520,
}

CATEGORICAL_HANDLING_ORDER = {"drop": 0, "onehot": 1, "ordinal": 2}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def selected_outer_fold_indices(args: Any) -> List[int]:
    if getattr(args, "run_all_outer_folds", False) or getattr(args, "mode", "") == "full":
        return list(range(int(args.outer_folds)))
    outer_fold_index = int(args.outer_fold_index)
    if outer_fold_index < 0 or outer_fold_index >= int(args.outer_folds):
        raise ValueError("--outer-fold-index is zero-based and must be in [0, outer_folds)")
    return [outer_fold_index]


def select_datasets(args: Any, task_type: str) -> List[str]:
    requested = list(getattr(args, f"{task_type}_datasets"))
    if requested == ["all"]:
        selected = list(DEFAULT_TASK_DATASETS[task_type])
    else:
        selected = requested
    skip_all = set(getattr(args, "skip_datasets", []) or [])
    skip_task = set(getattr(args, f"skip_{task_type}_datasets", []) or [])
    return [dataset for dataset in selected if dataset not in skip_all | skip_task]


def dataset_dir_for_task(args: Any, task_type: str) -> Path:
    mapping = {
        "regression": getattr(args, "regression_dir", ""),
        "binary": getattr(args, "binary_dir", ""),
        "multiclass": getattr(args, "multiclass_dir", ""),
    }
    return Path(mapping[task_type]).expanduser().resolve()


def create_ordered_tasks(args: Any, candidates_by_model: Mapping[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    categorical_handlings = list(getattr(args, "categorical_handlings", ["drop", "onehot"]))
    for model in args.models:
        for categorical_handling in categorical_handlings:
            for task_type in args.task_types:
                dataset_dir = dataset_dir_for_task(args, task_type) if hasattr(args, f"{task_type}_dir") else None
                for dataset_name in select_datasets(args, task_type):
                    dataset_path = str(dataset_dir / dataset_name) if dataset_dir is not None else ""
                    for outer_index in selected_outer_fold_indices(args):
                        task_id = (
                            f"{safe_name(model)}__{safe_name(categorical_handling)}__"
                            f"{safe_name(task_type)}__{safe_name(dataset_name)}__out{outer_index + 1}"
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "model": model,
                                "categorical_handling": categorical_handling,
                                "task_type": task_type,
                                "dataset": dataset_name,
                                "dataset_path": dataset_path,
                                "outer_index": outer_index,
                                "candidates": candidates_by_model[model],
                            }
                        )
    return [task for _, task in sorted(enumerate(tasks), key=_task_sort_key(args))]


def _task_sort_key(args: Any):
    model_order = {model: index for index, model in enumerate(args.models)}

    def key(position_and_task: Tuple[int, Dict[str, Any]]) -> Tuple[Any, ...]:
        position, task = position_and_task
        dataset_key = (task["task_type"], task["dataset"])
        pressure = DATASET_PRESSURE_PROXY.get(dataset_key)
        instance_count = DATASET_INSTANCE_COUNTS.get(dataset_key)
        return (
            model_order.get(task["model"], sys.maxsize),
            CATEGORICAL_HANDLING_ORDER.get(task["categorical_handling"], sys.maxsize),
            pressure if pressure is not None else sys.maxsize,
            instance_count if instance_count is not None else sys.maxsize,
            int(task["outer_index"]),
            position,
        )

    def unwrapped(position_and_task: Tuple[int, Dict[str, Any]]) -> Tuple[Any, ...]:
        return key(position_and_task)

    return unwrapped
