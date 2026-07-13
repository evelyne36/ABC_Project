from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from random600.utils.runner import safe_name  # noqa: E402
from timeseries_extra_random300.scripts.run_timeseries_benchmark import (  # noqa: E402
    build_parser as build_benchmark_parser,
    normalize_args as normalize_benchmark_args,
    run_all,
)
from timeseries_extra_random300.scripts.summarize_results import summarize  # noqa: E402


DEFAULT_MODELS = ["current_metagb", "xgb_cpu", "lgbm_cpu", "catboost_cpu"]
DEFAULT_SEEDS = ["42"]
DEFAULT_SEQ_LENS = ["24", "48"]
DEFAULT_PRED_LENS = ["1", "6", "12"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tree-model time-series benchmark in dataset tiers."
    )
    parser.add_argument(
        "--tier-config",
        default=str(ADAPTER_ROOT / "configs" / "tiered_tree_benchmark.json"),
        help="JSON file containing ordered tiers with name/datasets fields.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=None,
        help="Optional tier names to run. Defaults to all tiers in config order.",
    )
    parser.add_argument(
        "--base-run-id",
        default=None,
        help="Parent run id. Each seed/tier writes to <base-run-id>__seed<seed>__<tier-name>.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned tier commands without running.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--split-protocol", choices=["nested", "fixed"], default="fixed")
    parser.add_argument("--feature-mode", choices=["window", "window_summary", "origin_state", "tabpfn_ts"], default="window_summary")
    parser.add_argument("--target-mode", choices=["direct", "sequence"], default="sequence")
    parser.add_argument("--output-scope", choices=["target", "all"], default="target")
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--seq-lens", nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--pred-lens", nargs="+", default=DEFAULT_PRED_LENS)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--max-raw-rows", type=int, default=0)
    parser.add_argument(
        "--max-features",
        type=int,
        default=64,
        help="Channel cap passed through to the benchmark runner. Default keeps PEMS feasible.",
    )
    parser.add_argument("--search-space-json", default=str(ADAPTER_ROOT / "configs" / "timeseries_search_space.json"))
    parser.add_argument("--xgboost-python-package", default=str(PROJECT_ROOT / "beforereview" / "xgboost-master" / "python-package"))
    parser.add_argument("--task-workers", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output-root", default=str(ADAPTER_ROOT))
    parser.add_argument("--no-time-features", dest="include_time_features", action="store_false", default=True)
    parser.add_argument("--metagb-fixed-score-agg", type=int, choices=[0, 1, 2], default=None)
    return parser


def load_tier_config(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError(f"{path} must contain a non-empty 'tiers' list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for tier in tiers:
        name = str(tier.get("name", "")).strip()
        datasets = tier.get("datasets", [])
        if not name:
            raise ValueError(f"{path} contains a tier without a name")
        if name in seen:
            raise ValueError(f"{path} contains duplicate tier name: {name}")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"Tier {name} must contain a non-empty datasets list")
        seen.add(name)
        normalized.append(
            {
                "name": safe_name(name),
                "description": str(tier.get("description", "")),
                "datasets": [str(dataset) for dataset in datasets],
            }
        )
    return normalized


def select_tiers(tiers: Sequence[dict[str, Any]], requested: Sequence[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return list(tiers)
    requested_safe = [safe_name(name) for name in requested]
    by_name = {tier["name"]: tier for tier in tiers}
    missing = [name for name in requested_safe if name not in by_name]
    if missing:
        available = ", ".join(tier["name"] for tier in tiers)
        raise ValueError(f"Unknown tier(s): {', '.join(missing)}. Available tiers: {available}")
    return [by_name[name] for name in requested_safe]


def benchmark_argv(
    args: argparse.Namespace,
    tier: dict[str, Any],
    tier_run_id: str,
    seeds: Sequence[str] | None = None,
) -> list[str]:
    seed_values = [str(seed) for seed in (seeds if seeds is not None else args.seeds)]
    argv = [
        "--mode",
        args.mode,
        "--split-protocol",
        args.split_protocol,
        "--datasets",
        *tier["datasets"],
        "--models",
        *args.models,
        "--feature-mode",
        args.feature_mode,
        "--target-mode",
        args.target_mode,
        "--output-scope",
        args.output_scope,
        "--seeds",
        *seed_values,
        "--seq-lens",
        *[str(seq_len) for seq_len in args.seq_lens],
        "--pred-lens",
        *[str(pred_len) for pred_len in args.pred_lens],
        "--max-candidates",
        str(args.max_candidates),
        "--max-raw-rows",
        str(args.max_raw_rows),
        "--max-features",
        str(args.max_features),
        "--search-space-json",
        args.search_space_json,
        "--xgboost-python-package",
        args.xgboost_python_package,
        "--task-workers",
        str(args.task_workers),
        "--n-jobs",
        str(args.n_jobs),
        "--progress-every",
        str(args.progress_every),
        "--output-root",
        args.output_root,
        "--run-id",
        tier_run_id,
    ]
    if not args.include_time_features:
        argv.append("--no-time-features")
    if args.metagb_fixed_score_agg is not None:
        argv.extend(["--metagb-fixed-score-agg", str(args.metagb_fixed_score_agg)])
    return argv


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def refresh_combined_outputs(parent_dir: Path, tier_results: Sequence[dict[str, Any]]) -> None:
    frames: list[pd.DataFrame] = []
    for item in tier_results:
        summary_path = Path(item["results_dir"]) / "summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame.insert(0, "tier", item["tier"])
        frame.insert(1, "tier_run_id", item["run_id"])
        frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    parent_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(parent_dir / "summary.csv", index=False)
    summarize(parent_dir)


def run_tiered(args: argparse.Namespace) -> dict[str, Any]:
    base_run_id = safe_name(args.base_run_id) if args.base_run_id else f"ts_tree_tiered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parent_dir = Path(args.output_root).expanduser().resolve() / "results" / base_run_id
    tiers = select_tiers(load_tier_config(Path(args.tier_config).expanduser().resolve()), args.tiers)
    manifest: dict[str, Any] = {
        "base_run_id": base_run_id,
        "tier_config": str(Path(args.tier_config).expanduser().resolve()),
        "dry_run": bool(args.dry_run),
        "schedule_order": "seed_then_tier",
        "seeds": [str(seed) for seed in args.seeds],
        "tiers_requested": [tier["name"] for tier in tiers],
        "runs": [],
    }
    write_json(parent_dir / "tier_manifest.json", manifest)

    completed: list[dict[str, Any]] = []
    total_runs = len(args.seeds) * len(tiers)
    run_index = 0
    for seed in args.seeds:
        seed_value = str(seed)
        for tier_index, tier in enumerate(tiers, start=1):
            run_index += 1
            tier_run_id = f"{base_run_id}__seed{safe_name(seed_value)}__{tier['name']}"
            argv = benchmark_argv(args, tier, tier_run_id, seeds=[seed_value])
            tier_record = {
                "index": run_index,
                "seed": seed_value,
                "tier_index": tier_index,
                "tier": tier["name"],
                "description": tier["description"],
                "datasets": tier["datasets"],
                "run_id": tier_run_id,
                "argv": argv,
                "status": "planned" if args.dry_run else "running",
            }
            manifest["runs"].append(tier_record)
            write_json(parent_dir / "tier_manifest.json", manifest)
            print(
                f"[run {run_index}/{total_runs}] seed={seed_value} "
                f"tier={tier['name']}: {' '.join(tier['datasets'])}",
                flush=True,
            )
            if args.dry_run:
                print("python3 timeseries_extra_random300/scripts/run_timeseries_benchmark.py " + " ".join(argv), flush=True)
                tier_record["status"] = "dry_run"
                continue

            benchmark_args = normalize_benchmark_args(build_benchmark_parser().parse_args(argv))
            result = run_all(benchmark_args)
            tier_record.update(
                {
                    "status": "ok",
                    "results_dir": result["results_dir"],
                    "summary_rows": result["summary_rows"],
                    "candidate_rows": result["candidate_rows"],
                }
            )
            completed.append({"tier": tier["name"], "run_id": tier_run_id, **result})
            refresh_combined_outputs(parent_dir, completed)
            write_json(parent_dir / "tier_manifest.json", manifest)

    write_json(parent_dir / "tier_manifest.json", manifest)
    return {
        "base_run_id": base_run_id,
        "results_dir": str(parent_dir),
        "tiers": len(tiers),
        "seeds": len(args.seeds),
        "runs": total_runs,
        "completed_runs": len(completed),
        "completed_tiers": len(completed),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run_tiered(args)
    print(f"results_dir: {result['results_dir']}")
    print(f"tiers: {result['tiers']}")
    print(f"seeds: {result['seeds']}")
    print(f"runs: {result['runs']}")
    print(f"completed_runs: {result['completed_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
