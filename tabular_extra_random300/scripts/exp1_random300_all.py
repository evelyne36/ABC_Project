from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabular_extra_random300.utils.runner import run_all, safe_name  # noqa: E402


def default_cpu_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def default_tabpfn_dir() -> Path:
    return ROOT / "tabular_extra_random300" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random300 TabPFN benchmark for xgb_cpu and current_metagb across regression/binary/multiclass tasks."
    )
    tabpfn_dir = default_tabpfn_dir()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--task-types", nargs="+", choices=["regression", "binary", "multiclass"], default=["regression", "binary", "multiclass"])
    parser.add_argument("--models", nargs="+", choices=["xgb_cpu", "current_metagb", "current_metagb_pred0"], default=["xgb_cpu", "current_metagb"])
    parser.add_argument("--tabpfn-dir", default=str(tabpfn_dir), help="Directory containing staged tabular extension CSV files.")
    parser.add_argument("--regression-dir", default=None)
    parser.add_argument("--binary-dir", default=None)
    parser.add_argument("--multiclass-dir", default=None)
    parser.add_argument("--regression-datasets", nargs="+", default=["all"])
    parser.add_argument("--binary-datasets", nargs="+", default=["all"])
    parser.add_argument("--multiclass-datasets", nargs="+", default=["all"])
    parser.add_argument("--skip-datasets", nargs="*", default=[], help="Dataset filenames to skip for all task types.")
    parser.add_argument("--skip-regression-datasets", nargs="*", default=[], help="Regression dataset filenames to skip.")
    parser.add_argument("--skip-binary-datasets", nargs="*", default=[], help="Binary dataset filenames to skip.")
    parser.add_argument("--skip-multiclass-datasets", nargs="*", default=[], help="Multiclass dataset filenames to skip.")
    parser.add_argument("--xgboost-python-package", default=str(ROOT / "beforereview" / "xgboost-master" / "python-package"))
    parser.add_argument("--xgb-config", default=None, help="Deprecated compatibility option; ignored by paired candidate generation.")
    parser.add_argument("--search-space-json", default=str(ROOT / "tabular_extra_random300" / "configs" / "default_search_space.json"))
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--outer-fold-index", type=int, default=0)
    parser.add_argument("--run-all-outer-folds", action="store_true")
    parser.add_argument("--categorical-handling", choices=["drop", "onehot", "ordinal"], default="ordinal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-workers", type=int, default=default_cpu_jobs())
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-rows-per-dataset", type=int, default=0, help="Optional deterministic row cap for lightweight smoke runs.")
    parser.add_argument("--output-root", default=str(ROOT / "tabular_extra_random300"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--binary-objective", choices=["binary:logistic", "binary:logitraw"], default="binary:logistic")
    parser.add_argument("--multiclass-objective", choices=["multi:softprob", "multi:softmax"], default="multi:softprob")
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--metagb-lambda-l2", type=float, default=1.0)
    parser.add_argument("--metagb-score-agg", type=int, default=0, choices=[0, 1, 2], help="Fallback for candidates without metagb_score_agg.")
    parser.add_argument("--metagb-pred-value-type", choices=["bin", "raw", "standardized_raw"], default="raw")
    parser.add_argument("--metagb-hist-subtract", dest="metagb_hist_subtract", action="store_true", default=False)
    parser.add_argument("--no-metagb-hist-subtract", dest="metagb_hist_subtract", action="store_false")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least 2")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2")
    if args.mode == "full":
        args.run_all_outer_folds = True
    if args.max_candidates <= 0:
        args.max_candidates = 30 if args.mode == "quick" else 300
    if args.max_candidates % 30 != 0 and "current_metagb" in args.models:
        raise ValueError("--max-candidates must be divisible by 30 when current_metagb is enabled")
    if args.progress_every <= 0:
        args.progress_every = max(1, args.max_candidates)
    if args.n_jobs != 1 and args.task_workers > 1:
        print("WARNING: --n-jobs > 1 with --task-workers > 1 can oversubscribe CPU.", flush=True)

    tabpfn_dir = Path(args.tabpfn_dir).expanduser().resolve()
    if args.regression_dir is None:
        args.regression_dir = str(tabpfn_dir)
    if args.binary_dir is None:
        args.binary_dir = str(tabpfn_dir)
    if args.multiclass_dir is None:
        args.multiclass_dir = str(tabpfn_dir)

    output_root = Path(args.output_root).expanduser().resolve()
    run_id = safe_name(args.run_id) if args.run_id else f"tabular_extra_random300_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.run_id = run_id
    args.results_dir = str(output_root / "results" / run_id)
    args.logs_dir = str(output_root / "logs" / run_id)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args())
    metadata = run_all(args)
    print(f"results_dir: {metadata['results_dir']}")
    print(f"logs_dir: {metadata['logs_dir']}")
    print(f"summary_rows: {metadata['summary_rows']}")
    print(f"candidate_rows: {metadata['candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
