#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabular_extra_random300.scripts.run_random300_server import main as server_main  # noqa: E402


def add_default(args: list[str], flag: str, values: list[str]) -> None:
    if flag not in args:
        args.extend([flag, *values])


if __name__ == "__main__":
    forwarded = sys.argv[1:]
    defaulted = list(forwarded)
    add_default(defaulted, "--experiment-name", ["tabular_extra_random300_main_xgb_vs_current_metagb"])
    add_default(defaulted, "--mode", ["full"])
    add_default(defaulted, "--task-types", ["regression", "binary", "multiclass"])
    add_default(defaulted, "--models", ["xgb_cpu", "current_metagb"])
    add_default(defaulted, "--max-candidates", ["300"])
    add_default(defaulted, "--outer-folds", ["5"])
    add_default(defaulted, "--inner-folds", ["4"])
    add_default(defaulted, "--task-workers", ["16"])
    add_default(defaulted, "--n-jobs", ["1"])
    sys.argv = [sys.argv[0], *defaulted]
    raise SystemExit(server_main())
