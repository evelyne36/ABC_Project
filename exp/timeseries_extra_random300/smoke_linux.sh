#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
XGBOOST_PYTHON_PACKAGE="${XGBOOST_PYTHON_PACKAGE:-${REPO_ROOT}/CrossGB/xgboost-master/python-package}"
if [[ -d "${XGBOOST_PYTHON_PACKAGE}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/exp:${XGBOOST_PYTHON_PACKAGE}:${PYTHONPATH:-}"
fi

mkdir -p exp/timeseries_extra_random300/nohup_logs
RUN_ID="${RUN_ID:-smoke_timeseries_extra_linux_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" exp/timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode smoke \
  --datasets ILI \
  --models xgb_cpu crossgb \
  --outer-folds 2 \
  --inner-folds 2 \
  --outer-fold-index 0 \
  --max-candidates 30 \
  --seq-len 12 \
  --pred-len 1 \
  --max-raw-rows 180 \
  --search-space-json exp/timeseries_extra_random300/configs/smoke_search_space.json \
  --task-workers 1 \
  --n-jobs 1 \
  --progress-every 10 \
  --run-id "${RUN_ID}"
