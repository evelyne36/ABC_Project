#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
XGBOOST_PYTHON_PACKAGE="${XGBOOST_PYTHON_PACKAGE:-${REPO_ROOT}/beforereview/xgboost-master/python-package}"
if [[ -d "${XGBOOST_PYTHON_PACKAGE}" ]]; then
  export PYTHONPATH="${XGBOOST_PYTHON_PACKAGE}:${PYTHONPATH:-}"
fi

mkdir -p tabular_extra_random300/nohup_logs
RUN_ID="${RUN_ID:-smoke_tabular_extra_linux_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" tabular_extra_random300/scripts/exp1_random300_all.py \
  --mode quick \
  --task-types regression \
  --models xgb_cpu current_metagb \
  --regression-datasets regression__energy_efficiency.csv \
  --max-candidates 30 \
  --outer-folds 2 \
  --inner-folds 2 \
  --outer-fold-index 0 \
  --task-workers 1 \
  --n-jobs 1 \
  --progress-every 10 \
  --run-id "${RUN_ID}"
