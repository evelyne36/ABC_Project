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

mkdir -p tabarena_30/nohup_logs
RUN_ID="${RUN_ID:-smoke_tabarena30_linux_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" tabarena_30/scripts/exp1_tabarena30_all.py \
  --mode quick \
  --task-types binary \
  --models current_metagb \
  --categorical-handling drop \
  --binary-datasets blood-transfusion-service-center.csv \
  --max-candidates 300 \
  --outer-folds 2 \
  --inner-folds 2 \
  --outer-fold-index 0 \
  --task-workers 1 \
  --n-jobs 1 \
  --progress-every 10 \
  --run-id "${RUN_ID}"
