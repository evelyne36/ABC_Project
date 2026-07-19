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

HANDLING="${HANDLING:-drop}"
TASK_TYPES="${TASK_TYPES:-binary multiclass}"
MODELS="${MODELS:-current_metagb}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
RUN_ID="${RUN_ID:-tabarena30_${HANDLING}_full_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-tabarena_30/nohup_logs/${RUN_ID}.log}"

mkdir -p tabarena_30/nohup_logs

nohup "${PYTHON_BIN}" tabarena_30/scripts/exp1_tabarena30_all.py \
  --mode full \
  --task-types ${TASK_TYPES} \
  --models ${MODELS} \
  --categorical-handling "${HANDLING}" \
  --max-candidates 300 \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers "${TASK_WORKERS}" \
  --n-jobs "${N_JOBS}" \
  --progress-every 10 \
  --run-id "${RUN_ID}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "run_id=${RUN_ID}"
echo "pid=${PID}"
echo "log=${REPO_ROOT}/${LOG_FILE}"
echo "results=${REPO_ROOT}/tabarena_30/results/${RUN_ID}"
