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

HANDLING="${HANDLING:-ordinal}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-300}"
MAX_ROWS_PER_DATASET="${MAX_ROWS_PER_DATASET:-0}"
RUN_ID="${RUN_ID:-tabular_comprehensive_${HANDLING}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-tabular_extra_random300/nohup_logs/${RUN_ID}.log}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p tabular_extra_random300/nohup_logs

CMD=(
  "${PYTHON_BIN}" tabular_extra_random300/scripts/exp1_random300_all.py
  --mode full \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb \
  --regression-datasets all \
  --binary-datasets all \
  --multiclass-datasets all \
  --categorical-handling "${HANDLING}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers "${TASK_WORKERS}" \
  --n-jobs "${N_JOBS}" \
  --progress-every 50 \
  --max-rows-per-dataset "${MAX_ROWS_PER_DATASET}" \
  --run-id "${RUN_ID}"
)

echo "run_id=${RUN_ID}"
echo "handling=${HANDLING}"
echo "log=${REPO_ROOT}/${LOG_FILE}"
echo "results=${REPO_ROOT}/tabular_extra_random300/results/${RUN_ID}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "pid=${PID}"
