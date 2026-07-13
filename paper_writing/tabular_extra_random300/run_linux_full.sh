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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

HANDLING="${HANDLING:-ordinal}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
RUN_ID="${RUN_ID:-tabular_extra_random300_${HANDLING}_full_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-tabular_extra_random300/nohup_logs/${RUN_ID}.log}"

mkdir -p tabular_extra_random300/nohup_logs

nohup "${PYTHON_BIN}" tabular_extra_random300/scripts/exp1_random300_all.py \
  --mode full \
  --task-types regression binary multiclass \
  --models xgb_cpu current_metagb \
  --regression-datasets regression__energy_efficiency.csv regression__space_ga.csv regression__kin8nm.csv \
  --binary-datasets binary__credit-approval.csv binary__kc1.csv binary__phoneme.csv \
  --multiclass-datasets multiclass__vehicle.csv multiclass__segment.csv multiclass__balance-scale.csv \
  --categorical-handling "${HANDLING}" \
  --max-candidates 300 \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers "${TASK_WORKERS}" \
  --n-jobs "${N_JOBS}" \
  --progress-every 50 \
  --run-id "${RUN_ID}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "run_id=${RUN_ID}"
echo "pid=${PID}"
echo "log=${REPO_ROOT}/${LOG_FILE}"
echo "results=${REPO_ROOT}/tabular_extra_random300/results/${RUN_ID}"
