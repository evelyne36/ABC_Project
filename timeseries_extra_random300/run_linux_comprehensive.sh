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

# Pressure order uses rows_or_timesteps * (seq_len * channels)^2 from execution_order_by_pressure.csv.
DATASETS="${DATASETS:-ILI exchange_rate ETTh1 ETTh2 solar ETTm1 ETTm2 Weather PEMS08 PEMS04 PEMS03 electricity PEMS07 traffic}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
SEQ_LEN="${SEQ_LEN:-24}"
PRED_LEN="${PRED_LEN:-1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-300}"
MAX_RAW_ROWS="${MAX_RAW_ROWS:-0}"
MAX_FEATURES="${MAX_FEATURES:-0}"
RUN_ID="${RUN_ID:-timeseries_comprehensive_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-timeseries_extra_random300/nohup_logs/${RUN_ID}.log}"

mkdir -p timeseries_extra_random300/nohup_logs

nohup "${PYTHON_BIN}" timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full \
  --datasets ${DATASETS} \
  --models xgb_cpu current_metagb \
  --outer-folds 5 \
  --inner-folds 4 \
  --max-candidates "${MAX_CANDIDATES}" \
  --seq-len "${SEQ_LEN}" \
  --pred-len "${PRED_LEN}" \
  --max-raw-rows "${MAX_RAW_ROWS}" \
  --max-features "${MAX_FEATURES}" \
  --task-workers "${TASK_WORKERS}" \
  --n-jobs "${N_JOBS}" \
  --progress-every 50 \
  --run-id "${RUN_ID}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "run_id=${RUN_ID}"
echo "pid=${PID}"
echo "log=${REPO_ROOT}/${LOG_FILE}"
echo "results=${REPO_ROOT}/timeseries_extra_random300/results/${RUN_ID}"
