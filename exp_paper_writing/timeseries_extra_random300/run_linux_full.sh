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

DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate Weather ILI}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
SEEDS="${SEEDS:-42}"
SEQ_LENS="${SEQ_LENS:-48}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
TARGET_MODE="${TARGET_MODE:-sequence}"
OUTPUT_SCOPE="${OUTPUT_SCOPE:-all}"
RUN_ID="${RUN_ID:-timeseries_extra_random300_full_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-timeseries_extra_random300/nohup_logs/${RUN_ID}.log}"

mkdir -p timeseries_extra_random300/nohup_logs

nohup "${PYTHON_BIN}" timeseries_extra_random300/scripts/run_timeseries_benchmark.py \
  --mode full \
  --split-protocol fixed \
  --target-mode "${TARGET_MODE}" \
  --output-scope "${OUTPUT_SCOPE}" \
  --datasets ${DATASETS} \
  --models xgb_cpu current_metagb \
  --max-candidates 300 \
  --seeds ${SEEDS} \
  --seq-lens ${SEQ_LENS} \
  --pred-lens ${PRED_LENS} \
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
