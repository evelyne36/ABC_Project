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

BASE_RUN_ID="${BASE_RUN_ID:-ts_tree_tiered_$(date +%Y%m%d_%H%M%S)}"
TIERS="${TIERS:-}"
MODELS="${MODELS:-crossgb xgb_cpu lgbm_cpu catboost_cpu}"
SEEDS="${SEEDS:-42}"
SEQ_LENS="${SEQ_LENS:-24 48}"
PRED_LENS="${PRED_LENS:-1 6 12}"
MAX_CANDIDATES="${MAX_CANDIDATES:-300}"
MAX_RAW_ROWS="${MAX_RAW_ROWS:-0}"
MAX_FEATURES="${MAX_FEATURES:-64}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SEARCH_SPACE_JSON="${SEARCH_SPACE_JSON:-exp/timeseries_extra_random300/configs/timeseries_search_space.json}"
LOG_FILE="${LOG_FILE:-exp/timeseries_extra_random300/nohup_logs/${BASE_RUN_ID}.log}"

mkdir -p exp/timeseries_extra_random300/nohup_logs

TIER_ARGS=()
if [[ -n "${TIERS}" ]]; then
  TIER_ARGS=(--tiers ${TIERS})
fi

nohup "${PYTHON_BIN}" exp/timeseries_extra_random300/scripts/run_tiered_timeseries_benchmark.py \
  --base-run-id "${BASE_RUN_ID}" \
  "${TIER_ARGS[@]}" \
  --mode full \
  --split-protocol fixed \
  --models ${MODELS} \
  --feature-mode window_summary \
  --target-mode sequence \
  --output-scope target \
  --seeds ${SEEDS} \
  --seq-lens ${SEQ_LENS} \
  --pred-lens ${PRED_LENS} \
  --max-candidates "${MAX_CANDIDATES}" \
  --max-raw-rows "${MAX_RAW_ROWS}" \
  --max-features "${MAX_FEATURES}" \
  --search-space-json "${SEARCH_SPACE_JSON}" \
  --xgboost-python-package "${XGBOOST_PYTHON_PACKAGE}" \
  --task-workers "${TASK_WORKERS}" \
  --n-jobs "${N_JOBS}" \
  --progress-every "${PROGRESS_EVERY}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "base_run_id=${BASE_RUN_ID}"
echo "pid=${PID}"
echo "log=${REPO_ROOT}/${LOG_FILE}"
echo "combined_results=${REPO_ROOT}/exp/timeseries_extra_random300/results/${BASE_RUN_ID}"
