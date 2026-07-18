#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: Python was not found. Activate the environment or set PYTHON_BIN." >&2
    exit 1
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/tmp_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

RUN_ID="${RUN_ID:-synthetic_baseline_random300_seed42}"
MODEL_NAMES="${MODEL_NAMES:-lightgbm catboost random_forest}"
read -r -a MODEL_ARGS <<< "${MODEL_NAMES}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${PROJECT_DIR}/logs/${RUN_ID}.pid}"
mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"

echo "Using PYTHON_BIN=${PYTHON_BIN}"
echo "RUN_ID=${RUN_ID}"
echo "MODELS=${MODEL_NAMES}"
echo "LOG_FILE=${LOG_FILE}"
echo "PID_FILE=${PID_FILE}"

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  EXISTING_COMMAND="$(
    ps -p "${EXISTING_PID}" -o command= 2>/dev/null || true
  )"
  if [[ "${EXISTING_COMMAND}" == *"synthetic_baselinemodel/run_benchmark.py"* ]] \
    && [[ "${EXISTING_COMMAND}" == *"--run-id ${RUN_ID}"* ]]; then
    echo "ERROR: run already appears active with PID=${EXISTING_PID}" >&2
    echo "Monitor: tail -f ${LOG_FILE}" >&2
    exit 1
  fi
fi

{
  echo
  echo "===== launch $(date '+%Y-%m-%d %H:%M:%S') run_id=${RUN_ID} ====="
} >> "${LOG_FILE}"

nohup "${PYTHON_BIN}" synthetic_baselinemodel/run_benchmark.py \
  --protocol synthetic_baseline_random300_single_seed \
  --models "${MODEL_ARGS[@]}" \
  --search-seed 42 \
  --data-seeds 42 \
  --task-workers "${TASK_WORKERS:-15}" \
  --n-jobs "${N_JOBS:-1}" \
  --run-id "${RUN_ID}" \
  "$@" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

RUN_PID=$!
echo "${RUN_PID}" > "${PID_FILE}"

echo "Started detached run with PID=${RUN_PID}"
echo "You may now disconnect from SSH."
echo "Monitor: tail -f ${LOG_FILE}"
echo "Task count: find ${PROJECT_DIR}/results/${RUN_ID}/tasks -name '*.json' | wc -l"
echo "Resume after interruption: bash synthetic_baselinemodel/run_linux_full.sh --resume"
