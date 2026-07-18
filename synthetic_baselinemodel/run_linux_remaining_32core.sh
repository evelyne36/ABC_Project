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

RUN_ID="${RUN_ID:-synthetic_catboost_randomforest_random300_seed42}"
RESULTS_DIR="${PROJECT_DIR}/results/${RUN_ID}"

export TASK_WORKERS="${TASK_WORKERS:-30}"
export N_JOBS=1
export RUN_ID
export MODEL_NAMES="catboost random_forest"
export LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/${RUN_ID}_32core.log}"
export PID_FILE="${PID_FILE:-${PROJECT_DIR}/logs/${RUN_ID}_32core.pid}"

echo "Launching with TASK_WORKERS=${TASK_WORKERS}, N_JOBS=${N_JOBS}"
echo "This standalone run contains only: ${MODEL_NAMES}"

if [[ -f "${RESULTS_DIR}/protocol.json" ]]; then
  echo "Existing CatBoost/RF run detected; enabling --resume."
  bash synthetic_baselinemodel/run_linux_full.sh --resume "$@"
else
  echo "No existing CatBoost/RF protocol found; starting a new two-model run."
  bash synthetic_baselinemodel/run_linux_full.sh "$@"
fi
