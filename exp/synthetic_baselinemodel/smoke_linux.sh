#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/tmp_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

RUN_ID="${RUN_ID:-synthetic_baseline_smoke_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" exp/synthetic_baselinemodel/run_benchmark.py \
  --protocol synthetic_baseline_framework_smoke \
  --scenario-limit 1 \
  --max-candidates 2 \
  --task-workers 1 \
  --n-jobs 1 \
  --run-id "${RUN_ID}" \
  "$@"
