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
    echo "ERROR: neither python nor python3 was found. Activate your conda env or set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi
echo "Using PYTHON_BIN=${PYTHON_BIN}"

RUN_ID="${RUN_ID:-smoke_random300_baselinemodel_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode quick \
  --task-types regression \
  --models linear extra_trees \
  --categorical-handlings drop onehot \
  --regression-datasets regression__Moneyball.csv \
  --max-candidates 2 \
  --outer-folds 2 \
  --inner-folds 2 \
  --outer-fold-index 0 \
  --task-workers 1 \
  --n-jobs 1 \
  --progress-every 1 \
  --run-id "${RUN_ID}"
