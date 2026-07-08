#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3.13}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

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

