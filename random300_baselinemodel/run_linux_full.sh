#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3.13}"
RUN_ID="${RUN_ID:-random300_baselinemodel_full_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" random300_baselinemodel/scripts/exp1_random300_all.py \
  --mode full \
  --task-types regression binary multiclass \
  --models linear extra_trees random_forest lightgbm catboost torchmlp realmlp tabm \
  --categorical-handlings drop onehot \
  --max-candidates "${MAX_CANDIDATES:-300}" \
  --outer-folds 5 \
  --inner-folds 4 \
  --task-workers "${TASK_WORKERS:-8}" \
  --n-jobs "${N_JOBS:-1}" \
  --progress-every "${PROGRESS_EVERY:-5}" \
  --run-id "${RUN_ID}"
