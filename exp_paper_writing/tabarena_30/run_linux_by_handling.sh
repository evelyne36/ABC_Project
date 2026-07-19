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

TASK_TYPE="${TASK_TYPE:-binary}"
HANDLING="${HANDLING:-drop}"
MODELS="${MODELS:-current_metagb}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-300}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
DATASET_LIST="${DATASET_LIST:-tabarena_30/configs/${TASK_TYPE}_${HANDLING}_datasets.txt}"
RUN_ID="${RUN_ID:-tabarena30_${TASK_TYPE}_${HANDLING}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-tabarena_30/nohup_logs/${RUN_ID}.log}"
DRY_RUN="${DRY_RUN:-0}"

case "${TASK_TYPE}" in
  binary) DATASET_FLAG="--binary-datasets" ;;
  multiclass) DATASET_FLAG="--multiclass-datasets" ;;
  regression) DATASET_FLAG="--regression-datasets" ;;
  *) echo "Unsupported TASK_TYPE=${TASK_TYPE}" >&2; exit 2 ;;
esac

if [[ "${TASK_TYPE}" == "regression" && "${HANDLING}" == "onehot" ]]; then
  echo "Skip regression onehot: remaining Food_Delivery_Time.csv is intentionally abandoned as too slow." >&2
  exit 0
fi

if [[ ! -f "${DATASET_LIST}" ]]; then
  echo "Dataset list not found: ${DATASET_LIST}" >&2
  exit 2
fi

DATASETS=()
while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
  DATASETS+=("${line}")
done < "${DATASET_LIST}"
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "Dataset list is empty: ${DATASET_LIST}" >&2
  exit 2
fi

mkdir -p tabarena_30/nohup_logs

CMD=(
  "${PYTHON_BIN}" tabarena_30/scripts/exp1_tabarena30_all.py
  --mode full
  --task-types "${TASK_TYPE}"
  --models "${MODELS}"
  --categorical-handling "${HANDLING}"
  --max-candidates "${MAX_CANDIDATES}"
  --outer-folds "${OUTER_FOLDS}"
  --inner-folds "${INNER_FOLDS}"
  --task-workers "${TASK_WORKERS}"
  --n-jobs "${N_JOBS}"
  --progress-every "${PROGRESS_EVERY}"
  --run-id "${RUN_ID}"
  "${DATASET_FLAG}" "${DATASETS[@]}"
)

echo "run_id=${RUN_ID}"
echo "task_type=${TASK_TYPE}"
echo "handling=${HANDLING}"
echo "dataset_list=${REPO_ROOT}/${DATASET_LIST}"
printf 'datasets=%s\n' "${DATASETS[*]}"
echo "log=${REPO_ROOT}/${LOG_FILE}"
echo "results=${REPO_ROOT}/tabarena_30/results/${RUN_ID}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID="$!"
echo "pid=${PID}"
