#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ACTION="${1:-help}"
SCOPE="${2:-synthetic}"

RUN_TAG="${RUN_TAG:-v1}"
TASK_WORKERS="${TASK_WORKERS:-15}"
N_JOBS="${N_JOBS:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
CATEGORICAL_HANDLING="${CATEGORICAL_HANDLING:-ordinal}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/metagb_ablation/results}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/metagb_ablation/logs}"
XGB_PYTHON_PACKAGE="${XGB_PYTHON_PACKAGE:-${PROJECT_ROOT}/beforereview/xgboost-master/python-package}"

MASTER_LOG="${LOG_DIR}/queue_${SCOPE}_${RUN_TAG}.log"
PID_FILE="${LOG_DIR}/queue_${SCOPE}_${RUN_TAG}.pid"

REAL_DATASETS=(
  "diabetes"
  "anneal"
  "credit-g"
  "maternal_health_risk"
  "airfoil_self_noise"
  "seismic-bumps"
  "miami_housing"
  "physiochemical_protein"
)

usage() {
  cat <<'EOF'
Usage:
  bash metagb_ablation/scripts/launch_ablation.sh start synthetic
  bash metagb_ablation/scripts/launch_ablation.sh start real
  bash metagb_ablation/scripts/launch_ablation.sh status [synthetic|real]
  bash metagb_ablation/scripts/launch_ablation.sh tail [synthetic|real]
  bash metagb_ablation/scripts/launch_ablation.sh approve-synthetic
  bash metagb_ablation/scripts/launch_ablation.sh check

Scopes:
  synthetic  Run only the formal synthetic experiment.
  real       Run real datasets sequentially only after the analysis gate passes.

Required two-stage workflow:
  1. start synthetic
  2. Analyze outputs and write:
       results/syn_formal_<RUN_TAG>/SYNTHETIC_CONCLUSION.md
  3. approve-synthetic
  4. start real

Optional environment variables:
  RUN_TAG=v1                    Stable suffix used by run IDs; keep it for resume.
  TASK_WORKERS=15               Parallel independent tasks.
  N_JOBS=1                     Threads inside each estimator.
  OUTPUT_ROOT=/data/results     Result root.
  LOG_DIR=/data/logs            Queue and stage logs.
  PYTHON_BIN=python3            Python executable.
  RUN_PREFLIGHT=1               Run tests before the queue; set 0 to skip.
  CATEGORICAL_HANDLING=ordinal  drop, onehot, or ordinal.
  Formal launcher variant count is fixed at 16.

The launcher always passes --resume. A new run starts normally; an interrupted
run resumes only missing task shards.
EOF
}

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

validate_scope() {
  case "$SCOPE" in
    synthetic|real) ;;
    *)
      echo "Invalid scope: ${SCOPE}" >&2
      usage
      exit 2
      ;;
  esac
}

synthetic_result_dir() {
  printf "%s/syn_formal_%s" "$OUTPUT_ROOT" "$RUN_TAG"
}

synthetic_conclusion_path() {
  printf "%s/SYNTHETIC_CONCLUSION.md" "$(synthetic_result_dir)"
}

synthetic_approval_path() {
  printf "%s/synthetic_analysis_approval.json" "$(synthetic_result_dir)"
}

pid_is_active() {
  local file="$1"
  local pid
  local command
  [[ -f "$file" ]] || return 1
  pid="$(tr -d '[:space:]' < "$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == *"launch_ablation.sh"* ]]
}

ensure_no_conflicting_queue() {
  local file
  local pid
  shopt -s nullglob
  for file in "${LOG_DIR}"/queue_*_"${RUN_TAG}".pid; do
    if pid_is_active "$file"; then
      pid="$(tr -d '[:space:]' < "$file")"
      echo "A queue with RUN_TAG=${RUN_TAG} is already active: pid=${pid}" >&2
      echo "PID file: ${file}" >&2
      exit 1
    fi
  done
  shopt -u nullglob
}

print_configuration() {
  cat <<EOF
project_root=${PROJECT_ROOT}
scope=${SCOPE}
run_tag=${RUN_TAG}
task_workers=${TASK_WORKERS}
n_jobs=${N_JOBS}
categorical_handling=${CATEGORICAL_HANDLING}
variant_count=16
output_root=${OUTPUT_ROOT}
log_dir=${LOG_DIR}
xgboost_python_package=${XGB_PYTHON_PACKAGE}
EOF
}

run_preflight() {
  echo "[$(timestamp)] preflight started"
  "$PYTHON_BIN" -c \
    "import xgboost; print('xgboost_version=', xgboost.__version__); print('xgboost_path=', xgboost.__file__)"
  "$PYTHON_BIN" -m pytest -q "${PROJECT_ROOT}/metagb_ablation/tests"
  "$PYTHON_BIN" -m pytest -q \
    "${PROJECT_ROOT}/beforereview/xgboost-master/tests/python/test_metagb.py" \
    -k "pred_fraction or leaf_feature or bias"
  echo "[$(timestamp)] preflight passed"
}

audit_result() {
  local run_id="$1"
  local result_dir="${OUTPUT_ROOT}/${run_id}"
  "$PYTHON_BIN" - "$result_dir" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
audit_path = result_dir / "completeness_audit.json"
if not audit_path.exists():
    raise SystemExit(f"Missing audit: {audit_path}")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
print(
    "audit",
    f"run_id={result_dir.name}",
    f"passed={audit.get('passed')}",
    f"tasks={audit.get('actual_tasks')}/{audit.get('expected_tasks')}",
    f"fold_rows={audit.get('actual_fold_rows')}/{audit.get('expected_fold_rows')}",
    f"predictions={audit.get('actual_predictions')}/{audit.get('expected_predictions')}",
)
if not audit.get("passed"):
    print("issues=", audit.get("issues"))
    raise SystemExit(2)
PY
}

run_stage() {
  local run_id="$1"
  shift
  local stage_log="${LOG_DIR}/${run_id}.log"
  local command_status
  local -a command=(
    "$PYTHON_BIN"
    -m
    metagb_ablation.run_ablation
    "$@"
    --task-workers
    "$TASK_WORKERS"
    --n-jobs
    "$N_JOBS"
    --xgboost-python-package
    "$XGB_PYTHON_PACKAGE"
    --output-root
    "$OUTPUT_ROOT"
    --run-id
    "$run_id"
    --resume
  )
  command+=(--include-split-scoreagg)

  echo "[$(timestamp)] stage started: ${run_id}"
  echo "stage_log=${stage_log}"
  printf "command="
  printf " %q" "${command[@]}"
  printf "\n"

  set +e
  "${command[@]}" 2>&1 | tee -a "$stage_log"
  command_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$command_status" -ne 0 ]]; then
    echo "[$(timestamp)] stage failed: ${run_id}, exit=${command_status}" >&2
    return "$command_status"
  fi

  audit_result "$run_id"
  echo "[$(timestamp)] stage completed: ${run_id}"
}

run_synthetic() {
  run_stage \
    "syn_formal_${RUN_TAG}" \
    --protocol formal \
    --domains synthetic
}

run_real_queue() {
  local dataset
  local run_id
  for dataset in "${REAL_DATASETS[@]}"; do
    run_id="real_formal_${dataset}_${RUN_TAG}"
    run_stage \
      "$run_id" \
      --protocol formal \
      --domains real \
      --real-datasets "$dataset" \
      --categorical-handling "$CATEGORICAL_HANDLING"
  done
}

approve_synthetic_analysis() {
  local result_dir
  local conclusion_path
  local approval_path
  result_dir="$(synthetic_result_dir)"
  conclusion_path="$(synthetic_conclusion_path)"
  approval_path="$(synthetic_approval_path)"
  "$PYTHON_BIN" - "$result_dir" "$conclusion_path" "$approval_path" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

result_dir = Path(sys.argv[1]).resolve()
conclusion_path = Path(sys.argv[2]).resolve()
approval_path = Path(sys.argv[3]).resolve()
audit_path = result_dir / "completeness_audit.json"
protocol_path = result_dir / "protocol.json"
execution_path = result_dir / "execution_manifest.json"

for path in (audit_path, protocol_path, execution_path):
    if not path.exists():
        raise SystemExit(f"Missing required Synthetic artifact: {path}")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
if not audit.get("passed"):
    raise SystemExit(f"Synthetic audit has not passed: {audit.get('issues')}")
if not conclusion_path.exists() or conclusion_path.stat().st_size == 0:
    raise SystemExit(
        "Synthetic conclusion is missing or empty. Analyze the results and write: "
        f"{conclusion_path}"
    )

protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
execution = json.loads(execution_path.read_text(encoding="utf-8"))
if protocol.get("variant_count") != 16:
    raise SystemExit(
        "Synthetic analysis approval requires the formal 16-variant protocol; "
        f"found variant_count={protocol.get('variant_count')}"
    )
conclusion_text = conclusion_path.read_text(encoding="utf-8")
required_markers = (
    "SYNTHETIC_ANALYSIS_STATUS: COMPLETE",
    "REAL_STAGE_DECISION: GO",
)
missing_markers = [
    marker for marker in required_markers if marker not in conclusion_text
]
if missing_markers:
    raise SystemExit(
        "Synthetic conclusion is not final. Missing markers: "
        f"{missing_markers}"
    )
unfinished_tokens = ("TBD", "TODO", "待填写", "<填写")
present_unfinished = [
    token for token in unfinished_tokens if token in conclusion_text
]
if present_unfinished:
    raise SystemExit(
        "Synthetic conclusion still contains unfinished placeholders: "
        f"{present_unfinished}"
    )
conclusion_hash = hashlib.sha256(conclusion_path.read_bytes()).hexdigest()
payload = {
    "approved_at_utc": datetime.now(timezone.utc).isoformat(),
    "synthetic_run_id": result_dir.name,
    "config_hash": protocol["config_hash"],
    "execution_hash": execution["execution_hash"],
    "variant_hash": protocol["variant_hash"],
    "variant_count": protocol["variant_count"],
    "conclusion_path": str(conclusion_path),
    "conclusion_sha256": conclusion_hash,
    "audit_passed": True,
}
temporary = approval_path.with_suffix(approval_path.suffix + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
temporary.replace(approval_path)
print(f"Synthetic analysis approved: {approval_path}")
print(f"conclusion_sha256={conclusion_hash}")
PY
}

ensure_synthetic_analysis_gate() {
  local result_dir
  local conclusion_path
  local approval_path
  result_dir="$(synthetic_result_dir)"
  conclusion_path="$(synthetic_conclusion_path)"
  approval_path="$(synthetic_approval_path)"
  "$PYTHON_BIN" - "$result_dir" "$conclusion_path" "$approval_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1]).resolve()
conclusion_path = Path(sys.argv[2]).resolve()
approval_path = Path(sys.argv[3]).resolve()
audit_path = result_dir / "completeness_audit.json"
protocol_path = result_dir / "protocol.json"
execution_path = result_dir / "execution_manifest.json"

for path in (
    audit_path,
    protocol_path,
    execution_path,
    conclusion_path,
    approval_path,
):
    if not path.exists():
        raise SystemExit(
            "Real execution is blocked until Synthetic analysis is complete. "
            f"Missing: {path}"
        )

audit = json.loads(audit_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
execution = json.loads(execution_path.read_text(encoding="utf-8"))
approval = json.loads(approval_path.read_text(encoding="utf-8"))
actual_conclusion_hash = hashlib.sha256(conclusion_path.read_bytes()).hexdigest()

checks = {
    "audit_passed": audit.get("passed") is True,
    "formal_16_variant_synthetic": protocol.get("variant_count") == 16,
    "config_hash": approval.get("config_hash") == protocol.get("config_hash"),
    "variant_hash": approval.get("variant_hash") == protocol.get("variant_hash"),
    "variant_count": approval.get("variant_count") == 16,
    "execution_hash": (
        approval.get("execution_hash") == execution.get("execution_hash")
    ),
    "conclusion_sha256": (
        approval.get("conclusion_sha256") == actual_conclusion_hash
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Synthetic analysis gate failed: {failed}")
print(
    "Synthetic analysis gate passed:",
    f"run_id={result_dir.name}",
    f"approval={approval_path}",
)
PY
}

run_queue() {
  local exit_code
  finish_queue() {
    exit_code=$?
    trap - EXIT
    if [[ "$exit_code" -eq 0 ]]; then
      echo "[$(timestamp)] queue completed successfully: scope=${SCOPE}"
    else
      echo "[$(timestamp)] queue stopped with exit=${exit_code}: scope=${SCOPE}" >&2
    fi
    exit "$exit_code"
  }
  trap finish_queue EXIT

  cd "$PROJECT_ROOT"
  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
  export PYTHONPATH="${XGB_PYTHON_PACKAGE}${PYTHONPATH:+:${PYTHONPATH}}"

  echo "[$(timestamp)] queue started"
  print_configuration

  if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    run_preflight
  else
    echo "[$(timestamp)] preflight skipped"
  fi

  case "$SCOPE" in
    synthetic)
      run_synthetic
      ;;
    real)
      ensure_synthetic_analysis_gate
      run_real_queue
      ;;
  esac
}

start_queue() {
  local pid
  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
  if [[ "$SCOPE" == "real" ]]; then
    ensure_synthetic_analysis_gate
  fi
  ensure_no_conflicting_queue
  touch "$MASTER_LOG"

  nohup env \
    RUN_TAG="$RUN_TAG" \
    TASK_WORKERS="$TASK_WORKERS" \
    N_JOBS="$N_JOBS" \
    RUN_PREFLIGHT="$RUN_PREFLIGHT" \
    CATEGORICAL_HANDLING="$CATEGORICAL_HANDLING" \
    PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    LOG_DIR="$LOG_DIR" \
    XGB_PYTHON_PACKAGE="$XGB_PYTHON_PACKAGE" \
    bash "$SCRIPT_PATH" _run "$SCOPE" \
    >> "$MASTER_LOG" 2>&1 &
  pid=$!
  printf "%s\n" "$pid" > "$PID_FILE"

  echo "Queue started."
  echo "pid=${pid}"
  echo "scope=${SCOPE}"
  echo "master_log=${MASTER_LOG}"
  echo "pid_file=${PID_FILE}"
  echo
  echo "Monitor:"
  echo "  bash metagb_ablation/scripts/launch_ablation.sh status ${SCOPE}"
  echo "  bash metagb_ablation/scripts/launch_ablation.sh tail ${SCOPE}"
}

show_status() {
  local pid=""
  mkdir -p "$LOG_DIR"
  if pid_is_active "$PID_FILE"; then
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    echo "status=running pid=${pid} scope=${SCOPE} run_tag=${RUN_TAG}"
  else
    echo "status=not-running scope=${SCOPE} run_tag=${RUN_TAG}"
  fi
  echo "master_log=${MASTER_LOG}"
  if [[ -f "$MASTER_LOG" ]]; then
    echo
    tail -n 25 "$MASTER_LOG"
  fi
}

tail_log() {
  if [[ ! -f "$MASTER_LOG" ]]; then
    echo "Log does not exist: ${MASTER_LOG}" >&2
    exit 1
  fi
  tail -f "$MASTER_LOG"
}

run_check() {
  cd "$PROJECT_ROOT"
  export PYTHONPATH="${XGB_PYTHON_PACKAGE}${PYTHONPATH:+:${PYTHONPATH}}"
  run_preflight
}

case "$ACTION" in
  start)
    validate_scope
    start_queue
    ;;
  status)
    validate_scope
    show_status
    ;;
  tail)
    validate_scope
    tail_log
    ;;
  check)
    run_check
    ;;
  approve-synthetic)
    approve_synthetic_analysis
    ;;
  _run)
    validate_scope
    run_queue
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage
    exit 2
    ;;
esac
