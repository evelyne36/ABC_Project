#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${EXP12_PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

case "${1:-start}" in
  start)
    exec "${PYTHON_BIN}" exp12/run_all_full.py start "${@:2}"
    ;;
  fg|foreground)
    exec "${PYTHON_BIN}" exp12/run_all_full.py start --wait "${@:2}"
    ;;
  status)
    exec "${PYTHON_BIN}" exp12/run_all_full.py status "${@:2}"
    ;;
  stop)
    exec "${PYTHON_BIN}" exp12/run_all_full.py stop "${@:2}"
    ;;
  -h|--help|help)
    exec "${PYTHON_BIN}" exp12/run_all_full.py --help
    ;;
  *)
    exec "${PYTHON_BIN}" exp12/run_all_full.py "$@"
    ;;
esac

