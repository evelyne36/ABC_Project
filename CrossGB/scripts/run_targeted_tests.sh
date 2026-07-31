#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CROSSGB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XGBOOST_ROOT="${CROSSGB_XGBOOST_ROOT:-${CROSSGB_ROOT}/xgboost-master}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH="${XGBOOST_ROOT}/python-package${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m pytest -q "${XGBOOST_ROOT}/tests/python/test_crossgb.py"
PYTHONPATH="${XGBOOST_ROOT}/python-package${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_crossgb.py"
PYTHONPATH="${XGBOOST_ROOT}/python-package${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/roundtrip_crossgb.py"
