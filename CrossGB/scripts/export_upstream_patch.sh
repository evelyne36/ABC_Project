#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/official-xgboost-at-baseline" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CROSSGB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_ROOT="${CROSSGB_ROOT}/xgboost-master"
UPSTREAM_ROOT="$(cd "$1" && pwd)"
BASELINE="41591eeff5b10caabe82d65fc7164b447eabd249"
OUTPUT="${CROSSGB_ROOT}/patches/crossgb-xgboost-${BASELINE:0:12}.patch"
scope=(
  include/xgboost/tree_model.h
  python-package/xgboost/sklearn.py
  src/gbm/gbtree.cc
  src/learner.cc
  src/predictor/cpu_predictor.cc
  src/tree/param.h
  src/tree/tree_model.cc
  src/tree/tree_updater.cc
  src/tree/tree_view.cc
  src/tree/tree_view.h
  src/tree/updater_crossgb_hist.cc
  tests/python/test_crossgb.py
)

actual="$(git -C "${UPSTREAM_ROOT}" rev-parse HEAD)"
if [[ "${actual}" != "${BASELINE}" ]]; then
  echo "Expected upstream ${BASELINE}, got ${actual}" >&2
  exit 1
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/crossgb-patch.XXXXXX")"
trap 'rm -rf "${tmp}"' EXIT
for file in "${scope[@]}"; do
  mkdir -p "${tmp}/upstream/$(dirname "${file}")" "${tmp}/crossgb/$(dirname "${file}")"
  if [[ -f "${UPSTREAM_ROOT}/${file}" ]]; then
    cp "${UPSTREAM_ROOT}/${file}" "${tmp}/upstream/${file}"
  fi
  cp "${VENDOR_ROOT}/${file}" "${tmp}/crossgb/${file}"
done

(
  cd "${tmp}"
  git diff --no-index --binary --src-prefix=a/ --dst-prefix=b/ upstream crossgb
) >"${OUTPUT}" || status=$?
if [[ "${status:-0}" -gt 1 ]]; then
  exit "${status}"
fi
echo "${OUTPUT}"
