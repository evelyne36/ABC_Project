#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CROSSGB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT="${CROSSGB_ROOT}/patches/MANIFEST.sha256"
files=(
  README.md
  patches/PATCH_SCOPE.txt
  patches/UPSTREAM_BASELINE.md
  patches/crossgb-xgboost-41591eeff5b1.patch
  scripts/build_crossgb.sh
  scripts/export_upstream_patch.sh
  scripts/generate_manifest.sh
  scripts/preflight_crossgb.sh
  scripts/roundtrip_crossgb.py
  scripts/run_targeted_tests.sh
  scripts/smoke_crossgb.py
  scripts/test_linux_release.sh
  scripts/verify_crossgb.sh
  xgboost-master/include/xgboost/tree_model.h
  xgboost-master/python-package/xgboost/sklearn.py
  xgboost-master/src/gbm/gbtree.cc
  xgboost-master/src/learner.cc
  xgboost-master/src/predictor/cpu_predictor.cc
  xgboost-master/src/tree/param.h
  xgboost-master/src/tree/tree_model.cc
  xgboost-master/src/tree/tree_updater.cc
  xgboost-master/src/tree/tree_view.cc
  xgboost-master/src/tree/tree_view.h
  xgboost-master/src/tree/updater_crossgb_hist.cc
  xgboost-master/tests/python/test_crossgb.py
)

cd "${CROSSGB_ROOT}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${files[@]}" >"${OUTPUT}"
else
  shasum -a 256 "${files[@]}" >"${OUTPUT}"
fi
echo "${OUTPUT}"
