#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CROSSGB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XGBOOST_ROOT="${CROSSGB_XGBOOST_ROOT:-${CROSSGB_ROOT}/xgboost-master}"
BUILD_DIR="${CROSSGB_BUILD_DIR:-${XGBOOST_ROOT}/build_crossgb}"
BUILD_JOBS="${CROSSGB_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"

"${SCRIPT_DIR}/preflight_crossgb.sh"

if command -v ninja >/dev/null 2>&1; then
  cmake -S "${XGBOOST_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CROSSGB_BUILD_TYPE:-Release}" \
    -DUSE_CUDA=OFF \
    -DUSE_OPENMP="${CROSSGB_USE_OPENMP:-ON}" \
    -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=OFF \
    -GNinja
else
  cmake -S "${XGBOOST_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CROSSGB_BUILD_TYPE:-Release}" \
    -DUSE_CUDA=OFF \
    -DUSE_OPENMP="${CROSSGB_USE_OPENMP:-ON}" \
    -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=OFF
fi
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

echo "CrossGB native library:"
find "${XGBOOST_ROOT}/lib" -maxdepth 1 -type f \
  \( -name 'libxgboost*.so*' -o -name 'libxgboost*.dylib' -o -name 'xgboost.dll' \) -print
