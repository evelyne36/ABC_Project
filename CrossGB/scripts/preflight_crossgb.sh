#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CROSSGB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XGBOOST_ROOT="${CROSSGB_XGBOOST_ROOT:-${CROSSGB_ROOT}/xgboost-master}"
EXPECTED_UPSTREAM_REVISION="41591eeff5b10caabe82d65fc7164b447eabd249"
PYTHON_BIN="${PYTHON_BIN:-python}"
CXX_BIN="${CXX:-c++}"

fail() {
  echo "CrossGB preflight error: $*" >&2
  exit 1
}

version_at_least() {
  local actual="$1"
  local required_major="$2"
  local required_minor="$3"
  local major
  local minor
  IFS=. read -r major minor _ <<<"${actual}"
  [[ "${major}" =~ ^[0-9]+$ && "${minor}" =~ ^[0-9]+$ ]] ||
    fail "cannot parse version '${actual}'"
  ((major > required_major || (major == required_major && minor >= required_minor)))
}

for command_name in cmake git grep strings "${PYTHON_BIN}" "${CXX_BIN}"; do
  command -v "${command_name}" >/dev/null 2>&1 ||
    fail "required command '${command_name}' was not found on PATH"
done
if ! command -v ninja >/dev/null 2>&1 && ! command -v make >/dev/null 2>&1; then
  fail "neither Ninja nor Make was found; install ninja-build or make"
fi

read -r python_major python_minor < <(
  "${PYTHON_BIN}" -c 'import sys; print(sys.version_info.major, sys.version_info.minor)'
)
if ((python_major < 3 || (python_major == 3 && python_minor < 12))); then
  fail "'${PYTHON_BIN}' is Python ${python_major}.${python_minor}; Python 3.12+ is required"
fi

cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
version_at_least "${cmake_version}" 3 18 ||
  fail "CMake ${cmake_version} is too old; CMake 3.18+ is required"

compiler_output="$("${CXX_BIN}" --version 2>&1)"
compiler_first_line="${compiler_output%%$'\n'*}"
if [[ "${compiler_output}" == *Clang* || "${compiler_output}" == *clang* ]]; then
  compiler_family="Clang"
  compiler_version="$(
    printf '%s\n' "${compiler_first_line}" |
      sed -E 's/.*[Vv]ersion[[:space:]]+([0-9]+(\.[0-9]+)*).*/\1/'
  )"
  version_at_least "${compiler_version}" 9 0 ||
    fail "Clang ${compiler_version} is too old; Clang 9+ is required"
elif [[ "${compiler_output}" == *GCC* || "${compiler_output}" == *gcc* ||
  "${compiler_output}" == *g++* || "${compiler_output}" == *"Free Software Foundation"* ]]; then
  compiler_family="GCC"
  compiler_version="$("${CXX_BIN}" -dumpfullversion -dumpversion)"
  version_at_least "${compiler_version}" 8 1 ||
    fail "GCC ${compiler_version} is too old; GCC 8.1+ is required"
else
  compiler_family="other"
  compiler_version="unknown"
  echo "CrossGB preflight warning: unrecognized C++ compiler: ${compiler_first_line}" >&2
fi

test -f "${XGBOOST_ROOT}/CMakeLists.txt" ||
  fail "bundled XGBoost source is missing: ${XGBOOST_ROOT}/CMakeLists.txt"
test -f "${XGBOOST_ROOT}/dmlc-core/CMakeLists.txt" ||
  fail "bundled dmlc-core source is missing"
test -f "${XGBOOST_ROOT}/src/tree/updater_crossgb_hist.cc" ||
  fail "CrossGB updater source is missing"
if [[ -d "${XGBOOST_ROOT}/.git" ]]; then
  actual_revision="$(git -C "${XGBOOST_ROOT}" rev-parse HEAD)"
  [[ "${actual_revision}" == "${EXPECTED_UPSTREAM_REVISION}" ]] ||
    fail "checkout revision ${actual_revision} does not match ${EXPECTED_UPSTREAM_REVISION}"
fi

temporary="$(mktemp -d "${TMPDIR:-/tmp}/crossgb-preflight.XXXXXX")"
trap 'rm -rf "${temporary}"' EXIT
printf '#include <filesystem>\nint main() { return 0; }\n' >"${temporary}/cxx17.cc"
"${CXX_BIN}" -std=c++17 -c "${temporary}/cxx17.cc" -o "${temporary}/cxx17.o" ||
  fail "'${CXX_BIN}' cannot compile C++17 source"

if [[ "$(uname -s)" == "Linux" && "${CROSSGB_USE_OPENMP:-ON}" != "OFF" ]]; then
  printf '#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n' \
    >"${temporary}/openmp.cc"
  "${CXX_BIN}" -std=c++17 -fopenmp "${temporary}/openmp.cc" \
    -o "${temporary}/openmp-check" ||
    fail "OpenMP is unavailable; install the compiler OpenMP runtime or set CROSSGB_USE_OPENMP=OFF"
fi

echo "CrossGB preflight: OK"
echo "  platform: $(uname -s) $(uname -m)"
echo "  python: $("${PYTHON_BIN}" --version 2>&1)"
echo "  cmake: ${cmake_version}"
echo "  compiler: ${compiler_family} ${compiler_version} (${CXX_BIN})"
echo "  generator: $(command -v ninja >/dev/null 2>&1 && echo Ninja || echo Make)"
echo "  source: ${XGBOOST_ROOT}"
