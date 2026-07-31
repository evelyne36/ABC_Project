#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CrossGB Linux release test must run on Linux." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/preflight_crossgb.sh"
"${SCRIPT_DIR}/build_crossgb.sh"
"${SCRIPT_DIR}/verify_crossgb.sh"
"${SCRIPT_DIR}/run_targeted_tests.sh"

echo "CrossGB Linux release test: OK"
