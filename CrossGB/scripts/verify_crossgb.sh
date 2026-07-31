#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CROSSGB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XGBOOST_ROOT="${CROSSGB_XGBOOST_ROOT:-${CROSSGB_ROOT}/xgboost-master}"
PYTHON_BIN="${PYTHON_BIN:-python}"

test -f "${XGBOOST_ROOT}/src/tree/updater_crossgb_hist.cc"
test -f "${XGBOOST_ROOT}/tests/python/test_crossgb.py"
grep -Fq 'DMLC_REGISTRY_LINK_TAG(updater_crossgb_hist)' \
  "${XGBOOST_ROOT}/src/tree/tree_updater.cc"
grep -Fq 'XGBOOST_REGISTER_TREE_UPDATER(CrossGBHistMaker, "grow_crossgb_histmaker")' \
  "${XGBOOST_ROOT}/src/tree/updater_crossgb_hist.cc"
case "$(uname -s)" in
  Darwin) library="${XGBOOST_ROOT}/lib/libxgboost.dylib" ;;
  Linux) library="${XGBOOST_ROOT}/lib/libxgboost.so" ;;
  *) echo "Unsupported verification platform: $(uname -s)" >&2; exit 2 ;;
esac
test -f "${library}"
strings_output="$(mktemp "${TMPDIR:-/tmp}/crossgb-strings.XXXXXX")"
trap 'rm -f "${strings_output}"' EXIT
strings "${library}" >"${strings_output}"
grep -Fq 'grow_crossgb_histmaker' "${strings_output}"
grep -Fq 'enable_crossgb' "${strings_output}"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${library}"
else
  shasum -a 256 "${library}"
fi

CROSSGB_EXPECTED_XGBOOST_ROOT="${XGBOOST_ROOT}" \
  PYTHONPATH="${XGBOOST_ROOT}/python-package${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import xgboost as xgb

root = Path(os.environ["CROSSGB_EXPECTED_XGBOOST_ROOT"]).resolve()
package = Path(xgb.__file__).resolve()
library = Path(xgb.build_info()["libxgboost"]).resolve()
package.relative_to(root / "python-package")
library.relative_to(root / "lib")
print(package)
print(library)
PY
