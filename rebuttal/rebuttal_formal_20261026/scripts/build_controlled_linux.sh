#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
  echo 'CGB build is reserved for the authorized Linux server.' >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
native_source="$project_root/rebuttal_execution_20261025/native/controlled"
cmake -S "$native_source" -B "$native_source/build-rebuttal" -DUSE_CUDA=OFF -DUSE_OPENMP=ON -DBUILD_WITH_GIT_HASH=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$native_source/build-rebuttal" --parallel 8
