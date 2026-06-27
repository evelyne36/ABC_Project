#!/usr/bin/env bash
set -euo pipefail
TASK_TYPE=multiclass HANDLING=onehot bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_linux_by_handling.sh"
