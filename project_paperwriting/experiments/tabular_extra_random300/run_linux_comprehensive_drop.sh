#!/usr/bin/env bash
set -euo pipefail
HANDLING=drop bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_linux_comprehensive.sh"
