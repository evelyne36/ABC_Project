from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp12.utils.benchmark_core import main_for_script


if __name__ == "__main__":
    raise SystemExit(main_for_script("exp2_ordinal", "exp2", "ordinal"))

