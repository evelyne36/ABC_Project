from __future__ import annotations

import sys
from pathlib import Path

from compare_xgbbase_xgbv2_xgbv3_speed_0518 import main as compare_main


def main() -> int:
    if "--versions" not in sys.argv:
        sys.argv.extend(["--versions", "xgbbase", "xgbv5"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(Path(__file__).resolve().parent / "results0519_xgbbase_vs_xgbv5")])
    return compare_main()


if __name__ == "__main__":
    raise SystemExit(main())
