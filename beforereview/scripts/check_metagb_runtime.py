#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Optional


def expected_lib_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libxgboost.dylib"
    if system == "Windows":
        return "xgboost.dll"
    return "libxgboost.so"


def find_native_lib(project_root: Path) -> Optional[Path]:
    lib_dir = project_root / "xgboost-master" / "lib"
    preferred = lib_dir / expected_lib_name()
    if preferred.exists():
        return preferred
    system = platform.system()
    if system == "Darwin":
        candidates = sorted(lib_dir.glob("libxgboost*.dylib"))
    elif system == "Windows":
        candidates = sorted(lib_dir.glob("xgboost*.dll"))
    else:
        candidates = sorted(lib_dir.glob("libxgboost*.so*"))
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check local embedded MetaGB/XGBoost runtime.")
    parser.add_argument("--project-root", default=str(repo_root))
    parser.add_argument("--xgboost-python-package", default=None)
    parser.add_argument("--skip-fit", action="store_true")
    return parser.parse_args()


def main() -> int:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "This modified XGBoost checkout requires Python 3.10+ "
            f"(current: {sys.version.split()[0]})."
        )

    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    python_package = (
        Path(args.xgboost_python_package).expanduser().resolve()
        if args.xgboost_python_package
        else project_root / "xgboost-master" / "python-package"
    )
    lib_path = find_native_lib(project_root)

    print("platform:", platform.platform())
    print("project_root:", project_root)
    print("python_package:", python_package)
    print("native_lib:", lib_path or project_root / "xgboost-master" / "lib" / expected_lib_name())

    if not python_package.exists():
        raise FileNotFoundError(f"Local XGBoost python package not found: {python_package}")
    if lib_path is None:
        raise FileNotFoundError(
            f"Native XGBoost library not found under {project_root / 'xgboost-master' / 'lib'}. "
            "Rebuild xgboost-master on this machine first."
        )

    for module_name in list(sys.modules):
        if module_name == "xgboost" or module_name.startswith("xgboost."):
            del sys.modules[module_name]
    sys.path.insert(0, str(python_package))

    import xgboost as xgb

    print("xgboost_file:", Path(xgb.__file__).resolve())
    if python_package not in Path(xgb.__file__).resolve().parents:
        raise RuntimeError("Imported xgboost is not from the local python-package directory.")

    if args.skip_fit:
        print("status: OK (import/library path only)")
        return 0

    model = xgb.XGBRegressor(
        n_estimators=1,
        max_depth=1,
        tree_method="hist",
        device="cpu",
        objective="reg:squarederror",
        enable_metagb=True,
        metagb_pred_colsample_bytree=1.0,
        verbosity=0,
    )
    model.fit([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], [0.0, 1.0, 1.0, 2.0])
    config = json.loads(model.get_booster().save_config())
    tree_params = config["learner"]["gradient_booster"]["tree_train_param"]
    print("enable_metagb:", tree_params.get("enable_metagb"))
    print("metagb_pred_colsample_bytree:", tree_params.get("metagb_pred_colsample_bytree"))
    if str(tree_params.get("enable_metagb")) not in {"1", "true", "True"}:
        raise RuntimeError("MetaGB was not enabled in the trained booster config.")

    print("status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
