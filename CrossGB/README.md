# CrossGB native XGBoost extension

This directory contains both:

- `xgboost-master/`: the complete modified XGBoost source tree; and
- `patches/`: an auditable CrossGB patch against a pinned official XGBoost
  revision.

The canonical public interface is:

- `enable_crossgb=True`;
- `crossgb_*` algorithm parameters;
- updater registry key `grow_crossgb_histmaker`;
- model fields `crossgb_leaf_features` and
  `crossgb_leaf_features_vector`.

Configurations and model files use only the canonical CrossGB names listed
above.

## Upstream baseline

CrossGB is based on the official XGBoost commit:

```text
41591eeff5b10caabe82d65fc7164b447eabd249
```

The exact comparison evidence is recorded in
[`patches/UPSTREAM_BASELINE.md`](patches/UPSTREAM_BASELINE.md). The patch scope
and modification boundary are listed in
[`patches/PATCH_SCOPE.txt`](patches/PATCH_SCOPE.txt).

## Option A: build the bundled full source

### 1. Install system prerequisites

Required tools:

- Python 3.12 or newer;
- GCC 8.1 or newer, or Clang 9 or newer, with C++17 support;
- CMake 3.18 or newer;
- Git;
- `strings` from GNU binutils;
- OpenMP (enabled by default for CPU experiments).

Example on Ubuntu 24.04 with the default GCC toolchain:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential binutils cmake git ninja-build \
  python3.12-dev python3.12-venv
```

GCC provides its OpenMP runtime on this path. If the server uses Clang instead,
also install `libomp-dev`. Older Ubuntu releases may not provide Python 3.12
through their default package repositories; use a Python 3.12 Conda/Mamba
environment or a newer distribution rather than silently running an older
interpreter.

Example on macOS:

```bash
brew install cmake libomp python@3.12
```

### 2. Create the Python environment

Run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

The final command must report Python 3.12 or newer.

Do not install the PyPI `xgboost` wheel for CrossGB experiments.

### 3. Run the toolchain preflight

```bash
bash CrossGB/scripts/preflight_crossgb.sh
```

The preflight checks Python, CMake, compiler, C++17, build tools, binutils, the
bundled `dmlc-core`, and the CrossGB updater source. On Linux it also verifies
the default OpenMP toolchain. Set `CROSSGB_USE_OPENMP=OFF` for a deliberate
single-threaded build without OpenMP.

### 4. Compile the native library

```bash
bash CrossGB/scripts/build_crossgb.sh
```

The build script reruns the preflight, then performs a Release, CPU-only CMake
build with CUDA disabled. Its default build directory is
`CrossGB/xgboost-master/build_crossgb/`; the resulting native library is
placed under `CrossGB/xgboost-master/lib/`.

Useful overrides:

```bash
CROSSGB_BUILD_JOBS=16 bash CrossGB/scripts/build_crossgb.sh
CROSSGB_USE_OPENMP=OFF bash CrossGB/scripts/build_crossgb.sh
CROSSGB_BUILD_DIR=/tmp/crossgb-build bash CrossGB/scripts/build_crossgb.sh
```

### 5. Use the source Python package

```bash
export PYTHONPATH="$PWD/CrossGB/xgboost-master/python-package:${PYTHONPATH:-}"
python -c 'import xgboost; print(xgboost.__file__); print(xgboost.build_info()["libxgboost"])'
```

Both printed paths must point into this repository. This check prevents an
unrelated site-packages XGBoost installation from silently running the
experiments. This is an intentional source-tree import through `PYTHONPATH`,
not an editable or PyPI installation.

### 6. Run the build and behavior tests

```bash
bash CrossGB/scripts/verify_crossgb.sh
bash CrossGB/scripts/run_targeted_tests.sh
```

`verify_crossgb.sh` checks that the compiled library exists, contains the
CrossGB updater and parameter symbols, and is loaded by the local Python
package. It uses standard `grep` and does not require ripgrep.
`run_targeted_tests.sh` runs:

- `tests/python/test_crossgb.py`;
- `scripts/smoke_crossgb.py`;
- `scripts/roundtrip_crossgb.py`.

The smoke test trains through the canonical API. The round-trip test verifies
that models serialize and reload the canonical CrossGB fields.

On a clean Linux release host, the complete native sequence can be run as one
command after installing `requirements.txt`:

```bash
bash CrossGB/scripts/test_linux_release.sh
```

This repeats the preflight, CPU build, local-library verification, targeted
pytest suite, smoke training, and canonical serialization round trip.

## Option B: reconstruct CrossGB from official XGBoost

This path demonstrates exactly how the official library becomes CrossGB
without relying on the bundled source snapshot.

### 1. Clone and pin official XGBoost

```bash
git clone --recursive https://github.com/dmlc/xgboost.git xgboost-crossgb
cd xgboost-crossgb
git checkout 41591eeff5b10caabe82d65fc7164b447eabd249
git submodule update --init --recursive
```

### 2. Check and apply the auditable patch

Set `CROSSGB_REPO` to the root of this repository:

```bash
export CROSSGB_REPO=/absolute/path/to/SourceCode_CrossGB
git apply -p2 --check \
  "$CROSSGB_REPO/CrossGB/patches/crossgb-xgboost-41591eeff5b1.patch"
git apply -p2 \
  "$CROSSGB_REPO/CrossGB/patches/crossgb-xgboost-41591eeff5b1.patch"
export CROSSGB_XGBOOST_ROOT="$PWD"
bash "$CROSSGB_REPO/CrossGB/scripts/preflight_crossgb.sh"
```

The `--check` command must succeed before applying. `-p2` removes the patch's
audit-only `upstream/` and `crossgb/` path prefixes. When the selected source
directory is a Git checkout, the preflight additionally requires its `HEAD` to
equal the pinned revision. `CROSSGB_XGBOOST_ROOT` makes every repository build
and verification script operate on this patched official checkout rather than
on the bundled snapshot.

### 3. Compile the patched clone

```bash
CROSSGB_XGBOOST_ROOT="$PWD" \
  CROSSGB_BUILD_DIR="$PWD/build_crossgb" \
  bash "$CROSSGB_REPO/CrossGB/scripts/build_crossgb.sh"

CROSSGB_XGBOOST_ROOT="$PWD" \
  bash "$CROSSGB_REPO/CrossGB/scripts/verify_crossgb.sh"

CROSSGB_XGBOOST_ROOT="$PWD" \
  bash "$CROSSGB_REPO/CrossGB/scripts/run_targeted_tests.sh"
```

These commands use the same pinned, CPU-only Release configuration and the
same import-path checks as the bundled-source path. On Linux, the complete
patched-checkout verification is also available as one command:

```bash
CROSSGB_XGBOOST_ROOT="$PWD" \
  bash "$CROSSGB_REPO/CrossGB/scripts/test_linux_release.sh"
```

## What the patch changes

The CrossGB extension is implemented in
`src/tree/updater_crossgb_hist.cc` (XGBoost uses C++, so the updater source has
a `.cc` suffix rather than `.c`). The integration proceeds through six
connected changes:

1. `src/tree/param.h` defines the CrossGB training parameters.
2. `src/gbm/gbtree.cc` selects
   `grow_crossgb_histmaker` for CPU histogram training.
3. `src/tree/updater_crossgb_hist.cc` builds CrossGB histograms, chooses
   conditional and predictive features, computes leaf coefficients, and
   registers the new updater.
4. `include/xgboost/tree_model.h`, `src/tree/tree_model.cc`, and
   `src/tree/tree_view.*` store and serialize the selected predictive feature
   for each leaf.
5. `src/predictor/cpu_predictor.cc` evaluates a CrossGB leaf as its learned
   coefficient times the selected feature value.
6. `python-package/xgboost/sklearn.py`, `src/learner.cc`, and
   `tests/python/test_crossgb.py` expose, validate, and test the public API and
   serialization behavior.

`src/tree/tree_updater.cc` contains the static registry force-link required for
the new updater translation unit.

## Re-export and verify the patch

Given an official checkout at the pinned commit:

```bash
bash CrossGB/scripts/export_upstream_patch.sh /path/to/official/xgboost
bash CrossGB/scripts/generate_manifest.sh
(cd CrossGB && shasum -a 256 -c patches/MANIFEST.sha256)
git -C /path/to/official/xgboost apply -p2 --check \
  "$PWD/CrossGB/patches/crossgb-xgboost-41591eeff5b1.patch"
```

On Linux, replace `shasum -a 256 -c` with `sha256sum -c` if `shasum` is not
available. `MANIFEST.sha256` covers the patch, integration files, tests, and
build/audit scripts.

## Supported boundary

The supported CrossGB path is CPU histogram training on in-memory tabular
data, including regression, binary classification, and scalar-tree
multiclass. GPU, distributed, external-memory, and experimental
multi-output-tree paths are not claimed as supported CrossGB configurations.

Build directories and compiled libraries are generated artifacts and should
not be included in a source-only release.
