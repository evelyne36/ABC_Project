# CrossGB source-release audit

This file records the submission-package issues found during the final source
audit and how each issue was resolved. It is not an experiment entry point;
reviewers should start from `README.md`.

## Resolved issues

1. **Linux prerequisites were underspecified.**

   Resolution: `CrossGB/README.md` now specifies Python 3.12+, CMake 3.18+,
   GCC 8.1+ or Clang 9+, C++17, binutils, a build generator, and OpenMP.
   `CrossGB/scripts/preflight_crossgb.sh` checks these requirements before
   CMake starts, including a real C++17 compilation and a Linux OpenMP
   compilation probe.

2. **The verifier depended on undeclared `ripgrep`.**

   Resolution: `verify_crossgb.sh` uses portable `grep`, verifies required
   updater symbols, and asserts that both the imported Python package and
   native library come from the submitted source tree.

3. **The root dependency file did not install every experiment environment.**

   Resolution: the root `requirements.txt` is now the one-shot Linux CPU
   manifest for the complete submitted suite: scientific Python packages,
   LightGBM, CatBoost, CPU-only PyTorch, TorchMetrics, Lightning, PyTabKit,
   plotting, Parquet, and tests. It deliberately excludes the PyPI `xgboost`
   wheel and contains no nested `-r` includes. Experiment-specific requirement
   files remain only as smaller optional subsets. A Linux x86-64/Python 3.12
   pip dry-run resolves the complete manifest; `torch==2.12.1` resolves from
   the configured official CPU index to the reported `2.12.1+cpu` Linux wheel.

4. **The authoritative OpenML registry did not cover every paper and launcher
   dataset.**

   Resolution: the fixed-ID registry now covers all 27 required additional
   tabular files. Eight formerly missing public datasets use fixed numeric
   OpenML data IDs. Friedman1 is generated deterministically with 2,000 rows,
   10 features, noise 1.0, and seed 42.

5. **Fresh OpenML conversion did not reproduce the retained complete-case
   protocol.**

   Resolution: the downloader applies one explicit complete-case transform,
   writes the target last, checks registered row and feature counts, and emits
   canonical line endings. Fresh converted CSVs receive a local SHA-256
   sidecar; retained snapshots remain auditable through
   `data_registry/checksums.sha256`. TabArena downloads additionally verify the
   provider ARFF MD5.

6. **`verify --allow-missing` could hide corrupt data.**

   Resolution: missing/unavailable entries and integrity failures now use
   separate counters. `--allow-missing` suppresses only the former; invalid,
   unverified, or checksum-mismatched files still return a non-zero status.

7. **The release audit could report success after checking zero files.**

   Resolution: `scripts/audit_release.py` scans the actual extracted tree
   instead of relying on an enclosing Git worktree, fails on an empty scan,
   and checks ignored files as well as tracked files.

8. **The package contained anonymity and cleanliness risks.**

   Resolution: operating-system metadata, personal absolute paths, private
   workspace names, historical data-download manifests, generated audit CSVs,
   candidate inventories, and stale execution-order records were removed.
   The release audit rejects personal home paths, private project markers,
   build products, datasets, results, logs, caches, and noncanonical
   experiment Markdown.

9. **Experiment documentation had competing entry points and weak paper
   correspondence.**

   Resolution: each of the seven experiment directories now has exactly one
   English `README.md`. Necessary protocol details were merged into those
   files; 17 duplicate deployment guides, internal reports, runtime notes, and
   historical launch documents were removed. Every experiment README now
   states its paper/appendix mapping, data command, smoke-versus-formal
   distinction, formal command, output contract, and limitation.

10. **The source layout referred to historical `docs_*` material.**

    Resolution: historical `docs_*` directories and references are absent from
    the submitted package. The root README, CrossGB build README, and seven
    experiment READMEs are the only execution documentation.

11. **Submission defaults and paper-panel membership were ambiguous.**

    Resolution: the Random300 and extra-tabular generic runners default to
    one-hot preprocessing, and regression tests lock those defaults. Explicit
    drop-handling launchers and the strong-baseline drop condition remain
    available. The strong-baseline ten-regression schedule includes
    `QSAR_fish_toxicity.csv` with an explicit TabArena fallback directory. The
    main synthetic grid contains only dimensions 10 and 20; the unrelated
    high-dimensional nuisance stress case remains separately labelled.

12. **The official-source reconstruction path was not exercised end to end.**

    Resolution: all native scripts accept `CROSSGB_XGBOOST_ROOT`. The patch was
    regenerated from the pinned official commit, applied to a clean official
    checkout, compiled there, imported from that checkout, and passed the
    targeted tests, training smoke, and canonical serialization tests.
    The build README now gives that exact command sequence.

13. **Licensing and source-comment presentation needed a release-level pass.**

    Resolution: CrossGB-specific code now has a root MIT `LICENSE`; embedded
    XGBoost retains its upstream license. Temporary implementation-style
    comments were removed from the updater. Source code, APIs, examples, labels,
    serialized fields, environment variables, and outputs use CrossGB names.

## Validation commands

Run from the extracted repository root:

```bash
python -m pytest -q tests
python scripts/audit_release.py

find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

(cd CrossGB && sha256sum -c patches/MANIFEST.sha256)
git -C /path/to/official/xgboost apply -p2 --check \
  "$PWD/CrossGB/patches/crossgb-xgboost-41591eeff5b1.patch"
```

On a clean Linux CPU host, after `python -m pip install -r requirements.txt`:

```bash
bash CrossGB/scripts/test_linux_release.sh
```

This one command performs the toolchain preflight, Release CPU build, local
library verification, targeted pytest suite, smoke training, and canonical
model serialization round trip.

## Evidence collected during packaging

- release audit checks more than 1,700 actual package files;
- release/data/requirements regression tests pass;
- the patch-applied official XGBoost checkout builds from a clean source tree;
- targeted CrossGB tests, smoke training, and serialization round trip pass;
- all shell scripts pass `bash -n`;
- the CrossGB patch applies cleanly to the pinned official XGBoost revision;
- all files listed by `CrossGB/patches/MANIFEST.sha256` pass checksum
  verification;
- the root one-shot requirements manifest resolves for Linux x86-64 with
  CPython 3.12 in pip dry-run mode;
- the dataset registry reports OpenML 27/27, TabArena 51/51, and time series
  11/14 downloadable;
- all 27 retained OpenML snapshot checksums pass, and fresh fixed-ID downloads
  were exercised for Boston, Moneyball, Breast-W, and Credit Approval.

The native build recorded during this packaging audit ran on a patch-applied
official checkout on Darwin ARM64 with OpenMP deliberately disabled. Linux
OpenMP compilation is therefore enforced by the preflight and the dedicated
Linux release script rather than being inferred from the local build.

## Intentional release limits

- Dataset payloads are not redistributed.
- The time-series registry retains three later-expansion entries without
  pinned payload checksums (`electricity`, `traffic`, and `solar`). They are
  not used by the submitted paper runs and are skipped automatically.
- Full 300-candidate nested-cross-validation experiments are computationally
  expensive; smoke commands validate installation and contracts but do not
  reproduce paper statistics.
- Parameters, labels, examples, serialized fields, and generated artifacts use
  CrossGB names exclusively.
