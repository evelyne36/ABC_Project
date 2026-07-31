# Upstream baseline evidence

Pinned official XGBoost commit:

```text
41591eeff5b10caabe82d65fc7164b447eabd249
2026-05-14 Support vector leaf SHAP on GPU (#12210)
```

Evidence collected against a filtered clone of
`https://github.com/dmlc/xgboost.git`:

- `CMakeLists.txt`, `src/context.cc`, `src/tree/updater_approx.cc`,
  `python-package/pyproject.toml`, and `tests/python/test_basic.py` have exact
  Git blob-hash matches.
- Exhaustive comparison of tracked vendor files, excluding flattened
  `dmlc-core` and the known CrossGB modification boundary, checked 1,370
  files: 1,368 exact matches, zero missing upstream paths, and two explained
  differences.
- `src/tree/tree_updater.cc` differs because CrossGB adds the required static
  registry force-link.
- `R-package/src/xgboost_R.h` differs by one pre-existing line: the vendor
  snapshot removed an upstream `<<<<<<< HEAD` merge marker.

The outer repository is not an XGBoost Git clone and cannot itself prove
ancestry. The pinned SHA is established by content identity, not by inherited
Git metadata.
