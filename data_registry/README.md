# Public dataset registry

This directory keeps reproducible dataset metadata outside ignored runtime data
directories. Dataset payloads are intentionally not versioned.

```bash
python3 scripts/download_datasets.py group
python3 scripts/download_datasets.py list --group openml
python3 scripts/download_datasets.py list --group tabarena
python3 scripts/download_datasets.py download --dataset tabarena:airfoil_self_noise
python3 scripts/download_datasets.py download --group timeseries --dataset ETTh1
python3 scripts/download_datasets.py verify --group timeseries --allow-missing
```

OpenML entries use fixed numeric data IDs from
`openml_source_manifest.json`; names or `version="active"` are never used.
The processed table is reconstructed with a deterministic complete-case
transform, checked against the registered row/feature counts, and written with
the target as the last column. `friedman1` is generated deterministically with
scikit-learn's `make_friedman1` using 2,000 rows, 10 features, noise 1.0, and
seed 42.
The 51 TabArena entries use the fixed data/task IDs, target columns, dataset
versions, source URLs, and license metadata captured from OpenML Suite 457 in
`tabarena_suite_457.json`. The suite reported `in_preparation` when retrieved
on 2026-07-29, so this registry deliberately pins the captured member IDs
instead of following the mutable suite alias. Retained CSV snapshots are
checked against `checksums.sha256`. TabArena downloads additionally check the
provider ARFF MD5. Every fresh converted CSV receives a local `.sha256`
sidecar; subsequent verification uses that sidecar. The registry's historical
CSV checksum is not imposed on a fresh fixed-ID OpenML conversion because
harmless pandas/scikit-learn dtype and line-ending differences can change CSV
bytes while preserving the registered complete-case table.
Time-series entries use fixed URLs from `timeseries_registry.json`.
`checksums.sha256` pins every payload that was present during cleanup.

A download is refused when an entry lacks a pinned checksum. Consequently,
`electricity`, `traffic`, and `solar` are listed but not automatically fetched.

The retained evidence does not establish a dataset-specific license for every
payload. Listings therefore report `unknown; verify upstream dataset metadata`.
Historical per-dataset license notes remain in `tabarena_catalog.md`, but are
not treated as an automated redistribution grant.
