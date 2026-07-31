# CrossGB

CrossGB is a research extension of the XGBoost CPU histogram tree builder. It
separates the feature used to route samples through a tree (the conditional
feature) from the feature used to form the response inside each leaf (the
predictive feature).

Standard XGBoost uses a scalar response in each leaf, whereas CrossGB uses a
selected feature or a constant bias feature:

```text
XGBoost: leaf(x) = w
CrossGB: leaf(x) = beta * z_j(x)
```

The bias feature is always `1`, so CrossGB retains a constant-leaf fallback.
Setting `enable_crossgb=True` selects the CrossGB updater; leaving it disabled
uses the ordinary XGBoost updater in the same local source tree.

## Reviewer quick start

Run the following commands from the extracted repository root on a Linux CPU
server:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

bash CrossGB/scripts/build_crossgb.sh
bash CrossGB/scripts/verify_crossgb.sh
bash CrossGB/scripts/run_targeted_tests.sh

export PYTHONPATH="$PWD/exp:$PWD/CrossGB/xgboost-master/python-package"
python scripts/download_datasets.py group
```

The root `requirements.txt` is the authoritative Linux CPU environment for all
submitted experiments, including the conventional and neural tabular baselines.
It lists every experiment dependency directly, so reviewers need only one
`pip install -r requirements.txt`; the smaller experiment-local manifests are
optional subsets for constrained hosts. The root manifest deliberately does
**not** install the PyPI `xgboost` wheel. CrossGB must be built from the included
source.

The manifest declares `torch==2.12.1` together with the official PyTorch CPU
wheel index. On Linux x86-64 this resolves to the reported
`torch-2.12.1+cpu` build; the local `+cpu` suffix is intentionally not written
in the portable requirement specifier.

The reported paper environment used CPython 3.13.14, NumPy 2.3.5, Pandas
2.3.3, SciPy 1.16.3, and scikit-learn 1.7.2. These scientific-package versions
are pinned by the root manifest. Python 3.12 is the supported minimum for the
bundled XGBoost source; use Python 3.13 for the closest environment match.

After installing the environment, a clean Linux host can execute the complete
preflight, build, verification, targeted-test, smoke, and serialization
sequence with:

```bash
bash CrossGB/scripts/test_linux_release.sh
```

Run the smoke command in the README of the experiment of interest before
starting a full 300-candidate search. Full nested-cross-validation runs can
require substantial CPU time.

## Repository layout

```text
CrossGB/
  README.md                    Linux build, upstream reconstruction, and tests
  xgboost-master/              complete modified XGBoost source tree
  patches/                     auditable patch against a pinned upstream revision
  scripts/                     build and runtime verification scripts
exp/
  random300/                   paired tabular XGBoost/CrossGB experiments
  tabular_extra_random300/     additional OpenML tabular experiments
  random300_baselinemodel/     independently tuned tabular baselines
  synthetic_crossgb/           mechanism-aligned synthetic experiments
  synthetic_baselinemodel/     synthetic conventional-tree baselines
  crossgb_ablation/            fixed-backbone CrossGB ablations
  timeseries_extra_random300/  tabularized time-series experiments
scripts/
  download_datasets.py         sole public-data download and verification entry
  audit_release.py             source-package cleanliness and anonymity audit
data_registry/                 fixed dataset identifiers, sources, and checksums
tests/                         release and data-pipeline regression tests
RELEASE_AUDIT.md               resolved submission audit and validation record
```

This README, `CrossGB/README.md`, and the single `README.md` in each experiment
directory are the authoritative documentation. Historical internal notes,
duplicate launch guides, result reports, generated logs, and datasets are not
part of the source release.

## Supported implementation path

The supported CrossGB configuration is:

- Linux CPU training;
- `tree_method="hist"` or `"auto"`;
- regression, binary classification, and scalar-tree multiclass
  (`multi_strategy="one_output_per_tree"`);
- in-memory tabular data.

GPU training or prediction, distributed training, external-memory training, and
the experimental multi-output-tree branch are outside the claimed CrossGB
support boundary.

The bundled XGBoost source requires Python 3.12 or newer, CMake 3.18 or newer,
and either GCC 8.1 or newer or Clang 9 or newer. The exact Ubuntu prerequisites,
build flags, source-tree import procedure, upstream revision, patch replay, and
tests are documented in [`CrossGB/README.md`](CrossGB/README.md).

## Minimal CrossGB example

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    tree_method="hist",
    device="cpu",
    enable_crossgb=True,
    crossgb_pred_colsample_bytree=1.0,
    crossgb_score_agg=0,
    crossgb_lambda_l2=1.0,
    n_estimators=100,
    max_depth=4,
    n_jobs=1,
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

All submitted experiments, examples, labels, serialized fields, and outputs use
CrossGB names.

## Data availability and placement

**Datasets are intentionally not included.** Every empirical dataset used by
the paper is public and is downloaded into its expected experiment path by the
root downloader. Synthetic datasets are generated from source code and fixed
seeds.

Inspect the fixed registry:

```bash
python scripts/download_datasets.py group
python scripts/download_datasets.py list --group openml
python scripts/download_datasets.py list --group tabarena
python scripts/download_datasets.py list --group timeseries
```

Download and verify a group:

```bash
python scripts/download_datasets.py download --group openml
python scripts/download_datasets.py verify --group openml

python scripts/download_datasets.py download --group tabarena
python scripts/download_datasets.py verify --group tabarena

python scripts/download_datasets.py download --group timeseries
python scripts/download_datasets.py verify --group timeseries --allow-missing
```

| Group | Paper use | Download location |
| --- | --- | --- |
| `openml` | Main Tables 3–4 and Appendix E additional tabular panels | `exp/tabular_extra_random300/data/` |
| `tabarena` | Main paired tabular study and Appendix E Tables 14–15 | `dataset/TabArena_regression/`, `dataset/TabArena_binary/`, `dataset/TabArena_multiclass/` |
| `timeseries` | Main Table 5 and Appendix F Table 18 | `exp/timeseries_extra_random300/data/raw/` |
| generated synthetic data | Main Table 1 and fixed-backbone ablations | created at runtime; no download |

The 51-dataset TabArena mapping is pinned to the data IDs, task IDs, versions,
targets, source URLs, and source checksums captured from OpenML Suite 457. The
suite alias was marked `in_preparation` when recorded and is therefore not used
as a mutable dependency.

The time-series registry contains three later-expansion entries without retained
checksums (`electricity`, `traffic`, and `solar`). They are not used by the
submitted paper runs and are skipped by group download. `--allow-missing`
suppresses only missing or explicitly unavailable entries; integrity failures
still return a non-zero exit status.

## Experimental logic

The main tabular benchmark pipeline is:

```text
public dataset
  -> deterministic dataset-wide schema normalization
  -> outer train/test split
  -> shared XGBoost/CrossGB candidate backbones
  -> inner cross-validation on outer-train
  -> candidate selection
  -> refit on the complete outer-train split
  -> evaluation on the untouched outer-test split
  -> fold-level and dataset-level aggregation
```

Schema normalization fixes the target-last layout, removes rows with missing
targets or incomplete required inputs, and constructs one deterministic
categorical feature schema before splitting. It does not use target statistics.
Using one schema is intentional: train, validation, and test matrices must have
identical columns even when a category is absent from a particular training
fold (including the limiting zero-observation case). All learned model
parameters, candidate selection, inner-CV fitting, and final refitting remain
restricted to the corresponding training partitions; the outer test fold is
used only for evaluation.

XGBoost and CrossGB share the same sampled tree backbone: depth, learning rate,
number of estimators, sampling, and regularization. CrossGB additionally
evaluates predictive-feature fractions and split-score aggregation. Each formal
runner records candidate sets, split protocol, seeds, selected configuration,
metrics, runtime, and source identity.

## Paper-to-code guide

| Experiment | Paper or appendix target | Documentation |
| --- | --- | --- |
| Synthetic CrossGB | Main Section 4.2 and Table 1; Appendix C | [`exp/synthetic_crossgb/README.md`](exp/synthetic_crossgb/README.md) |
| Synthetic baselines | Conventional-tree columns in Main Table 1; Appendix C | [`exp/synthetic_baselinemodel/README.md`](exp/synthetic_baselinemodel/README.md) |
| CrossGB ablation | Main Section 4.3 and Table 2; Appendix D | [`exp/crossgb_ablation/README.md`](exp/crossgb_ablation/README.md) |
| Random300 | Paired TabArena panels in Appendix E Tables 14–15 | [`exp/random300/README.md`](exp/random300/README.md) |
| Extra tabular | Main Section 4.4 and Tables 3–4; Appendix E Tables 14–15 | [`exp/tabular_extra_random300/README.md`](exp/tabular_extra_random300/README.md) |
| Tabular baselines | Baseline columns in Main Tables 3–4; Appendix B.3 and Table 16 | [`exp/random300_baselinemodel/README.md`](exp/random300_baselinemodel/README.md) |
| Time series | Main Section 4.5 and Table 5; Appendix F Table 18 | [`exp/timeseries_extra_random300/README.md`](exp/timeseries_extra_random300/README.md) |

Each experiment README distinguishes the quick smoke configuration from the
formal paper protocol, gives the required data command, and identifies the
generated result files.

## Reproducibility and release checks

Before a formal run:

1. install the root environment and record `python -m pip freeze`;
2. verify the pinned upstream revision and `CrossGB/patches/MANIFEST.sha256`;
3. build and run the CrossGB verifier, targeted pytest, smoke, and round-trip
   tests;
4. download and verify only the dataset group required by the experiment;
5. run the exact formal command and seeds from the experiment README;
6. keep each estimator single-threaded when using process-level task workers;
7. retain the generated environment, split, candidate, provenance, and summary
   artifacts;
8. before redistributing the source, run:

```bash
python -m pytest -q tests
python scripts/audit_release.py
```

Generated `logs/`, `results/`, `nohup_logs/`, build directories, datasets, and
compiled libraries are excluded from this source release. See
[`RELEASE_AUDIT.md`](RELEASE_AUDIT.md) for the resolved audit items and
validation commands.

## License and attribution

CrossGB-specific source and experiment code is released under the repository
[`LICENSE`](LICENSE) (MIT). The embedded XGBoost source retains the upstream
XGBoost license and notices. CrossGB-specific changes are identified by the
patch and modified-file manifest under `CrossGB/patches/`. Dataset licenses
remain with their original providers and are recorded when available; no
dataset payload is redistributed here.
