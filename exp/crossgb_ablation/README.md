# CrossGB mechanism ablation

## Purpose

This experiment freezes an XGBoost backbone and isolates CrossGB split
features, leaf prediction, predictive-feature fractions, and score
aggregation. The formal grid has 16 variants and covers both synthetic and
selected real-data diagnostics.

## Paper correspondence

This directory reproduces the formal fixed-backbone mechanism study summarized
in the main paper. See Appendix Sections B.5 and D, especially Tables 5 and
10--13. These results are mechanism attribution and sensitivity evidence; they
must not be pooled with the equal-budget end-to-end model comparisons.

## Environment

From the repository root, install the common environment and build the local
CrossGB-enabled XGBoost checkout:

```bash
python3 -m pip install -r requirements.txt
bash CrossGB/scripts/build_crossgb.sh
bash CrossGB/scripts/verify_crossgb.sh
```

## Data and protocol

- Real data: repository `dataset/TabArena_*` directories.
- Synthetic data: scenarios from `exp/synthetic_crossgb/scenarios.py`.
- Smoke and formal protocols freeze datasets, seeds, dimensions, CV folds,
  variants, and candidate budgets.
- `source_identity` hashes the active experiment code and the current
  `CrossGB/xgboost-master` implementation.

Run all commands from the repository root.

Download the real-data panel before starting the gated real stage:

```bash
python3 scripts/download_datasets.py download --group tabarena
python3 scripts/download_datasets.py verify --group tabarena
```

## Smoke versus formal execution

The smoke commands below use the frozen smoke protocol and one task to verify
imports, configuration, artifact writing, and the selected variant path. They
do not reproduce the paper statistics. The formal launcher runs all frozen
scenarios, dimensions, seeds, folds, and 16 variants.

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 -m crossgb_ablation.run_ablation \
  --protocol smoke --domains synthetic --plan-only \
  --run-id crossgb_ablation_plan
```

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 -m crossgb_ablation.run_ablation \
  --protocol smoke --domains synthetic --limit-tasks 1 \
  --task-workers 1 --n-jobs 1 \
  --run-id crossgb_ablation_smoke
```

### Formal two-stage run

```bash
bash exp/crossgb_ablation/scripts/launch_ablation.sh start synthetic
```

After completing and approving the synthetic conclusion according to the
launcher gate, start real data with:

```bash
bash exp/crossgb_ablation/scripts/launch_ablation.sh approve-synthetic
bash exp/crossgb_ablation/scripts/launch_ablation.sh start real
```

## Outputs and resume

Artifacts are written below `exp/crossgb_ablation/results/<run-id>/` and
include task shards, backbone selections, fold/prediction tables, statistical
tables, source/environment identity, execution manifests, and completeness
audits. The launcher always passes `--resume`; direct runs resume with the same
arguments, run ID, and `--resume`. Hash checks reject protocol/source changes.
Use only a run whose completeness and provenance audits pass when regenerating
the appendix tables.

## Limitations

The real stage is deliberately gated on a completed synthetic analysis.
Fixed-backbone ablations isolate mechanisms but do not replace an end-to-end
retuned model comparison. Do not bypass failed provenance or completeness
audits.
