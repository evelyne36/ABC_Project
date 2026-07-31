# Synthetic CrossGB benchmark

## Purpose

This suite tests where CrossGB helps or fails under controlled synthetic
regression and classification mechanisms. The main protocol compares
`xgb_cpu` with `crossgb`; ablation protocols also include `crossgb_pred0`.

## Paper correspondence

This directory reproduces the main synthetic mechanism study. Appendix Section
C and Table 6 specify all generators; Tables 7--8 report the complete
five-model dimension-10 and dimension-20 results, and Table 9 reports the
paired CrossGB--XGBoost mechanism diagnostics. Formal component attribution is
implemented separately in `exp/crossgb_ablation`.

## Environment

```bash
python3 -m pip install -r requirements.txt
bash CrossGB/scripts/build_crossgb.sh
bash CrossGB/scripts/verify_crossgb.sh
```

## Data and protocol

Data are generated deterministically from `scenarios.py`. Protocols in
`protocols.py` freeze scenarios, dimensions, data seeds, nested-CV folds, and
candidate budgets. `synthetic_main_framework_smoke` uses 2x2 CV and 30
candidates; `synthetic_main_random300_single_seed` uses 5x4 CV and 300
candidates.

Run all commands from the repository root.

## Smoke versus full execution

The smoke protocol uses 2x2 nested CV and 30 candidates to validate generation,
training, and artifact contracts. It is non-confirmatory. The full protocol
uses 5x4 nested CV, 300 candidates, and the frozen main scenario grid.

### Smoke run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/synthetic_crossgb/run_benchmark.py \
  --protocol synthetic_main_framework_smoke \
  --run-id synthetic_crossgb_smoke
```

### Full run

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/synthetic_crossgb/run_benchmark.py \
  --protocol synthetic_main_random300_single_seed \
  --task-workers 4 --n-jobs 1 \
  --run-id synthetic_crossgb_random300
```

## Outputs and resume

Artifacts live in `exp/synthetic_crossgb/results/<run-id>/`. They include task
shards, candidate and fold tables, split/candidate audits, paired statistics,
case/type summaries, and metadata. Resume an identical interrupted run with:

```bash
PYTHONPATH=exp:CrossGB/xgboost-master/python-package \
python3 exp/synthetic_crossgb/run_benchmark.py \
  --protocol synthetic_main_random300_single_seed \
  --run-id synthetic_crossgb_random300 --resume
```

The protocol hash rejects incompatible resume settings.
The appendix tables are derived from the fold, case/type summary, paired
statistics, and audit files in this directory; do not substitute smoke output.

## Limitations

Synthetic mechanisms isolate hypotheses rather than represent natural dataset
frequency. The single-seed main protocol is exploratory; use multi-seed
confirmatory protocols before making population-level claims.
