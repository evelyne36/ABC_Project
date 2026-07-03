# Adapter Academic

This folder contains the academic version of the MetaGB mechanism-aware preflight adapter.

It is separate from `metagb_preflight_adapter/`, which has been saved as:

```text
metagb_preflight_adapter_saved_20260623/
```

## Claim

The adapter is not a supervised benchmark-win predictor. It is a theorem-guided
mechanism-alignment diagnostic:

```text
MetaGB gains are structurally plausible when local residual/gradient mass has
strong projection onto a predictive feature inside conditional regions.
```

The score is derived from MetaGB's second-order local linear leaf objective and
is validated through:

1. synthetic mechanism recovery;
2. true predictive-feature hit rate;
3. real benchmark retrospective audit;
4. ablation and sensitivity checks.

## Commands

Run synthetic mechanism recovery:

```bash
python3 adapter_academic/scripts/run_synthetic_mechanism.py
```

Run real benchmark audit using the existing preflight table:

```bash
python3 adapter_academic/scripts/run_real_audit.py
```

Run ablation and sensitivity over synthetic outputs:

```bash
python3 adapter_academic/scripts/run_ablation_sensitivity.py
```

## Outputs

```text
adapter_academic/outputs/synthetic_scores.csv
adapter_academic/outputs/synthetic_summary.csv
adapter_academic/outputs/synthetic_mechanism_report.md
adapter_academic/outputs/real_audit_scores.csv
adapter_academic/outputs/real_audit_report.md
adapter_academic/outputs/ablation_results.csv
adapter_academic/outputs/sensitivity_results.csv
adapter_academic/outputs/ablation_sensitivity_report.md
```

## Paper Position

This module is intended as MetaGB's third contribution:

1. Method: feature-selected local linear leaves.
2. System: integration into XGBoost hist.
3. Analysis: mechanism-aware preflight adapter that explains applicability boundaries.
