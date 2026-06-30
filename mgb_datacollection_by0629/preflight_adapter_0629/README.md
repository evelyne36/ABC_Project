# MetaGB Preflight Adapter 0629

This subproject tests whether cheap preflight analysis can estimate a dataset's
fit to MetaGB before running a full MGB-vs-XGB experiment.

It reads existing 0629 result labels from:

```text
mgb_datacollection_by0629/derived/dataset_level_mgb_vs_xgb.csv
```

and writes outputs only under:

```text
mgb_datacollection_by0629/preflight_adapter_0629/outputs/
```

The current score has two layers:

- `mechanism_alignment_score_raw`: intrinsic mechanism score, closest to the
  original `metagb_preflight_adapter` idea.
- `mechanism_alignment_score`: family-calibrated adapter score used for final
  bands and audit tables.

The family-calibrated score is intentionally coarse:

- synthetic: `0.00-0.35`, `0.35-0.60`, `0.60-1.00`;
- TabArena / tabular otherdata: `0.00-0.30`, `0.30-0.50`, `0.50-1.00`.

The per-row score only reads dataset preflight features. The family-specific
rules and thresholds are calibrated against the 0629 retrospective outcomes, so
they should be treated as a current evidence-backed prioritization rule rather
than an externally validated classifier.
