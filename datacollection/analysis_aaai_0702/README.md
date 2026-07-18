# datacollection0702 NeurIPS-style Analysis Notes

This directory contains the incremental Markdown-only analysis for the
`datacollection0702` experimental snapshot. It intentionally lives inside a new
subdirectory to avoid mixing new 0702 interpretation with the earlier
`mgb_datacollection_by0629` reports.

## Files

- `mgb_vs_xgb_neurips_analysis_0702.md`: main analysis report, written as a
  paper-facing experimental synthesis.
- `appendix_tables_0702.md`: supporting tables, coverage notes, and representative
  positive/negative cases used by the main report.

## Scope Contract

The analysis is documentation-only. It does not modify benchmark code, raw logs,
JSON artifacts, CSV summaries, or experiment outputs.

The primary statistical unit is a paired outer fold where both `xgb_cpu` and
`current_metagb` completed successfully under the same dataset, task, preprocessing
mode, protocol, and outer-fold index.
