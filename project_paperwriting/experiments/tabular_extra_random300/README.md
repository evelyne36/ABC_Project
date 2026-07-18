# Tabular Extra Random300 Candidate Pool

This directory is an isolated planning/inventory surface for additional tabular
datasets beyond the current TabPFN subset.

The rule is inventory first, filtering second:

- `candidate_inventory.csv` keeps all currently known candidates, including
  datasets not selected for the first batch.
- `selected_manifest.csv` records the current low-cost, MetaGB-plausible first
  and second batches.

Selection prioritizes:

1. low `rows * features^2` pressure;
2. low/mid-dimensional tabular structure;
3. plausible local one-feature linear response or conditional/predictive
   decoupling;
4. availability in the current workspace.

Datasets not selected remain in the inventory for later recovery if results are
insufficient.
