# Time-Series Extra Random300 Candidate Pool

This directory records additional time-series candidates before any new data
download or full benchmark run.

The rule is inventory first, filtering second:

- `candidate_inventory.csv` lists registered, available, and unverified
  candidates.
- `selected_manifest.csv` records the current first batch, second batch, and
  reserve choices.

Time-series experiments should keep the LSTF-style protocol already used by
`timeseries_metagb_adapter`:

1. chronological splits only;
2. scaler fit on train only;
3. supervised windows built from past values only;
4. no one-hot path by default;
5. compare XGBoost and MetaGB on identical window features.

The pressure proxy is approximate:

```text
pressure_proxy ~= timesteps * (seq_len * channels)^2
```

High-channel datasets such as `traffic`, `electricity`, and PEMS should start
with channel subsets, shorter `seq_len`, or 30-candidate smoke runs before any
Random300 commitment.
