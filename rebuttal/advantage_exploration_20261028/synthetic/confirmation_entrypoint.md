# A06 confirmation entry point

`prepare_confirmation.py` is separate from the frozen DEV generator and manifest. It accepts no case or seed overrides. It permits only `A06_two_sparse_modules` and reserved seeds 111001–111005, calls the unchanged DEV `base_data` and `signal_for`, and reuses the same `random_split` and `save` functions. The written confirmation manifest records the complete freeze file hash; existing confirmation output is never overwritten.

The required root file is `rebuttal/advantage_exploration_20261028/confirmation_freeze.json`. Before any data draw, the entry point verifies that DEV generator/source hashes remain frozen and that this file contains:

- `frozen_utc`: timezone-aware ISO timestamp after DEV preregistration and no later than the current time.
- `cases`: exactly `["A06_two_sparse_modules", "gas_turbine_nox"]` (order immaterial).
- `methods`: all six methods, exactly once: CrossGB, XGB, LGB_constant, LGB_linear, PL, LinXGBoost.
- `choices`: a mapping `choices[case][method]` for every case/method pair. Each choice must contain a nonempty `params` object and matching `params_sha256`.

Parameter hashes are SHA-256 of UTF-8 `json.dumps(params, sort_keys=True, separators=(',', ':'), allow_nan=False)`. Additional fields such as candidate index, development validation evidence, selection rule, and pool/source hashes may be included. This entry point validates complete choices; the root task is responsible for confirming the completed 20-candidate HPO before writing this freeze.

The default mode checks the real freeze without drawing or writing any data (explicit `--check-only` is also accepted):

```sh
python rebuttal/advantage_exploration_20261028/synthetic/prepare_confirmation.py
```

After the root task has completed HPO and written the complete freeze, generate the five confirmation datasets:

```sh
python rebuttal/advantage_exploration_20261028/synthetic/prepare_confirmation.py --generate
```

Output is exclusively `synthetic/confirmation/data/` (five NPZ plus JSON sidecars) and `synthetic/confirmation/manifest.json`. No other synthetic cases, DEV artifacts, or models are written. The entry point does not fit or evaluate models and prints no held-out label summaries.
