"""LightGBM neighbour adapters kept OUTSIDE baselines/ on purpose.

`baselines/provenance.py:adapter_fingerprint()` hashes every file under
`baselines/` and `compat/`, and `scripts/run_nested.py` puts that fingerprint
into the resume identity of every panel.  Adding LightGBM next door would
invalidate the resume identity of panels that are still running, so the two
LightGBM variants live here and get their own runner
(`scripts/run_nested_lgb.py` / `scripts/run_task_lgb.py`), which is a
line-for-line copy of the original protocol with only the method dispatch and
the adapter digest swapped.
"""
from .adapter import METHODS, DEFAULTS, LightGBMAdapter, make_regressor

__all__ = ["METHODS", "DEFAULTS", "LightGBMAdapter", "make_regressor"]
