from pathlib import Path
import json
import numpy as np

from baselines.protocol import prepare_dataset, load_protocol


def test_synthetic_outer_and_inner_indices_match_archived_manifest(tmp_path):
    root=Path(__file__).resolve().parents[2]
    output=tmp_path/"regime_slope"
    prepare_dataset("regime_slope",output,root)
    X,y,folds,manifest=load_protocol(output)
    assert X.shape==(3000,10) and y.shape==(3000,)
    assert manifest["archived_split_entries_matched"]==5
    assert len(folds)==5 and all(len(f["inner"])==4 for f in folds)
    for fold in folds:
        assert not set(fold["train"]) & set(fold["test"])
        for inner in fold["inner"]:
            assert not set(inner["train"]) & set(inner["validation"])
            assert not set(fold["test"]) & (set(inner["train"])|set(inner["validation"]))
            assert set(inner["train"])|set(inner["validation"])==set(fold["train"])


def test_moneyball_retains_full_historical_rows_and_no_new_hpo(tmp_path):
    root=Path(__file__).resolve().parents[2]
    prepare_dataset("moneyball",tmp_path/"moneyball",root)
    X,y,folds,manifest=load_protocol(tmp_path/"moneyball")
    # Historical logs report 42 encoded features; 14 is the raw CSV count.
    assert X.shape==(114,42) and y.shape==(114,)
    assert X.dtype==np.float32 and y.dtype==np.float32
    assert manifest["model_fit_calls"]==0
    assert sorted(np.concatenate([f["test"] for f in folds]).tolist())==list(range(114))
