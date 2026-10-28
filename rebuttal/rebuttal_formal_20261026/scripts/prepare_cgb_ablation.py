#!/usr/bin/env python3
"""Freeze dim10 ablation arrays/backbones and exact archived split checks; zero fits."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "rebuttal_formal_20261026"
SOURCE = ROOT / "rebuttal_execution_20261025/source/exp"
ARCHIVE = ROOT / "aaai_FullCodeProject-check_codereview0701/datacollection0702/0716dataAddition/metagb_ablation_syn/syn_formal_formal16_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def main():
    import numpy as np
    sys.path.insert(0, str(SOURCE))
    from metagb_ablation.run_ablation import load_task_data, nested_splits
    from metagb_ablation.io_utils import canonical_hash, index_hash
    protocol = SimpleNamespace(**json.loads((ROOT / "rebuttal_server_config_20261025/historical/ablation_protocol.json").read_text())["protocol"])
    backbones = json.loads((ROOT / "rebuttal_server_config_20261025/historical/ablation_synthetic_backbones.json").read_text())
    manifests = {r["task_id"]: r for r in json.loads((ROOT / "rebuttal_server_config_20261025/historical/ablation_split_manifest.json").read_text())["entries"]}
    mechanisms = ["feature_switch", "categorical_regime_slope", "regime_slope", "curved_control", "sparse_multiplicative", "regime_logit", "pure_step_control"]
    # Resolve only the archived JSON task shards, not similarly named status files.
    shards = {p.stem: p for p in ARCHIVE.rglob("synthetic__*__dim10__seed*__out*.json")}
    e0, n2, report = [], [], []
    for scenario in mechanisms:
        for seed in protocol.synthetic_data_seeds:
            case = f"{scenario}__dim10__seed{seed}"
            first_id = f"synthetic__{case}__out1"
            first = json.loads(shards[first_id].read_text())
            task = dict(first["fold_rows"][0])
            X, y, n_classes, data_hash = load_task_data(task, protocol, "ordinal")
            outer, inner = nested_splits(task_type=task["task_type"], X=X, y=y,
                outer_folds=5, inner_folds=4, split_seed=protocol.cv_seed + seed)
            arrays = {"X": X, "y": y}
            for fold, (train, test) in enumerate(outer, 1):
                arrays[f"outer{fold}_train"] = train
                arrays[f"outer{fold}_test"] = test
            relative = Path("inputs/cgb_ablation") / case / "data.npz"
            target = FORMAL / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            checksum = digest(target)
            backbone = backbones[f"{scenario}::dim10"]["params"]
            for fold, (train, test) in enumerate(outer, 1):
                task_id = f"synthetic__{case}__out{fold}"
                source = shards[task_id]
                old = json.loads(source.read_text())
                reference = manifests[task_id]
                assert data_hash == reference["data_hash"]
                assert index_hash(train) == reference["outer_train_hash"]
                assert index_hash(test) == reference["outer_test_hash"]
                inner_entries = [{"inner_fold": j, "train_hash": index_hash(train[it]),
                    "validation_hash": index_hash(train[iv])} for j, (it, iv) in enumerate(inner[fold - 1], 1)]
                assert inner_entries == reference["inner"]
                assert canonical_hash(backbone) == old["backbone_hash"]
                observed = sorted((r for r in old["predictions"] if r["variant_id"] == "xgb"), key=lambda r: r["sample_id"])
                assert [r["sample_id"] for r in observed] == test.tolist()
                assert np.array_equal(np.asarray([r["y_true"] for r in observed]), y[test])
                rows = {r["variant_id"]: r for r in old["fold_rows"]}
                endpoints = ["xgb", "bias_p00_sa0", "split_p10_sa0", "full_p10_sa0"]
                payload = {"task_id": task_id, "scenario": scenario, "dimension": 10, "data_seed": seed,
                    "outer_fold": fold, "task_type": task["task_type"], "n_classes": n_classes,
                    "model_seed": protocol.model_seed, "data_npz": str(relative), "data_npz_sha256": checksum,
                    "backbone": backbone, "backbone_hash": old["backbone_hash"], "archived_split": reference,
                    "archive_shard": str(source.relative_to(ROOT)), "archive_shard_sha256": digest(source),
                    "historical_metrics": {v: float(rows[v]["test_primary_metric"]) for v in endpoints},
                    "primary_metric": rows["xgb"]["primary_metric"], "data_hash_scope": "archived data_hash identifies generator arguments; frozen NPZ additionally hashes actual arrays"}
                if seed == 1001 and fold == 1:
                    predictions = {}
                    for variant in endpoints:
                        ordered = sorted((r for r in old["predictions"] if r["variant_id"] == variant), key=lambda r: r["sample_id"])
                        assert [r["sample_id"] for r in ordered] == test.tolist()
                        assert np.array_equal(np.asarray([r["y_true"] for r in ordered]), y[test])
                        predictions[variant] = {"prediction": [r["y_pred"] for r in ordered]}
                        if task["task_type"] != "regression":
                            predictions[variant]["probability"] = [json.loads(r["probabilities_json"]) for r in ordered]
                    payload["historical_predictions"] = predictions
                    e0.extend({"task": f"configs/cgb_tasks/{task_id}.json", "variant": v} for v in endpoints)
                save(FORMAL / f"configs/cgb_tasks/{task_id}.json", payload)
                n2.append({"task": f"configs/cgb_tasks/{task_id}.json", "variant": "leafonly"})
                report.append({"task_id": task_id, "outer_and_inner_split_hashes_match": True, "backbone_hash_matches": True,
                    "all_test_targets_exactly_match_archive": True})
    save(FORMAL / "configs/R0_E0_queue.json", e0)
    save(FORMAL / "configs/N2_leafonly_queue.json", n2)
    save(FORMAL / "results/cgb_input_preflight.json", {"status": "inputs_ready_runtime_pending", "model_fit_calls": 0,
        "tasks": report, "R0_E0_total_fits": len(e0), "R0_E0_cgb_fits": 21, "R0_E0_xgb_fits": 7,
        "N2_leafonly_cgb_fits": len(n2), "actual_arrays_hashed": True,
        "generator_source_sha256": digest(SOURCE / "synthetic_metagb/scenarios.py"),
        "limitation": "Exact split/backbone provenance is verified; prediction compatibility requires Linux E0"})
    print(json.dumps({"matched_tasks": len(report), "R0_E0_fits": len(e0), "N2_fits": len(n2), "model_fit_calls": 0}))


if __name__ == "__main__":
    main()
