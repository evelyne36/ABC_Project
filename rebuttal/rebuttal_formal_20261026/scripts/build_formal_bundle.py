#!/usr/bin/env python3
"""Package only the task's frozen source, inputs, helpers and authored documents."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "rebuttal_formal_20261026"
SKIP = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "build-rebuttal"}


def main():
    files = set()
    paths = [FORMAL, ROOT / "rebuttal_execution_20261025/native/controlled"]
    paths += [ROOT / "rebuttal_execution_20261025/source/exp" / p for p in ["random300", "metagb_ablation", "synthetic_metagb"]]
    paths += [ROOT / "rebuttal_server_config_20261025/source/exp" / p for p in ["random600", "timeseries_extra_random300"]]
    paths += [ROOT / "rebuttal_server_config_20261025/inputs"]
    for directory in paths:
        for path in directory.rglob("*"):
            if not path.is_file() or path.is_symlink() or any(part in SKIP for part in path.relative_to(ROOT).parts): continue
            if path.suffix in {".pyc", ".dylib", ".so", ".log"} or path.name.endswith(".tar.gz"): continue
            if path.name.endswith(".tar.gz.sha256"): continue
            if path.name in {"bundle_status.json", "deployment_manifest.json"}: continue
            files.add(path)
    for name in ["ts_submitted_split_manifest.json", "ts_selected_models.json", "ablation_protocol.json", "ablation_synthetic_backbones.json", "ablation_split_manifest.json"]:
        files.add(ROOT / "rebuttal_server_config_20261025/historical" / name)
    for name in ["param.h", "updater_metagb_hist.cc"]:
        files.add(ROOT / "rebuttal_execution_20261025/native/historical/src/tree" / name)
    for item in json.loads((FORMAL / "configs/R0_E0_queue.json").read_text()):
        task = json.loads((FORMAL / item["task"]).read_text())
        files.add(ROOT / task["archive_shard"])
    registry = json.loads((ROOT / "rebuttal_server_config_20261025/inputs/ts_dataset_registry.json").read_text())
    expected = json.loads((ROOT / "rebuttal_server_config_20261025/inputs/ts_expected_data_hashes.json").read_text())
    for dataset in expected:
        files.add(ROOT / "aaai_FullCodeProject-check_codereview0701/paper_writing/timeseries_extra_random300/data/raw" / registry[dataset]["raw_path"])
    files.add(ROOT / "rebuttal_baselines_20261025/baseline_linux_bundle.tar.gz")
    files.add(ROOT / "rebuttal_server_config_20261025/scripts/run_ts_reference_panel.py")
    # Portable research context and archived comparisons for the receiving AI.
    files.update((ROOT / "aaai2027提交原件").glob("*.pdf"))
    files.update(ROOT.glob("Reviewers_*.pdf"))
    files.update(ROOT.glob("myself-*.docx"))
    files.add(ROOT / "Rebuttal讨论会1025纪要.md")
    for suffix in ["*.md", "*.txt", "*.docx"]:
        files.update((ROOT / "Xia'sSuggestions").glob(suffix))
    for name in ["已有证据复用审计_E0-E6.md", "已有结果复用审计_数据.json", "近邻baseline仓库核查_20261025.md", "近邻baseline仓库清单.csv", "审稿问题台账.csv"]:
        files.add(ROOT / "rebuttal_plan_20261025" / name)
    archive = ROOT / "aaai_FullCodeProject-check_codereview0701/datacollection0702"
    main = archive / "synthetic_main1020full_random300"
    baseline = archive / "0716dataAddition/synbaseline_lgbm0716/synthetic_baseline_random300_seed42_20260718_052755"
    for directory, methods in [(main, ["current_metagb", "xgb_cpu"]), (baseline, ["lightgbm"])]:
        for scenario in ["regime_slope", "feature_switch", "sparse_multiplicative", "curved_control"]:
            for method in methods:
                for fold in range(1, 6):
                    files.add(directory / f"tasks/{scenario}__dim10__seed42__out{fold}__{method}.json")
        for name in ["split_manifest.json", "protocol.json", "candidate_configs.json"]:
            files.add(directory / name)
    audit = json.loads((ROOT / "rebuttal_plan_20261025/已有结果复用审计_数据.json").read_text())
    for dataset in audit["real_panel"].values():
        for model in dataset.values():
            for record in model["records"]:
                files.add(ROOT / record["path"])
    rows = [{"path": str(p.relative_to(ROOT)), "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    manifest = json.dumps({"files": rows, "scope": "deployment inputs; Linux runtime checks pending"}, indent=2).encode()
    package = FORMAL / "formal_linux_bundle.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        for p in sorted(files): tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        info = tarfile.TarInfo("rebuttal_formal_20261026/deployment_manifest.json"); info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    report = {"path": str(package.relative_to(ROOT)), "files": len(files), "bytes": package.stat().st_size,
        "sha256": hashlib.sha256(package.read_bytes()).hexdigest()}
    (FORMAL / "results/bundle_status.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__": main()
