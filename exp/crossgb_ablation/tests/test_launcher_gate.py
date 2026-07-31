from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "crossgb_ablation" / "scripts" / "launch_ablation.sh"


def _write_gate_fixture(output_root: Path, run_tag: str) -> Path:
    result_dir = output_root / f"syn_formal_{run_tag}"
    result_dir.mkdir(parents=True)
    (result_dir / "completeness_audit.json").write_text(
        json.dumps({"passed": True, "issues": []}),
        encoding="utf-8",
    )
    (result_dir / "protocol.json").write_text(
        json.dumps(
            {
                "config_hash": "config",
                "variant_hash": "variants",
                "variant_count": 16,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "execution_manifest.json").write_text(
        json.dumps({"execution_hash": "execution"}),
        encoding="utf-8",
    )
    (result_dir / "SYNTHETIC_CONCLUSION.md").write_text(
        (
            "# Synthetic conclusion\n\n"
            "主消融、pred_fraction、score_agg、负对照和不确定性均已分析。\n\n"
            "SYNTHETIC_ANALYSIS_STATUS: COMPLETE\n\n"
            "REAL_STAGE_DECISION: GO\n"
        ),
        encoding="utf-8",
    )
    return result_dir


def _launcher_env(tmp_path: Path, run_tag: str) -> dict[str, str]:
    return {
        **os.environ,
        "RUN_TAG": run_tag,
        "OUTPUT_ROOT": str(tmp_path / "results"),
        "LOG_DIR": str(tmp_path / "logs"),
        "RUN_PREFLIGHT": "0",
    }


def test_real_gate_binds_approval_to_conclusion_hash(tmp_path: Path) -> None:
    run_tag = "gate"
    output_root = tmp_path / "results"
    result_dir = _write_gate_fixture(output_root, run_tag)
    env = _launcher_env(tmp_path, run_tag)

    approved = subprocess.run(
        ["bash", str(LAUNCHER), "approve-synthetic"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert approved.returncode == 0, approved.stderr
    approval = json.loads(
        (result_dir / "synthetic_analysis_approval.json").read_text(
            encoding="utf-8"
        )
    )
    assert approval["variant_count"] == 16
    assert approval["variant_hash"] == "variants"

    conclusion_path = result_dir / "SYNTHETIC_CONCLUSION.md"
    conclusion_path.write_text(
        conclusion_path.read_text(encoding="utf-8") + "\nchanged\n",
        encoding="utf-8",
    )
    blocked = subprocess.run(
        ["bash", str(LAUNCHER), "start", "real"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked.returncode != 0
    assert "conclusion_sha256" in blocked.stderr


def test_approval_rejects_reduced_14_variant_protocol(tmp_path: Path) -> None:
    run_tag = "reduced"
    output_root = tmp_path / "results"
    result_dir = _write_gate_fixture(output_root, run_tag)
    protocol_path = result_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["variant_count"] = 14
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    rejected = subprocess.run(
        ["bash", str(LAUNCHER), "approve-synthetic"],
        cwd=ROOT,
        env=_launcher_env(tmp_path, run_tag),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "16-variant" in rejected.stderr
