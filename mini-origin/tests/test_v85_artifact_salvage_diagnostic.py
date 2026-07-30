from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mini_origin import pmlb_development_v85 as development


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = (
    PROJECT_ROOT
    / "research-diagnostics"
    / "v85-authoritative-salvage-input"
)


@pytest.mark.skipif(shutil.which("rustc") is None, reason="Rust compiler unavailable")
def test_preserved_v85_artifact_rust_replay_and_final_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "results" / "v85-salvage"
    work.mkdir(parents=True)
    states = work / "states.txt"
    reference = work / "python-reference.json"
    shutil.copy2(INPUT_ROOT / "states.txt", states)
    shutil.copy2(INPUT_ROOT / "python-reference.json", reference)

    monkeypatch.chdir(tmp_path)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "salvage_v85_reference.py")],
        check=True,
    )

    solver = work / ("response_cost_lower_bound_v85.exe" if os.name == "nt" else "response_cost_lower_bound_v85")
    subprocess.run(
        [
            "rustc",
            "-C",
            "opt-level=3",
            "-C",
            "debuginfo=0",
            str(PROJECT_ROOT / "compiled" / "response_cost_lower_bound_v66.rs"),
            "-o",
            str(solver),
        ],
        check=True,
    )
    rust_results = work / "rust-results.json"
    subprocess.run(
        [
            str(solver),
            "--input",
            str(states),
            "--output",
            str(rust_results),
        ],
        check=True,
    )

    evidence_path = work / "evidence.json"
    evidence = development.validate(reference, rust_results, evidence_path)
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence == persisted
    assert evidence["development_gate"] is True
    assert evidence["contributing_dataset_count"] == 7
    assert evidence["base_state_count"] == 84
    assert evidence["profiled_state_count"] == 252
    assert evidence["rust_mismatch_count"] == 0
    assert evidence["plain_bounded_objective_mismatch_count"] == 0
    assert evidence["current_bounded_plan_mismatch_count"] == 0
    assert evidence["bounded_expansion_regression_count"] == 0
