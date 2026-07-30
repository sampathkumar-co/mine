from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V32_PRECOMMIT.json"
AGGREGATOR = HERE / "aggregate_v32.py"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def base_report(task_id: str, precommit_sha: str) -> dict[str, Any]:
    return {
        "schema": "lexigen-v32-task-report-v1",
        "task_id": task_id,
        "precommit_sha256": precommit_sha,
        "v30_precommit_sha256": "v30",
        "grammar_sha256": "grammar",
        "grammar_manifest_sha256": "manifest",
        "accepted_examples": 6,
        "generation": {
            "attempts": 6,
            "accepted": 6,
            "timeouts": 0,
            "failures": 0,
            "failure_examples": [],
        },
        "demonstration_sha256": "demo",
        "status": "no_program",
        "enumeration": {
            "schema": "lexigen-v30-task-enumeration-v1",
            "task_nontrivial": True,
            "concrete_candidates_tested": 127596,
            "runtime_invalid_candidates": 1000,
            "identity_candidates_rejected": 10,
            "exact_candidate_count": 0,
            "exact_candidates": [],
            "selected_candidate": None,
            "candidate_cap_reached": False,
        },
        "selected_candidate": None,
        "fresh_gate": None,
    }


def make_fresh_pass(report: dict[str, Any]) -> None:
    selected = {
        "structural_index": 4,
        "depth": 1,
        "nodes": 4,
        "ast_sha256": "ast",
        "parameters": {"c0": 5},
        "concrete_program_sha256": "program",
    }
    report["status"] = "fresh_pass"
    report["enumeration"]["exact_candidate_count"] = 2
    report["enumeration"]["exact_candidates"] = [selected, selected]
    report["enumeration"]["selected_candidate"] = selected
    report["selected_candidate"] = selected
    report["fresh_gate"] = {
        "schema": "lexigen-v32-immediate-fresh-gate-v1",
        "case_count": 100,
        "selected_ast_sha256": "ast",
        "parameters": {"c0": 5},
        "passed": True,
        "totals": {
            "requested_cases": 100,
            "generated_cases": 100,
            "generation_timeouts": 0,
            "generation_errors": 0,
            "primary_runtime_errors": 0,
            "independent_runtime_errors": 0,
            "runtime_disagreements": 0,
            "target_mismatches": 0,
            "passed_cases": 100,
        },
        "case_records": [],
    }


def run_aggregate(reports: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(AGGREGATOR),
            "--reports-root",
            str(reports),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def build_reports(root: Path, fresh_passes: int = 0) -> None:
    precommit = load(PRECOMMIT)
    precommit_sha = __import__("hashlib").sha256(PRECOMMIT.read_bytes()).hexdigest()
    for index, task_id in enumerate(precommit["fresh_identity_selection"]["task_ids"]):
        report = base_report(task_id, precommit_sha)
        if index < fresh_passes:
            make_fresh_pass(report)
        write(root / f"task-{task_id}.json", report)


def test_zero_pass_denominator() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        reports = root / "reports"
        reports.mkdir()
        output = root / "aggregate.json"
        build_reports(reports, fresh_passes=0)
        completed = run_aggregate(reports, output)
        assert completed.returncode == 0, completed.stderr
        result = load(output)
        assert result["completed_report_count"] == 64
        assert result["status_counts"]["no_program"] == 64
        assert result["fresh_pass_count"] == 0
        assert result["second_fresh_validated_identity_demonstrated"] is False
        assert result["multi_identity_public_transfer_demonstrated"] is False
        assert result["world_level_breakthrough"] is False


def test_claim_ladder_for_one_and_two_passes() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for pass_count in (1, 2):
            reports = root / f"reports-{pass_count}"
            reports.mkdir()
            output = root / f"aggregate-{pass_count}.json"
            build_reports(reports, fresh_passes=pass_count)
            completed = run_aggregate(reports, output)
            assert completed.returncode == 0, completed.stderr
            result = load(output)
            assert result["fresh_pass_count"] == pass_count
            assert result["second_fresh_validated_identity_demonstrated"] is True
            assert result["multi_identity_public_transfer_demonstrated"] is (pass_count >= 2)
            assert result["outside_human_reproduction_completed"] is False
            assert result["world_level_breakthrough"] is False


def test_missing_identity_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        reports = root / "reports"
        reports.mkdir()
        output = root / "aggregate.json"
        build_reports(reports)
        first = sorted(reports.glob("task-*.json"))[0]
        first.unlink()
        completed = run_aggregate(reports, output)
        assert completed.returncode != 0
        assert "report denominator mismatch" in completed.stderr


def main() -> None:
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
