from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def fresh_gate(passed: bool) -> dict:
    passed_cases = 100 if passed else 99
    mismatches = 0 if passed else 1
    return {
        "schema": "lexigen-v31-immediate-fresh-gate-v1",
        "color": 5,
        "case_count": 100,
        "totals": {
            "requested_cases": 100,
            "generated_cases": 100,
            "generation_timeouts": 0,
            "generation_errors": 0,
            "primary_runtime_errors": 0,
            "independent_runtime_errors": 0,
            "runtime_disagreements": 0,
            "target_mismatches": mismatches,
            "verifier_rejections": 0,
            "passed_cases": passed_cases,
        },
        "passed": passed,
        "case_records": [],
    }


def make_report(task_id: str, precommit_sha: str, status: str) -> dict:
    exact_colors = []
    if status in {"fresh_pass", "fresh_fail"}:
        exact_colors = [5]
    elif status == "ambiguous":
        exact_colors = [4, 5]
    gate = None
    if status == "fresh_pass":
        gate = fresh_gate(True)
    elif status == "fresh_fail":
        gate = fresh_gate(False)
    return {
        "schema": "lexigen-v31-task-report-v1",
        "task_id": task_id,
        "precommit_sha256": precommit_sha,
        "accepted_examples": 6,
        "generation": {
            "attempts": 6,
            "accepted": 6,
            "timeouts": 0,
            "failures": 0,
            "failure_examples": [],
        },
        "demonstration_sha256": "0" * 64,
        "candidate_count": 10,
        "status": status,
        "exact_colors": exact_colors,
        "invalid_candidate_count": 0,
        "identity_candidate_count": 0,
        "selected_color": 5 if gate else None,
        "fresh_gate": gate,
    }


def run_aggregate(root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "aggregate_v31.py"),
            "--reports-root", str(root),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_exact_denominator_and_claim_logic() -> None:
    precommit = load(HERE / "V31_PRECOMMIT.json")
    ids = precommit["fresh_identity_selection"]["task_ids"]
    precommit_sha = __import__("hashlib").sha256(
        (HERE / "V31_PRECOMMIT.json").read_bytes()
    ).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "reports"
        root.mkdir()
        for index, task_id in enumerate(ids):
            status = "no_program"
            if index == 0:
                status = "fresh_pass"
            elif index == 1:
                status = "fresh_fail"
            elif index == 2:
                status = "ambiguous"
            write(root / f"task-{task_id}.json", make_report(task_id, precommit_sha, status))
        output = Path(temporary) / "aggregate.json"
        completed = run_aggregate(root, output)
        assert completed.returncode == 0, completed.stderr
        report = load(output)
        assert report["completed_report_count"] == 64
        assert report["status_counts"]["fresh_pass"] == 1
        assert report["status_counts"]["fresh_fail"] == 1
        assert report["status_counts"]["ambiguous"] == 1
        assert report["repeated_public_task_level_transfer_demonstrated"] is True
        assert report["multi_identity_motif_recurrence_demonstrated"] is False
        assert report["world_level_breakthrough"] is False


def test_missing_identity_is_rejected() -> None:
    precommit = load(HERE / "V31_PRECOMMIT.json")
    ids = precommit["fresh_identity_selection"]["task_ids"]
    precommit_sha = __import__("hashlib").sha256(
        (HERE / "V31_PRECOMMIT.json").read_bytes()
    ).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "reports"
        root.mkdir()
        for task_id in ids[:-1]:
            write(root / f"task-{task_id}.json", make_report(task_id, precommit_sha, "no_program"))
        completed = run_aggregate(root, Path(temporary) / "aggregate.json")
        assert completed.returncode != 0
        assert "denominator mismatch" in completed.stderr


def main() -> None:
    tests = sorted(
        (name, value) for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
