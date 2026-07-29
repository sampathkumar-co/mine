from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V30_PRECOMMIT.json"
MANIFEST = HERE / "V30_GRAMMAR_MANIFEST.json"
AGGREGATOR = HERE / "aggregate_validation_v30.py"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def make_report(task_id: str, exact: bool = False):
    selected = {
        "structural_index": 0,
        "depth": 1,
        "nodes": 4,
        "ast_sha256": "a" * 64,
        "parameters": {"c0": 8},
        "concrete_program_sha256": "b" * 64,
    }
    return {
        "schema": "lexigen-v30-task-scan-v1",
        "task_id": task_id,
        "status": "completed",
        "accepted_examples": 6,
        "generation": {"attempts": 6, "timeouts": 0, "failures": 0, "failure_examples": []},
        "demonstration_sha256": "c" * 64,
        "precommit_sha256": sha256_file(PRECOMMIT),
        "grammar_sha256": "d" * 64,
        "grammar_manifest_sha256": sha256_file(MANIFEST),
        "enumeration": {
            "schema": "lexigen-v30-task-enumeration-v1",
            "task_nontrivial": True,
            "concrete_candidates_tested": 100,
            "runtime_invalid_candidates": 2,
            "identity_candidates_rejected": 3,
            "exact_candidate_count": 1 if exact else 0,
            "exact_candidates": [selected] if exact else [],
            "selected_candidate": selected if exact else None,
            "candidate_cap_reached": False,
        },
    }


def test_exact_denominator_and_success_accounting() -> None:
    precommit = load(PRECOMMIT)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reports = root / "reports"
        reports.mkdir()
        for index, task_id in enumerate(precommit["validation_task_ids"]):
            write(reports / f"task-{task_id}.json", make_report(task_id, exact=index == 0))
        output = root / "aggregate.json"
        subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--reports-root",
                str(reports),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = load(output)
        assert result["validation_task_count"] == 20
        assert result["completed_report_count"] == 20
        assert result["totals"]["concrete_candidates_tested"] == 2000
        assert result["heldout_synthesis_event_count"] == 1
        assert result["heldout_synthesis_demonstrated"] is True
        assert result["repeated_heldout_transfer_demonstrated"] is False
        assert result["fresh_validation_completed"] is False


def test_missing_identity_is_rejected() -> None:
    precommit = load(PRECOMMIT)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reports = root / "reports"
        reports.mkdir()
        for task_id in precommit["validation_task_ids"][:-1]:
            write(reports / f"task-{task_id}.json", make_report(task_id))
        completed = subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--reports-root",
                str(reports),
                "--output",
                str(root / "aggregate.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "missing report for frozen task" in completed.stderr


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
