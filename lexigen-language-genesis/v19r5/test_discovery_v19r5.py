from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise AssertionError(message)


def test_registry_identity():
    pre = load("V19R5_PRECOMMIT.json")
    registry = load("V19R5_REGISTRY.json")
    check(sha(HERE / "V19R5_REGISTRY.json") == pre["registry_sha256"], "registry hash")
    check(len(registry["discovery_task_ids"]) == 631, "discovery denominator")
    check(len(registry["validation_task_ids"]) == 244, "validation denominator")


def test_aborted_boundary():
    report = load("V19R5_ABORTED_RUN.json")
    check(report["status"] == "aborted_generator_nontermination", "abort status")
    check(report["blocking_discovery_index"] == 573, "blocking index")
    check(report["blocking_task_id"] == "e74e1818", "blocking task")
    check(report["observed_exact_complete_programs_through_checkpoint"] == 0, "program count")
    check(report["observed_structures_through_checkpoint"] == 0, "structure count")
    check(not report["claim_boundary"]["library_frozen"], "library claim")
    check(not report["claim_boundary"]["validation_started"], "validation claim")
    check(not report["validation_outputs_opened"], "validation outputs")


def test_cancelled_validator_isolated():
    path = HERE / "cancelled" / "validate_registry_v19r5.py"
    check(path.exists(), "cancelled validator missing")
    check(not (HERE / "V19R5_LIBRARY.json").exists(), "invented library exists")
    check(not (HERE / "V19R5_DISCOVERY_REPORT.json").exists(), "incomplete report exists")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
