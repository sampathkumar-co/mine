from __future__ import annotations

import json

from artifact_runtime import execute_artifact
from prototype import diagnose_representation_failure, invent_artifact, run
from rift0 import build_cases, build_report, validate_report


def test_benchmark_separates_bounded_language_from_fixed_point() -> None:
    report = build_report()
    validate_report(report)
    assert report["bounded_train"]["accuracy"] == 1.0
    assert report["bounded_transfer"]["accuracy"] == 0.0
    assert report["oracle_transfer"]["accuracy"] == 1.0


def test_diagnosis_requires_variable_depth_failures() -> None:
    diagnosis = diagnose_representation_failure(build_cases(range(4, 7), replicas=1))
    assert diagnosis["representation_failure_supported"] is True
    assert diagnosis["required_rounds"] == [4, 5, 6]


def test_artifact_executes_in_separate_runtime() -> None:
    cases = build_cases([9], replicas=1)
    artifact = invent_artifact(diagnose_representation_failure(build_cases(range(4, 7), replicas=1)))
    for case in cases:
        assert execute_artifact(artifact, case.step, case.seed) == case.independently_verified_target()


def test_prototype_writes_reproducible_artifacts(tmp_path) -> None:
    report = run(tmp_path)
    artifact = json.loads((tmp_path / "invented-language-artifact.json").read_text(encoding="utf-8"))
    persisted = json.loads((tmp_path / "prototype-report.json").read_text(encoding="utf-8"))
    assert artifact["schema"] == "lexigen-language-artifact-v1"
    assert report["artifact_transfer"]["accuracy"] == 1.0
    assert persisted["status"] == "research scaffold; no novelty or breakthrough claim"
