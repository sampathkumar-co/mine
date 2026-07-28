from __future__ import annotations

from rift4_revision2 import build_cases
from rift5 import (
    build_verifier_certificate,
    execute_artifact,
    run,
    synthesize_finalizer,
)


def test_finalizer_ast_is_synthesized() -> None:
    ast, tested = synthesize_finalizer(build_cases([5, 6], replicas=2))
    assert ast["op"] == "select_extreme"
    assert ast["mode"] == "max"
    assert tested >= 1


def test_generated_verifier_certificate_is_complete() -> None:
    ast, _ = synthesize_finalizer(build_cases([5, 6], replicas=2))
    certificate = build_verifier_certificate(ast)
    assert certificate["all_correct"] is True
    assert certificate["row_count"] == 36
    assert len(certificate["rows_sha256"]) == 64


def test_verified_artifact_transfers(tmp_path) -> None:
    report = run(tmp_path)
    artifact = report["artifact"]
    assert all(report["gate"].values())
    for case in build_cases(range(8, 12), replicas=1):
        assert execute_artifact(artifact, case) == case.independently_verified_target()
