from __future__ import annotations

from rift1 import (
    build_cases,
    execute_artifact,
    infer_instruction_inventory,
    run,
    solves,
    synthesize,
)


def test_instruction_inventory_is_inferred_per_mechanism() -> None:
    closure = infer_instruction_inventory(build_cases("closure", range(4, 7), replicas=1))
    union = infer_instruction_inventory(build_cases("trajectory_union", range(4, 7), replicas=1))
    cycle = infer_instruction_inventory(build_cases("two_cycle_canonical", range(4, 7), replicas=1))
    assert "IF_EQUAL" in closure
    assert "ACCUMULATE_CURRENT" in union
    assert "RETURN_CANONICAL_PAIR" in cycle
    assert len({closure, union, cycle}) == 3


def test_each_mechanism_synthesizes_and_transfers() -> None:
    for mechanism in ("closure", "trajectory_union", "two_cycle_canonical"):
        diagnostic = build_cases(mechanism, range(4, 7), replicas=1)
        transfer = build_cases(mechanism, range(7, 10), replicas=1)
        artifact = synthesize(diagnostic)
        assert artifact["synthesis"]["programs_tested"] > 1
        assert solves(artifact, transfer)
        for case in transfer:
            assert execute_artifact(artifact, case.step, case.seed) == case.independently_verified_target()


def test_rift1_report_is_exact(tmp_path) -> None:
    report = run(tmp_path)
    assert set(report["mechanisms"]) == {
        "closure",
        "trajectory_union",
        "two_cycle_canonical",
    }
    for result in report["mechanisms"].values():
        assert result["synthesized_transfer"]["accuracy"] == 1.0
        assert result["artifact"]["provenance"]["limitation"] == "opcode meanings remain human supplied"
