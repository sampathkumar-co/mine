from __future__ import annotations

from portable_family_runtime import run_portable_instance
from rift4 import (
    adapt_family,
    build_cases,
    execute_family,
    exhaustive_fixed_language_search,
    run,
)


def test_family_invents_canonical_max_extension() -> None:
    demonstrations = build_cases([5], replicas=1)
    instance, tested = adapt_family(demonstrations)
    assert instance["stop"] == "repeat"
    assert instance["finalize"] == "canonical_max"
    assert instance["extension_proof"]["new_finalizer"] == "canonical_max"
    assert tested == 10


def test_frozen_language_is_exhaustively_inexpressive() -> None:
    solved, tested = exhaustive_fixed_language_search(build_cases([5], replicas=1))
    assert solved is False
    assert tested == 2880


def test_extension_transfers_and_is_portable() -> None:
    instance, _ = adapt_family(build_cases([5], replicas=1))
    for case in build_cases(range(8, 12), replicas=1):
        expected = case.independently_verified_target()
        assert execute_family(instance, case) == expected
        assert run_portable_instance(instance, case.step, case.seed) == expected


def test_rift4_internal_gate(tmp_path) -> None:
    report = run(tmp_path)
    assert all(report["gate"].values())
    assert report["fixed_language_found_solution"] is False
    assert report["family_transfer_accuracy"] == 1.0
    assert report["portable_interpreter_accuracy"] == 1.0
    assert report["ablation_accuracy"] < 1.0
