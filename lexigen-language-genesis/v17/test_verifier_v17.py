from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from constructive_dsl_v17 import execute
from mutations_v17 import apply_output_mutation
from portable_constructive_dsl_v17 import execute_portable
from portable_verifier_v17 import verify_against_reference_portable
from verifier_grammar_v17 import verify_against_reference, verify_contract_integrity

HERE = Path(__file__).resolve().parent


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(gate: int) -> tuple[dict[str, Any], dict[str, Any]]:
    program = json.loads((HERE / "programs" / f"v17-program-{gate:02d}.json").read_text(encoding="utf-8"))
    contract = json.loads((HERE / "contracts" / f"v17-contract-{gate:02d}.json").read_text(encoding="utf-8"))
    return program, contract


FIXTURES = {
    1: ((0, 0, 0, 0, 0), (0, 3, 0, 4, 0), (0, 0, 0, 0, 0)),
    2: ((0, 1, 0), (2, 0, 3)),
    3: ((2, 0, 2), (1, 2, 3)),
}


def test_correct_outputs_accepted_by_both_verifiers() -> None:
    for gate in (1, 2, 3):
        program, contract = load(gate)
        source = FIXTURES[gate]
        primary = execute(program, source)
        portable = execute_portable(program, source)
        check(primary == portable, f"constructive runtimes disagree for gate {gate}")
        check(verify_against_reference(contract, program, source, primary, primary), f"primary verifier rejected gate {gate}")
        check(
            verify_against_reference_portable(contract, program, source, portable, portable),
            f"portable verifier rejected gate {gate}",
        )


def test_mutated_outputs_rejected_by_both_verifiers() -> None:
    for gate in (1, 2, 3):
        program, contract = load(gate)
        source = FIXTURES[gate]
        reference = execute(program, source)
        candidate = apply_output_mutation(reference, "flip_first_cell")
        check(candidate != reference, f"mutation was ineffective for gate {gate}")
        check(not verify_against_reference(contract, program, source, candidate, reference), f"primary admitted gate {gate}")
        check(
            not verify_against_reference_portable(contract, program, source, candidate, reference),
            f"portable admitted gate {gate}",
        )


def test_contract_tampering_is_rejected() -> None:
    program, contract = load(1)
    source = FIXTURES[1]
    reference = execute(program, source)
    tampered = copy.deepcopy(contract)
    tampered["program_sha256"] = "0" * 64
    check(not verify_contract_integrity(tampered), "primary integrity accepted tampering")
    check(not verify_against_reference(tampered, program, source, reference, reference), "primary accepted tampering")
    check(
        not verify_against_reference_portable(tampered, program, source, reference, reference),
        "portable accepted tampering",
    )


def test_contract_bindings_and_claim_boundary() -> None:
    for gate in (1, 2, 3):
        _program, contract = load(gate)
        check(contract["schema"] == "lexigen-v17-verifier-contract-v1", f"wrong schema gate {gate}")
        for name in (
            "program_sha256",
            "constructive_grammar_sha256",
            "portable_runtime_sha256",
            "demonstration_sha256",
            "verifier_grammar_sha256",
            "contract_sha256",
        ):
            check(isinstance(contract.get(name), str) and len(contract[name]) == 64, f"bad {name} gate {gate}")
        check(not contract["exact_digest_used"], f"learned exact digest gate {gate}")
        check(contract["soundness_anchor"] == {"name": "exact_digest", "mandatory": True}, f"missing anchor gate {gate}")


TESTS = [
    test_correct_outputs_accepted_by_both_verifiers,
    test_mutated_outputs_rejected_by_both_verifiers,
    test_contract_tampering_is_rejected,
    test_contract_bindings_and_claim_boundary,
]


def main() -> None:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"SUMMARY {len(TESTS)}/{len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
