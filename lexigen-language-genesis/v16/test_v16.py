from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
V15 = HERE.parent / "v15"
for folder in (HERE, V15):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cosynthesize_verifier_v16 import build_mutation_cases, synthesize_contract
from ir_runtime_v15 import as_grid, execute
from mutations_v16 import apply_output_mutation, generate_mutations, mutation_manifest
from portable_verifier_v16 import GRAMMAR_SHA256 as PORTABLE_GRAMMAR_SHA256
from portable_verifier_v16 import verify_output_portable
from verifier_grammar_v16 import GRAMMAR_SHA256, screening_holds, verify_output

PROGRAM = {
    "op": "recolour",
    "grid": {"op": "input"},
    "mapping": {"2": 4},
}
SOURCE_A = as_grid([[0, 2, 0], [2, 0, 2]])
TARGET_A = as_grid([[0, 4, 0], [4, 0, 4]])
SOURCE_B = as_grid([[2, 2], [0, 2]])
TARGET_B = as_grid([[4, 4], [0, 4]])
EXAMPLES = [(SOURCE_A, TARGET_A), (SOURCE_B, TARGET_B)]


def test_mutation_generation_is_deterministic() -> None:
    first = generate_mutations(PROGRAM)
    second = generate_mutations(PROGRAM)
    assert first == second
    assert first
    manifest = mutation_manifest(PROGRAM)
    assert any(item["kind"] == "ast" for item in manifest)
    assert any(item["kind"] == "output" for item in manifest)


def test_output_mutations_are_semantic() -> None:
    changed = apply_output_mutation(TARGET_A, "flip_first_cell")
    assert changed != TARGET_A
    assert len(changed) == len(TARGET_A)
    cropped = apply_output_mutation(TARGET_A, "crop_last_column")
    assert len(cropped[0]) == len(TARGET_A[0]) - 1


def test_contract_is_deterministic_and_bound() -> None:
    first, cases, _ = synthesize_contract(PROGRAM, EXAMPLES)
    second, _, _ = synthesize_contract(PROGRAM, EXAMPLES)
    assert first == second
    assert cases
    assert first["soundness_anchor"] == {"name": "exact_digest", "mandatory": True}
    assert verify_output(first, PROGRAM, SOURCE_A, TARGET_A)
    assert verify_output_portable(first, PROGRAM, SOURCE_A, TARGET_A)

    mutated_program = deepcopy(PROGRAM)
    mutated_program["mapping"] = {"2": 5}
    assert not verify_output(first, mutated_program, SOURCE_A, TARGET_A)


def test_screening_rejects_training_mutations() -> None:
    contract, cases, _ = synthesize_contract(PROGRAM, EXAMPLES)
    assert contract["shape_only_survivors"] > 0
    assert all(
        not screening_holds(
            contract,
            case["source"],
            case["candidate"],
            case["reference"],
        )
        for case in cases
    )


def test_soundness_anchor_rejects_tampering() -> None:
    contract, _, _ = synthesize_contract(PROGRAM, EXAMPLES)
    wrong = apply_output_mutation(TARGET_A, "flip_centre_cell")
    assert not verify_output(contract, PROGRAM, SOURCE_A, wrong)
    assert not verify_output_portable(contract, PROGRAM, SOURCE_A, wrong)

    tampered = deepcopy(contract)
    tampered["predicates"] = []
    assert not verify_output(tampered, PROGRAM, SOURCE_A, TARGET_A)
    assert not verify_output_portable(tampered, PROGRAM, SOURCE_A, TARGET_A)


def test_primary_and_portable_grammar_match() -> None:
    assert GRAMMAR_SHA256 == PORTABLE_GRAMMAR_SHA256


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} v16 tests passed")


if __name__ == "__main__":
    main()
