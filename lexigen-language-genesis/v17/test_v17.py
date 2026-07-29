from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from constructive_dsl_v17 import (
    FORBIDDEN_OPS,
    SCHEMA,
    ConstructiveDSLRuntimeError,
    canonical,
    execute,
    sha256_json,
    synthesize,
    walk_ops,
)
from portable_constructive_dsl_v17 import (
    PortableConstructiveError,
    execute_portable,
)

HERE = Path(__file__).resolve().parent
TASK_IDS = {"dc433765", "49d1d64f", "c8f0f002"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_error(fn: Callable[[], Any], errors: tuple[type[BaseException], ...], message: str) -> None:
    try:
        fn()
    except errors:
        return
    raise AssertionError(message)


def load_program(gate: int) -> dict[str, Any]:
    return json.loads(
        (HERE / "programs" / f"v17-program-{gate:02d}.json").read_text(
            encoding="utf-8"
        )
    )


def load_contract(gate: int) -> dict[str, Any]:
    return json.loads(
        (HERE / "contracts" / f"v17-contract-{gate:02d}.json").read_text(
            encoding="utf-8"
        )
    )


def identity_program() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "shape": {"rows": {"op": "height"}, "cols": {"op": "width"}},
        "cell": {
            "op": "sample",
            "row": {"op": "var", "name": "row"},
            "col": {"op": "var", "name": "col"},
            "default": 0,
        },
    }


def test_forbidden_opcode_rejected() -> None:
    program = {
        "schema": SCHEMA,
        "shape": {"rows": 1, "cols": 1},
        "cell": {"op": "recolour"},
    }
    expect_error(
        lambda: execute(program, ((0,),)),
        (ConstructiveDSLRuntimeError,),
        "primary runtime accepted a forbidden opcode",
    )
    expect_error(
        lambda: execute_portable(program, ((0,),)),
        (PortableConstructiveError,),
        "portable runtime accepted a forbidden opcode",
    )


def test_invalid_schema_rejected_by_both_runtimes() -> None:
    program = identity_program()
    program["schema"] = "not-v17"
    expect_error(
        lambda: execute(program, ((1,),)),
        (ConstructiveDSLRuntimeError,),
        "primary runtime accepted an invalid schema",
    )
    expect_error(
        lambda: execute_portable(program, ((1,),)),
        (PortableConstructiveError,),
        "portable runtime accepted an invalid schema",
    )


def test_ambiguous_unique_point_rejected_by_both_runtimes() -> None:
    program = {
        "schema": SCHEMA,
        "shape": {"rows": 1, "cols": 1},
        "cell": {
            "op": "if",
            "condition": {
                "op": "eq",
                "left": {"op": "unique_point", "colour": 1},
                "right": {"op": "pair", "row": 0, "col": 0},
            },
            "then": 1,
            "else": 0,
        },
    }
    grid = ((1, 1), (0, 0))
    expect_error(
        lambda: execute(program, grid),
        (ConstructiveDSLRuntimeError,),
        "primary runtime fabricated a unique point",
    )
    expect_error(
        lambda: execute_portable(program, grid),
        (PortableConstructiveError,),
        "portable runtime fabricated a unique point",
    )


def test_cell_local_recolour_primary_portable_agree() -> None:
    program = {
        "schema": SCHEMA,
        "shape": {"rows": {"op": "height"}, "cols": {"op": "width"}},
        "cell": {
            "op": "if",
            "condition": {
                "op": "eq",
                "left": {
                    "op": "sample",
                    "row": {"op": "var", "name": "row"},
                    "col": {"op": "var", "name": "col"},
                    "default": 0,
                },
                "right": 2,
            },
            "then": 7,
            "else": {
                "op": "sample",
                "row": {"op": "var", "name": "row"},
                "col": {"op": "var", "name": "col"},
                "default": 0,
            },
        },
    }
    grid = ((2, 0, 2), (1, 2, 3))
    expected = ((7, 0, 7), (1, 7, 3))
    check(execute(program, grid) == expected, "primary recolour output is wrong")
    check(
        execute_portable(program, grid) == expected,
        "portable recolour output is wrong",
    )


def test_coordinate_border_primary_portable_agree() -> None:
    program = load_program(2)
    grid = ((0, 1, 0), (2, 0, 3))
    primary = execute(program, grid)
    portable = execute_portable(program, grid)
    check(primary == portable, "coordinate-border runtimes disagree")
    check(len(primary) == 4 and len(primary[0]) == 5, "border shape is unexpected")


def test_directed_point_primary_portable_agree() -> None:
    program = load_program(1)
    grid = (
        (0, 0, 0, 0, 0),
        (0, 3, 0, 4, 0),
        (0, 0, 0, 0, 0),
    )
    check(
        execute(program, grid) == execute_portable(program, grid),
        "directed-point runtimes disagree",
    )


def test_mutating_ast_constant_changes_program_hash() -> None:
    program = load_program(3)
    mutated = copy.deepcopy(program)
    mutated["cell"]["then"] = int(mutated["cell"]["then"]) + 1
    check(
        sha256_json(program) != sha256_json(mutated),
        "AST mutation did not change program hash",
    )


def test_selected_programs_contain_no_forbidden_opcodes() -> None:
    for gate in (1, 2, 3):
        program = load_program(gate)
        hits = sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
        check(not hits, f"gate {gate} contains forbidden opcodes: {hits}")


def test_synthesizer_and_programs_contain_no_task_ids() -> None:
    signature = inspect.signature(synthesize)
    check(
        "task" not in signature.parameters and "task_id" not in signature.parameters,
        "synthesizer exposes a task identifier argument",
    )
    source = inspect.getsource(synthesize)
    for task_id in TASK_IDS:
        check(task_id not in source, f"synthesizer embeds task id {task_id}")
    for gate in (1, 2, 3):
        text = canonical(load_program(gate))
        for task_id in TASK_IDS:
            check(task_id not in text, f"gate {gate} embeds task id {task_id}")


def test_interpreters_agree_on_adversarial_rectangular_grids() -> None:
    program = identity_program()
    grids = (
        ((1, 2, 3, 4, 5, 6, 7),),
        ((1,), (2,), (3,), (4,), (5,)),
        ((9, 8, 7), (6, 5, 4)),
        tuple(tuple((row * 11 + col) % 10 for col in range(11)) for row in range(3)),
    )
    for grid in grids:
        check(
            execute(program, grid) == execute_portable(program, grid),
            f"runtimes disagree on {len(grid)}x{len(grid[0]) } grid",
        )


def test_out_of_bounds_shapes_rejected() -> None:
    for rows, cols in ((0, 1), (1, 0), (61, 1), (1, 61), (-1, 2)):
        program = {
            "schema": SCHEMA,
            "shape": {"rows": rows, "cols": cols},
            "cell": 0,
        }
        expect_error(
            lambda program=program: execute(program, ((0,),)),
            (ConstructiveDSLRuntimeError,),
            f"primary runtime accepted shape {rows}x{cols}",
        )
        expect_error(
            lambda program=program: execute_portable(program, ((0,),)),
            (PortableConstructiveError,),
            f"portable runtime accepted shape {rows}x{cols}",
        )


def test_canonical_serialization_is_deterministic() -> None:
    left = {"z": [3, {"b": 2, "a": 1}], "a": True}
    right = {"a": True, "z": [3, {"a": 1, "b": 2}]}
    check(canonical(left) == canonical(right), "canonical key ordering differs")
    check(sha256_json(left) == sha256_json(right), "canonical hashes differ")
    check(
        canonical(left).encode("utf-8").decode("utf-8") == canonical(left),
        "canonical serialization is not stable UTF-8",
    )


def test_contract_tampering_changes_hash() -> None:
    contract = load_contract(1)
    stored_hash = contract.pop("contract_sha256")
    check(
        sha256_json(contract) == stored_hash,
        "stored contract hash does not bind its payload",
    )
    tampered = copy.deepcopy(contract)
    tampered["program_sha256"] = "0" * 64
    check(
        sha256_json(tampered) != stored_hash,
        "contract tampering preserved the original hash",
    )


def test_candidate_ordering_is_deterministic() -> None:
    examples = [
        (((1, 0), (0, 1)), ((2, 0), (0, 2))),
        (((0, 1, 0),), ((0, 2, 0),)),
    ]
    program_a, report_a = synthesize(examples)
    program_b, report_b = synthesize(copy.deepcopy(examples))
    check(program_a == program_b, "repeated synthesis selected different programs")
    check(report_a == report_b, "repeated synthesis produced different reports")


def test_underdetermined_demonstrations_are_not_claimed_unique() -> None:
    examples = [(((0,),), ((0,),))]
    try:
        _program, report = synthesize(examples)
    except RuntimeError:
        return
    check(
        report["exact_survivors"] > 1,
        "underdetermined demonstrations produced a fabricated unique claim",
    )


TESTS = [
    test_forbidden_opcode_rejected,
    test_invalid_schema_rejected_by_both_runtimes,
    test_ambiguous_unique_point_rejected_by_both_runtimes,
    test_cell_local_recolour_primary_portable_agree,
    test_coordinate_border_primary_portable_agree,
    test_directed_point_primary_portable_agree,
    test_mutating_ast_constant_changes_program_hash,
    test_selected_programs_contain_no_forbidden_opcodes,
    test_synthesizer_and_programs_contain_no_task_ids,
    test_interpreters_agree_on_adversarial_rectangular_grids,
    test_out_of_bounds_shapes_rejected,
    test_canonical_serialization_is_deterministic,
    test_contract_tampering_changes_hash,
    test_candidate_ordering_is_deterministic,
    test_underdetermined_demonstrations_are_not_claimed_unique,
]


def main() -> None:
    passed = 0
    for test in TESTS:
        test()
        passed += 1
        print(f"PASS {test.__name__}")
    print(f"SUMMARY {passed}/{len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
