from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

from constructive_dsl_v17 import (
    ConstructiveDSLRuntimeError,
    FORBIDDEN_OPS,
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


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_program(gate: int) -> dict:
    path = HERE / "programs" / f"v17-program-{gate:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def expect_error(fn, error_type, message: str) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(message)


def test_forbidden_opcode_rejected() -> None:
    program = {
        "schema": "lexigen-v17-constructive-grid-v1",
        "shape": {"rows": 1, "cols": 1},
        "cell": {"op": "recolour"},
    }
    expect_error(
        lambda: execute(program, ((0,),)),
        ConstructiveDSLRuntimeError,
        "forbidden opcode was accepted",
    )


def test_bad_schema_rejected() -> None:
    program = {"schema": "bad", "shape": {"rows": 1, "cols": 1}, "cell": 0}
    expect_error(lambda: execute(program, ((0,),)), ConstructiveDSLRuntimeError, "primary accepted bad schema")
    expect_error(lambda: execute_portable(program, ((0,),)), PortableConstructiveError, "portable accepted bad schema")


def test_ambiguous_unique_point_rejected() -> None:
    program = {
        "schema": "lexigen-v17-constructive-grid-v1",
        "shape": {"rows": 1, "cols": 1},
        "cell": {"op": "first", "pair": {"op": "unique_point", "colour": 1}},
    }
    grid = ((1, 1),)
    expect_error(lambda: execute(program, grid), ConstructiveDSLRuntimeError, "primary accepted ambiguous point")
    expect_error(lambda: execute_portable(program, grid), PortableConstructiveError, "portable accepted ambiguous point")


def test_program_parity_and_hashes() -> None:
    grids = {
        1: ((0, 3, 0), (0, 0, 0), (0, 4, 0)),
        2: ((1, 2), (3, 4)),
        3: ((7, 0, 7), (2, 7, 3)),
    }
    for gate, grid in grids.items():
        program = load_program(gate)
        check(execute(program, grid) == execute_portable(program, grid), f"runtime mismatch gate {gate}")
        check(not (set(walk_ops(program)) & FORBIDDEN_OPS), f"forbidden opcode gate {gate}")
        mutated = copy.deepcopy(program)
        mutated["schema"] = mutated["schema"] + "-mutated"
        check(sha256_json(mutated) != sha256_json(program), f"hash did not change gate {gate}")
        check(canonical(program) == canonical(json.loads(canonical(program))), "canonical form unstable")


def test_synthesizer_interface_and_determinism() -> None:
    check("task" not in inspect.signature(synthesize).parameters, "synthesizer accepts task identity")
    examples = [(((7, 0), (0, 2)), ((5, 0), (0, 2)))]
    first, first_meta = synthesize(examples)
    second, second_meta = synthesize(examples)
    check(first == second, "synthesis is nondeterministic")
    check(first_meta == second_meta, "synthesis metadata is nondeterministic")
    check("task" not in canonical(first).lower(), "task identity leaked into program")


def test_invalid_shape_rejected() -> None:
    program = {
        "schema": "lexigen-v17-constructive-grid-v1",
        "shape": {"rows": -1, "cols": 2},
        "cell": 0,
    }
    expect_error(lambda: execute(program, ((0,),)), ConstructiveDSLRuntimeError, "primary accepted invalid shape")
    expect_error(lambda: execute_portable(program, ((0,),)), PortableConstructiveError, "portable accepted invalid shape")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"{len(tests)} v17 tests passed")


if __name__ == "__main__":
    main()
