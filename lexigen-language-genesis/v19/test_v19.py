from __future__ import annotations

import copy

from invent_production_v19 import invent_production
from portable_runtime_v19 import (
    PortableV19Error,
    execute_production_portable,
    execute_portable,
)
from primitive_runtime_v19 import (
    FORBIDDEN_OPS,
    PROGRAM_SCHEMA,
    PrimitiveRuntimeError,
    canonical,
    execute,
    execute_production,
    sha256_json,
    walk_ops,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def column_example(fill: int = 3):
    source = (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (1, 1, 1, 0, 1, 1, 1),
        (1, 1, 1, 0, 1, 1, 1),
        (1, 1, 1, 0, 1, 1, 1),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
    )
    target = tuple(
        tuple(fill if col == 3 else value for col, value in enumerate(row))
        for row in source
    )
    return source, target


def transpose(grid):
    return tuple(tuple(grid[row][col] for row in range(len(grid))) for col in range(len(grid[0])))


def test_invention_is_deterministic() -> None:
    examples = [column_example()]
    left = invent_production(examples)
    right = invent_production(examples)
    check(left[1] == right[1], "arguments changed")
    check(sha256_json(left[0]) == sha256_json(right[0]), "production changed")
    check(sha256_json(left[2]) == sha256_json(right[2]), "source program changed")


def test_primary_and_portable_agree() -> None:
    source, target = column_example()
    production, arguments, _, _ = invent_production([(source, target)])
    check(execute_production(production, arguments, source) == target, "primary failed")
    check(
        execute_production_portable(production, arguments, source) == target,
        "portable failed",
    )


def test_row_axis_is_generated() -> None:
    source, target = column_example(6)
    source, target = transpose(source), transpose(target)
    production, arguments, _, report = invent_production([(source, target)])
    check(report["selected_axis_kind"] == "row", "row production was not selected")
    check(execute_production(production, arguments, source) == target, "row primary failed")
    check(
        execute_production_portable(production, arguments, source) == target,
        "row portable failed",
    )


def test_invalid_unique_rejected_by_both() -> None:
    program = {
        "schema": PROGRAM_SCHEMA,
        "bindings": [{
            "name": "x",
            "expr": {"op": "unique", "items": {"op": "range", "stop": 2}},
        }],
        "shape": {"rows": 1, "cols": 1},
        "cell": 0,
    }
    try:
        execute(program, ((0,),))
    except PrimitiveRuntimeError:
        pass
    else:
        raise AssertionError("primary accepted invalid unique reduction")
    try:
        execute_portable(program, ((0,),))
    except PortableV19Error:
        pass
    else:
        raise AssertionError("portable accepted invalid unique reduction")


def test_forbidden_scene_operator_rejected() -> None:
    program = {
        "schema": PROGRAM_SCHEMA,
        "shape": {"rows": 1, "cols": 1},
        "cell": {"op": "fill_internal_blank_axis"},
    }
    for runner, error in (
        (execute, PrimitiveRuntimeError),
        (execute_portable, PortableV19Error),
    ):
        try:
            runner(program, ((0,),))
        except error:
            continue
        raise AssertionError("forbidden operator was accepted")


def test_tampering_changes_hash() -> None:
    production, _, _, _ = invent_production([column_example()])
    changed = copy.deepcopy(production)
    changed["body"]["cell"]["else"]["default"] = 9
    check(sha256_json(changed) != sha256_json(production), "tampering kept hash")


def test_no_named_scene_operator() -> None:
    production, _, program, _ = invent_production([column_example()])
    hits = (set(walk_ops(production)) | set(walk_ops(program))) & FORBIDDEN_OPS
    check(not hits, f"forbidden operators present: {sorted(hits)}")


def test_fill_parameter_transfers() -> None:
    source, target = column_example(3)
    production, arguments, _, _ = invent_production([(source, target)])
    changed_arguments = {"fill_colour": 8}
    expected = column_example(8)[1]
    check(
        execute_production(production, changed_arguments, source) == expected,
        "fill parameter did not transfer",
    )
    check(arguments == {"fill_colour": 3}, "wrong learned argument")


def test_canonical_serialization() -> None:
    value = {"b": [2, 1], "a": {"z": 0}}
    reordered = {"a": {"z": 0}, "b": [2, 1]}
    check(canonical(value) == canonical(reordered), "canonical ordering changed")
    check(sha256_json(value) == sha256_json(reordered), "canonical hash changed")


def test_task_identity_absent() -> None:
    production, _, program, _ = invent_production([column_example()])
    check("da2b0fe3" not in canonical(production), "task id in production")
    check("da2b0fe3" not in canonical(program), "task id in source program")


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
