from __future__ import annotations

from enumerator_v25_recovery import abstract_literal_colours, enumerate_programs
from runtime_v25 import eval_ast


def check(value, message):
    if not value:
        raise AssertionError(message)


def test_runtime_selected_object_paint():
    source = (
        (0, 1, 1, 0, 1, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )
    ast = {
        "op": "paint",
        "grid": {"op": "input_grid"},
        "points": {
            "op": "objects_to_points",
            "objects": {
                "op": "select_objects",
                "feature": "size",
                "extremum": "maximum",
                "objects": {
                    "op": "components4",
                    "points": {
                        "op": "points_of_color",
                        "colour": {"op": "literal_color", "value": 1},
                    },
                },
            },
        },
        "colour": {"op": "literal_color", "value": 2},
    }
    target = eval_ast(ast, source)
    check(target[0][1:3] == (2, 2), "largest component row")
    check(target[1][1:3] == (2, 2), "largest component body")
    check(target[0][4] == 1, "singleton preserved")


def test_literal_colour_abstraction():
    ast = {
        "op": "paint",
        "grid": {"op": "canvas", "colour": {"op": "literal_color", "value": 0}},
        "points": {"op": "points_of_color", "colour": {"op": "literal_color", "value": 3}},
        "colour": {"op": "literal_color", "value": 3},
    }
    abstract, arguments = abstract_literal_colours(ast)
    check(arguments == {"c0": 3, "c1": 0} or arguments == {"c0": 0, "c1": 3}, "arguments")
    names = []
    def walk(value):
        if isinstance(value, dict):
            if value.get("op") == "param_color":
                names.append(value["name"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(abstract)
    check(len(set(names)) == 2, "distinct parameters")


def test_semantic_enumerator_discovers_composition():
    source = ((0, 3, 0, 0, 3, 0),)
    target = ((0, 8, 8, 8, 8, 0),)
    report = enumerate_programs(
        [(source, target)],
        maximum_depth=3,
        maximum_unique_per_type_per_depth=5000,
        maximum_total_unique=30000,
        maximum_raw_candidates=300000,
    )
    check(report["enumeration_complete"], "enumeration incomplete")
    check(report["exact_concrete_programs"] >= 1, "no exact program")
    check(report["exact_abstract_structures"] >= 1, "no abstract structure")


def test_identity_task_is_not_counted():
    source = ((0, 1, 0),)
    report = enumerate_programs(
        [(source, source)],
        maximum_depth=1,
        maximum_unique_per_type_per_depth=500,
        maximum_total_unique=3000,
        maximum_raw_candidates=10000,
    )
    check(not report["nontrivial_task"], "identity marked nontrivial")
    check(report["exact_concrete_programs"] == 0, "identity exact program counted")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
