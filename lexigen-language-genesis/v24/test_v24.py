from __future__ import annotations

from runtime_v24 import execute
from scan_one_v24 import (
    EXPECTED_CANDIDATES,
    classifier_structures,
    gravity_structures,
    paint_structures,
)


def check(value, message):
    if not value:
        raise AssertionError(message)


def test_candidate_denominator():
    paint = sum(1 for _ in paint_structures())
    classify = sum(1 for _ in classifier_structures())
    gravity = sum(1 for _ in gravity_structures())
    check((paint, classify, gravity) == (1920, 80, 8), "structure denominator")
    check(paint * 100 + classify * 1000 + gravity == EXPECTED_CANDIDATES, "program denominator")


def test_paint_reflection():
    source = (
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )
    program = {
        "op": "paint_edit",
        "base_mode": "input",
        "component_filter": "all",
        "transform": "bbox_reflect_right",
        "region_mode": "points",
        "combine_mode": "source_union_mapped",
        "paint_mode": "source_colour",
        "source_colour": 2,
        "paint_colour": 9,
    }
    target = execute(program, source)
    check(target[1][3] == 2 and target[1][4] == 2 and target[2][4] == 2, "reflection")


def test_connect_aligned():
    source = ((0, 3, 0, 0, 3, 0),)
    program = {
        "op": "paint_edit",
        "base_mode": "input",
        "component_filter": "all",
        "transform": "identity",
        "region_mode": "connect_aligned",
        "combine_mode": "mapped_only",
        "paint_mode": "literal_colour",
        "source_colour": 3,
        "paint_colour": 8,
    }
    check(execute(program, source) == ((0, 8, 8, 8, 8, 0),), "connect aligned")


def test_relational_classifier():
    source = (
        (0, 4, 0, 4, 0),
        (0, 4, 0, 0, 0),
    )
    program = {
        "op": "relational_classify",
        "base_mode": "background_canvas",
        "component_filter": "all",
        "relation": "grid_flip_h",
        "source_colour": 4,
        "equal_colour": 7,
        "unequal_colour": 2,
    }
    target = execute(program, source)
    check(target[0][1] == 7 and target[0][3] == 7 and target[1][1] == 2, "classification")


def test_gravity_pack():
    source = (
        (0, 5, 0),
        (0, 0, 0),
        (0, 7, 0),
    )
    program = {
        "op": "gravity_pack",
        "axis": "columns",
        "direction": "end",
        "base_mode": "background_canvas",
    }
    check(execute(program, source) == ((0, 0, 0), (0, 5, 0), (0, 7, 0)), "gravity")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
