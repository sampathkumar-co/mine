from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
for folder in (HERE, V25):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from enumerator_v27 import enumerate_programs, grid_distance, support_distance, target_anchors
from runtime_v25 import as_grid


def grid(rows):
    return as_grid(rows)


def paint_border_example(colour: int):
    source = grid([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    target = grid([
        [0, 0, 0, 0, 0],
        [0, colour, colour, 0, 0],
        [0, colour, colour, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    return source, target


def run(examples, *, grid_beam=20):
    return enumerate_programs(
        examples,
        maximum_depth=3,
        maximum_unique_considered_per_type_per_depth=500,
        maximum_raw_candidates=100000,
        retained_total_caps={
            "Color": 100,
            "ObjectSet": 100,
            "PointSet": 150,
            "Grid": 100,
        },
        beam_per_depth={"ObjectSet": 20, "PointSet": 30, "Grid": grid_beam},
    )


def test_grid_distance():
    target = grid([[0, 1], [0, 0]])
    assert grid_distance((target,), (target,)) == (0, 0)
    changed = grid([[0, 0], [0, 0]])
    assert grid_distance((changed,), (target,)) == (0, 1)
    smaller = grid([[0]])
    assert grid_distance((smaller,), (target,))[0] == 1


def test_exact_match_survives_beam():
    examples = [paint_border_example(8), paint_border_example(8)]
    result = run(examples, grid_beam=1)
    assert result["exact_concrete_programs"] >= 1
    assert result["exact_abstract_structures"] >= 1
    assert result["retained_by_type"]["Grid"] <= 4


def test_support_types_run_before_grid():
    examples = [paint_border_example(8)]
    result = run(examples, grid_beam=5)
    depth_one = [item["type"] for item in result["statistics"] if item["depth"] == 1]
    assert depth_one == ["Color", "ObjectSet", "PointSet", "Grid"]


def test_literal_colour_is_abstracted():
    examples = [paint_border_example(8), paint_border_example(8)]
    result = run(examples, grid_beam=5)
    structures = result["exact_structures"]
    assert structures
    text = str(structures[0]["structure"])
    assert "param_color" in text
    assert "literal_color" not in text


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


def test_support_distance_prefers_changed_region():
    example = paint_border_example(8)
    anchors = target_anchors([example])
    changed = frozenset(
        (row, col)
        for row, (left_row, right_row) in enumerate(zip(*example))
        for col, (left, right) in enumerate(zip(left_row, right_row))
        if left != right
    )
    assert support_distance("PointSet", (changed,), anchors) == 0


def test_support_beams_bound_retention():
    result = run([paint_border_example(8)], grid_beam=5)
    assert result["retained_by_type"]["ObjectSet"] <= 100
    assert result["retained_by_type"]["PointSet"] <= 150
    assert result["retained_by_type"]["Grid"] <= 16


if __name__ == "__main__":
    main()
