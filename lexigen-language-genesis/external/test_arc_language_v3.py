from __future__ import annotations

from arc_language import as_grid
from arc_language_v3 import connect_anchor_spine, synthesize


def test_horizontal_spine_with_vertical_projections() -> None:
    source = as_grid(
        [
            [0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2, 0, 0],
        ]
    )
    expected = as_grid(
        [
            [0, 0, 2, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0],
            [2, 3, 3, 3, 3, 3, 2],
            [0, 0, 0, 0, 2, 0, 0],
        ]
    )
    assert connect_anchor_spine(source, 2, 3) == expected


def test_vertical_spine_with_horizontal_projections() -> None:
    source = as_grid(
        [
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0],
            [0, 0, 0, 0, 2],
            [0, 0, 2, 0, 0],
        ]
    )
    expected = as_grid(
        [
            [0, 0, 2, 0, 0],
            [0, 0, 3, 0, 0],
            [2, 3, 2, 0, 0],
            [0, 0, 3, 3, 2],
            [0, 0, 2, 0, 0],
        ]
    )
    assert connect_anchor_spine(source, 2, 3) == expected


def test_spine_operator_is_synthesizable() -> None:
    first = as_grid([[2, 0, 0, 2], [0, 0, 0, 0], [0, 2, 0, 0]])
    first_target = as_grid([[2, 3, 3, 2], [0, 3, 0, 0], [0, 2, 0, 0]])
    second = as_grid([[0, 2, 0], [0, 0, 2], [0, 2, 0]])
    second_target = as_grid([[0, 2, 0], [0, 3, 2], [0, 2, 0]])
    result = synthesize([(first, first_target), (second, second_target)])
    assert result.program is not None
    assert any(step["op"] == "connect_anchor_spine" for step in result.program)
