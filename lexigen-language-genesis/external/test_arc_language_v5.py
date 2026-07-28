from __future__ import annotations

from arc_language import as_grid
from arc_language_v5 import expand_legend_bounding_fields, synthesize


def test_single_legend_rectangle_expansion() -> None:
    source = as_grid(
        [
            [2, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    expected = as_grid(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 7, 7, 7, 7, 7],
            [0, 7, 2, 7, 2, 7],
            [0, 7, 7, 7, 7, 7],
            [0, 7, 2, 7, 2, 7],
            [0, 7, 7, 7, 7, 7],
        ]
    )
    assert expand_legend_bounding_fields(source) == expected


def test_legend_order_wins_overlap() -> None:
    source = as_grid(
        [
            [2, 7, 0, 0, 0, 0, 0],
            [3, 6, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 3, 0, 0],
        ]
    )
    result = expand_legend_bounding_fields(source)
    assert result[3][2] == 7
    assert result[5][2] == 6
    assert result[0][0] == 0 and result[1][1] == 0


def test_fill_colour_can_be_later_source_colour() -> None:
    source = as_grid(
        [
            [5, 6, 0, 0, 0, 0],
            [6, 8, 0, 0, 0, 0],
            [0, 0, 5, 0, 5, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 6, 0, 6, 0],
        ]
    )
    result = expand_legend_bounding_fields(source)
    assert result[1][1] == 6
    assert result[4][2] == 6
    assert result[4][1] == 6
    assert result[4][5] == 8


def test_legend_field_operator_is_synthesizable() -> None:
    first = as_grid([[2, 7, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]])
    first_target = as_grid([[7, 7, 7, 0], [7, 2, 7, 0], [7, 7, 7, 0]])
    second = as_grid([[4, 9, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 4, 0, 0]])
    second_target = as_grid([[0, 9, 9, 9, 0], [0, 9, 4, 9, 0], [0, 9, 9, 9, 0]])
    result = synthesize([(first, first_target), (second, second_target)])
    assert result.program is not None
    assert any(step["op"] == "expand_legend_bounding_fields" for step in result.program)
