from __future__ import annotations

from arc_language import as_grid
from arc_language_v4 import synthesize, trace_reflected_diagonal


def test_reflected_diagonal_narrow_grid() -> None:
    source = as_grid([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])
    expected = as_grid([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert trace_reflected_diagonal(
        source,
        source=1,
        vertical_direction=-1,
        horizontal_direction=1,
    ) == expected


def test_reflected_diagonal_wide_grid() -> None:
    source = as_grid([[0] * 5 for _ in range(5)])
    source = tuple(source[:-1]) + ((1, 0, 0, 0, 0),)
    expected = as_grid(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    assert trace_reflected_diagonal(
        source,
        source=1,
        vertical_direction=-1,
        horizontal_direction=1,
    ) == expected


def test_trajectory_operator_is_synthesizable() -> None:
    first = as_grid([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    first_target = as_grid([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    second = as_grid([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    second_target = as_grid([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    result = synthesize([(first, first_target), (second, second_target)])
    assert result.program is not None
    assert any(step["op"] == "trace_reflected_diagonal" for step in result.program)
