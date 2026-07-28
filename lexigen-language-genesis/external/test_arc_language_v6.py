from __future__ import annotations

from arc_language import as_grid
from arc_language_v6 import synthesize, transplant_matching_components_into_gray_holes


def test_exact_shape_transplant_and_source_erasure() -> None:
    source = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0],
            [0, 5, 0, 0, 5, 0, 3],
            [0, 5, 0, 5, 5, 0, 3],
            [0, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0],
            [0, 5, 3, 3, 5, 0, 0],
            [0, 5, 3, 5, 5, 0, 0],
            [0, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert transplant_matching_components_into_gray_holes(source) == expected


def test_dihedral_matching_can_rotate_source_shape() -> None:
    source = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 4],
            [0, 5, 0, 5, 5, 0, 4],
            [0, 5, 0, 0, 5, 0, 4],
            [0, 5, 5, 5, 5, 0, 0],
        ]
    )
    result = transplant_matching_components_into_gray_holes(source, allow_dihedral=True)
    assert result[2][2] == 4
    assert result[3][2] == 4
    assert result[3][3] == 4
    assert result[1][6] == result[2][6] == result[3][6] == 0


def test_operator_is_synthesizable() -> None:
    first = as_grid(
        [
            [0, 5, 5, 5, 0, 2],
            [0, 5, 0, 5, 0, 2],
            [0, 5, 5, 5, 0, 0],
        ]
    )
    first_target = as_grid(
        [
            [0, 5, 5, 5, 0, 0],
            [0, 5, 2, 5, 0, 0],
            [0, 5, 5, 5, 0, 0],
        ]
    )
    second = as_grid(
        [
            [0, 5, 5, 5, 5, 0, 7, 7],
            [0, 5, 0, 0, 5, 0, 0, 7],
            [0, 5, 0, 5, 5, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0],
        ]
    )
    second_target = as_grid(
        [
            [0, 5, 5, 5, 5, 0, 0, 0],
            [0, 5, 7, 7, 5, 0, 0, 0],
            [0, 5, 7, 5, 5, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0],
        ]
    )
    result = synthesize([(first, first_target), (second, second_target)])
    assert result.program is not None
    assert any(step["op"] == "transplant_matching_components_into_gray_holes" for step in result.program)
