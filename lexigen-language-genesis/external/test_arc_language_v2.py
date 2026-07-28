from __future__ import annotations

from arc_language import as_grid
from arc_language_v2 import execute_program, synthesize


def test_connect_aligned_is_learned_from_multiple_surfaces() -> None:
    examples = [
        (
            as_grid([[0, 2, 0, 0], [0, 0, 0, 0], [0, 2, 0, 2]]),
            as_grid([[0, 2, 0, 0], [0, 3, 0, 0], [0, 2, 3, 2]]),
        ),
        (
            as_grid([[2, 0, 0, 2], [0, 0, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0]]),
            as_grid([[2, 3, 3, 2], [0, 0, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0]]),
        ),
    ]
    result = synthesize(examples)
    assert result.program is not None
    assert result.program == ({"op": "connect_aligned", "source": 2, "fill": 3},)
    for source, target in examples:
        assert execute_program(result.program, source) == target
