from __future__ import annotations

from arc_language import as_grid, execute_program, language_artifact, synthesize


def test_single_primitive_baseline_detected() -> None:
    examples = [
        (
            as_grid([[1, 0], [2, 3]]),
            as_grid([[2, 1], [3, 0]]),
        ),
        (
            as_grid([[4, 5, 0], [0, 6, 7]]),
            as_grid([[0, 4], [6, 5], [7, 0]]),
        ),
    ]
    result = synthesize(examples)
    assert result.program is not None
    assert result.baseline_program is not None
    assert len(result.program) == 1


def test_composed_macro_required() -> None:
    examples = [
        (
            as_grid([[0, 0, 0], [0, 2, 0], [0, 3, 0]]),
            as_grid([[3, 2]]),
        ),
        (
            as_grid([[0, 4, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0]]),
            as_grid([[5, 4]]),
        ),
    ]
    result = synthesize(examples)
    assert result.program is not None
    assert result.baseline_program is None
    assert len(result.program) == 2
    for source, target in examples:
        assert execute_program(result.program, source) == target
    artifact = language_artifact(result.program, examples)
    assert artifact["schema"] == "lexigen-arc-language-artifact-v1"
    assert len(artifact["operational_semantics"]) == 2


def test_recolour_then_rotate_macro() -> None:
    examples = [
        (
            as_grid([[0, 1], [0, 1]]),
            as_grid([[7, 7], [0, 0]]),
        ),
        (
            as_grid([[0, 0, 2], [0, 0, 2]]),
            as_grid([[7, 7], [0, 0], [0, 0]]),
        ),
    ]
    result = synthesize(examples)
    assert result.program is not None
    assert len(result.program) >= 2
    for source, target in examples:
        assert execute_program(result.program, source) == target
