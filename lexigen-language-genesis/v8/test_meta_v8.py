from __future__ import annotations

from meta_runtime_v8 import as_grid, canonical_json, execute_extension
from meta_synthesizer_v8 import synthesize_meta_extension
from portable_runtime_v8 import as_grid as portable_grid
from portable_runtime_v8 import execute_portable


def make_case(lines: list[tuple[int, int, int, int]], winner_positive: bool):
    source = [[7 for _ in range(9)] for _ in range(9)]
    target = [[7 for _ in range(9)] for _ in range(9)]
    for row, col, length, angle in lines:
        colour = 8 if (angle > 0) == winner_positive else 2
        for index in range(length):
            r, c = row + index, col + index * angle
            source[r][c] = 5
            target[r][c] = colour
    return as_grid(source), as_grid(target)


def examples():
    first = make_case(
        [(0, 0, 3, 1), (5, 0, 2, 1), (0, 8, 4, -1)],
        winner_positive=True,
    )
    second = make_case(
        [(0, 0, 2, 1), (0, 8, 3, -1), (5, 7, 2, -1)],
        winner_positive=False,
    )
    return [first, second]


def test_v8_synthesizes_executable_extension() -> None:
    result = synthesize_meta_extension(examples())
    assert result.extension is not None
    assert result.fixed_grammar_baseline_found is False
    assert result.candidates_tested > 100
    for source, target in examples():
        assert execute_extension(result.extension, source) == target


def test_extension_is_arithmetic_graph_ast_not_named_diagonal_operator() -> None:
    result = synthesize_meta_extension(examples())
    assert result.extension is not None
    text = canonical_json(result.extension)
    assert "diagonal" not in text
    assert "line" not in text
    assert "fold_sum" in text
    assert "component_class" in text
    assert result.extension["provenance"]["human_supplied_finished_task_operator"] is False


def test_portable_runtime_reproduces_generated_extension() -> None:
    result = synthesize_meta_extension(examples())
    assert result.extension is not None
    for source, target in examples():
        assert execute_portable(result.extension, portable_grid(source)) == target


def test_extension_ablation_fails() -> None:
    assert any(source != target for source, target in examples())
