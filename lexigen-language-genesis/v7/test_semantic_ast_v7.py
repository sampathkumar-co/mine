from __future__ import annotations

from portable_runtime_v7 import execute_portable
from semantic_ast_v7 import as_grid, execute_ast, synthesize_ast


def positive_example():
    source = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 5, 0, 0, 5, 0, 3, 3, 0],
            [0, 5, 0, 5, 5, 0, 3, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    target = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 5, 3, 3, 5, 0, 0, 0, 0],
            [0, 5, 3, 5, 5, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return source, target


def mirrored_negative_example():
    source = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 5, 0, 0, 5, 0, 4, 0, 0],
            [0, 5, 0, 5, 5, 0, 4, 4, 0],
            [0, 5, 5, 5, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return source, source


def test_synthesizer_requires_exact_identity_shape() -> None:
    examples = [positive_example(), mirrored_negative_example()]
    result = synthesize_ast(examples)
    assert result.ast is not None
    assert result.ambiguous is False
    assert result.ast["scene"]["frame_colour"] == 5
    assert result.ast["scene"]["hole_boundary"] == "all"
    assert result.ast["scene"]["exclude_frame_objects"] is True
    assert result.ast["match"] == {"feature": "normalised_points", "symmetry": "identity"}
    assert result.ast["render"]["erase_source"] is True


def test_primary_and_portable_runtimes_agree() -> None:
    examples = [positive_example(), mirrored_negative_example()]
    result = synthesize_ast(examples)
    assert result.ast is not None
    for source, target in examples:
        assert execute_ast(result.ast, source) == target
        assert execute_portable(result.ast, source) == target
        assert execute_ast(result.ast, source) == execute_portable(result.ast, source)


def test_weaker_area_and_bbox_rules_are_rejected() -> None:
    positive, negative = positive_example(), mirrored_negative_example()
    result = synthesize_ast([positive, negative])
    assert result.ast is not None
    exact_features = {candidate["match"]["feature"] for candidate in result.exact_candidates}
    assert "area" not in exact_features
    assert "bbox" not in exact_features


def test_source_is_erased_and_unmatched_distractor_survives() -> None:
    source, target = positive_example()
    values = [list(row) for row in source]
    values[0][8] = 8
    source_with_distractor = as_grid(values)
    expected = [list(row) for row in target]
    expected[0][8] = 8
    expected_grid = as_grid(expected)
    result = synthesize_ast([(source, target), mirrored_negative_example()])
    assert result.ast is not None
    assert execute_ast(result.ast, source_with_distractor) == expected_grid
