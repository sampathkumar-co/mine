from __future__ import annotations

from object_motion_runtime_v9 import as_grid, canonical_json, execute_extension
from object_motion_synthesizer_v9 import synthesize_object_motion
from portable_runtime_v9 import as_grid as portable_grid
from portable_runtime_v9 import execute_portable


def example(marker_corner: str):
    source = [[7 for _ in range(8)] for _ in range(8)]
    target = [[7 for _ in range(8)] for _ in range(8)]
    r0, c0, height, width = 2, 2, 3, 4
    corners = {
        "nw": (r0, c0, -1, -1),
        "ne": (r0, c0 + width - 1, -1, 1),
        "sw": (r0 + height - 1, c0, 1, -1),
        "se": (r0 + height - 1, c0 + width - 1, 1, 1),
    }
    marker_row, marker_col, dr, dc = corners[marker_corner]
    for row in range(r0, r0 + height):
        for col in range(c0, c0 + width):
            source[row][col] = 4
            target[row + dr][col + dc] = 4
    source[marker_row][marker_col] = 8
    return as_grid(source), as_grid(target)


def examples():
    return [example("nw"), example("ne"), example("sw"), example("se")]


def test_v9_synthesizes_outward_motion() -> None:
    result = synthesize_object_motion(examples())
    assert result.extension is not None
    assert result.candidates_tested > 100
    assert result.extension["displacement"] == {
        "op": "derive_axis_vector_from_marker_extreme",
        "row": "outward",
        "col": "outward",
    }
    for source, target in examples():
        assert execute_extension(result.extension, source) == target


def test_v9_completes_marker_replaced_shape() -> None:
    result = synthesize_object_motion(examples())
    assert result.extension is not None
    assert result.extension["shape"]["mode"] in {"bbox_fill", "component_plus_marker"}
    assert result.extension["render"]["erase_source"] is True


def test_v9_portable_runtime_agrees() -> None:
    result = synthesize_object_motion(examples())
    assert result.extension is not None
    for source, target in examples():
        assert execute_portable(result.extension, portable_grid(source)) == target


def test_v9_artifact_has_no_finished_task_name() -> None:
    result = synthesize_object_motion(examples())
    assert result.extension is not None
    text = canonical_json(result.extension)
    assert "rectangle_shift" not in text
    assert "corner_marker_task" not in text
    assert result.extension["provenance"]["human_supplied_finished_task_operator"] is False
