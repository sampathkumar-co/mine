from __future__ import annotations

from hierarchical_runtime_v12 import as_grid, execute_program
from hierarchical_synthesizer_v12 import synthesize_hierarchical
from portable_runtime_v12 import execute_portable


def _completion_program():
    return {
        "schema": "lexigen-hierarchical-scene-v1",
        "stages": [
            {"mode": "separator_lines", "separator_colour": 0},
            {"mode": "complete_local_midpoints", "cell_background": 1, "require_same_colour": True},
            {"mode": "preserve_canvas"},
        ],
    }


def test_local_midpoints_complete_in_both_axes():
    source = as_grid([
        [7, 1, 0, 1, 1, 0, 7, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [7, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
    ])
    target = [list(row) for row in source]
    target[0][3] = 7
    target[3][0] = 7
    assert execute_program(_completion_program(), source) == as_grid(target)
    assert execute_portable(_completion_program(), source) == target


def test_container_mode_reduction_with_border():
    source = as_grid([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 5, 6, 0],
        [0, 2, 3, 0, 5, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 0, 4, 4, 0],
        [0, 8, 7, 0, 9, 4, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    target = as_grid([
        [0, 0, 0, 0],
        [0, 2, 5, 0],
        [0, 8, 4, 0],
        [0, 0, 0, 0],
    ])
    program = {
        "schema": "lexigen-hierarchical-scene-v1",
        "stages": [
            {"mode": "separator_lines", "separator_colour": 0},
            {"mode": "reduce_mode"},
            {"mode": "summary_grid", "border": 1, "canvas_colour": 0},
        ],
    }
    assert execute_program(program, source) == target
    assert execute_portable(program, source) == [list(row) for row in target]


def test_marker_chain_alignment_uses_local_frames():
    source = as_grid([
        [0, 0, 0, 0, 0, 0, 0, 5, 4],
        [2, 5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 5, 3, 5, 0, 0, 0],
    ])
    target = as_grid([
        [0, 0, 0, 0, 0, 0, 0],
        [2, 2, 3, 3, 3, 4, 4],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    program = {
        "schema": "lexigen-hierarchical-scene-v1",
        "stages": [
            {"mode": "marker_gap_chain", "background": 0, "marker": 5},
            {"mode": "align_local_frames", "rank_mode": "min_shift"},
            {"mode": "concatenate_segments"},
        ],
    }
    assert execute_program(program, source) == target
    assert execute_portable(program, source) == [list(row) for row in target]


def test_synthesizer_selects_hierarchical_family_not_noop():
    source = as_grid([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 5, 6, 0],
        [0, 2, 3, 0, 5, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 0, 4, 4, 0],
        [0, 8, 7, 0, 9, 4, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    target = as_grid([
        [0, 0, 0, 0],
        [0, 2, 5, 0],
        [0, 8, 4, 0],
        [0, 0, 0, 0],
    ])
    result = synthesize_hierarchical([(source, target)])
    assert result.program is not None
    assert result.program["stages"][1]["mode"] == "reduce_mode"
    assert execute_program(result.program, source) == target
    assert source != target
