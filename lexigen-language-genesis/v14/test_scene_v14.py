from __future__ import annotations

from portable_scene_runtime_v14 import execute_portable_pipeline
from scene_runtime_v14 import (
    as_grid,
    canonical_rectangular_layers,
    decode_regular_linegrid,
    edge_project,
    extend_corner_marked_rays,
    fill_internal_blank_axis,
    move_singleton_towards,
    overlay_equal_tiles,
)


def assert_portable(pipeline, source, expected):
    assert execute_portable_pipeline(pipeline, source) == expected


def draw_hollow(values, box, colour):
    r0, c0, r1, c1 = box
    for c in range(c0, c1 + 1):
        values[r0][c] = values[r1][c] = colour
    for r in range(r0, r1 + 1):
        values[r][c0] = values[r][c1] = colour


def test_singleton_motion_handles_diagonal_relation():
    source = as_grid([
        [3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4],
    ])
    expected = as_grid([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4],
    ])
    assert move_singleton_towards(source, 3, 4) == expected
    assert_portable(({"op": "move_singleton_towards", "source_colour": 3, "target_colour": 4},), source, expected)


def test_edge_projection_uses_inferred_canvas_colour():
    source = as_grid([[1, 2], [3, 4]])
    expected = as_grid([
        [0, 1, 2, 0],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [0, 3, 4, 0],
    ])
    assert edge_project(source, fill_colour=0) == expected
    assert_portable(({"op": "edge_project", "radius": 1, "fill_colour": 0},), source, expected)


def make_line_grid(bitmap, separator):
    height, width = len(bitmap), len(bitmap[0])
    output = [[separator for _ in range(width * 3 - 1)] for _ in range(height * 3 - 1)]
    for r, row in enumerate(bitmap):
        for c, value in enumerate(row):
            for dr in range(2):
                for dc in range(2):
                    output[r * 3 + dr][c * 3 + dc] = value
    return as_grid(output)


def test_structural_line_grid_ignores_literal_separator_colour():
    bitmap = as_grid([[1, 2], [3, 4]])
    for separator in (6, 8):
        source = make_line_grid(bitmap, separator)
        assert decode_regular_linegrid(source, "structural", "identity") == bitmap
        assert_portable(({
            "op": "decode_regular_linegrid",
            "line_colour": "structural",
            "transform": "identity",
        },), source, bitmap)


def test_tile_overlay_has_explicit_conflict_priority():
    source = as_grid([
        [1, 0, 4, 0],
        [0, 0, 0, 0],
        [3, 0, 2, 0],
        [0, 0, 0, 0],
    ])
    expected = as_grid([[4, 0], [0, 0]])
    order = (0, 3, 2, 1)
    assert overlay_equal_tiles(source, 2, 2, order) == expected
    assert_portable(({
        "op": "overlay_equal_tiles", "tile_rows": 2,
        "tile_cols": 2, "order": list(order),
    },), source, expected)


def test_nested_layers_preserve_repeated_colours():
    values = [[2 for _ in range(9)] for _ in range(9)]
    for r in range(1, 8):
        for c in range(1, 8):
            values[r][c] = 3
    for r in range(2, 7):
        for c in range(2, 7):
            values[r][c] = 2
    source = as_grid(values)
    expected = as_grid([
        [2, 2, 2, 2, 2],
        [2, 3, 3, 3, 2],
        [2, 3, 2, 3, 2],
        [2, 3, 3, 3, 2],
        [2, 2, 2, 2, 2],
    ])
    assert canonical_rectangular_layers(source, "components") == expected
    assert_portable(({
        "op": "canonical_rectangular_layers", "object_mode": "components",
    },), source, expected)


def test_overlapping_frames_use_occlusion_without_containment_cycle():
    values = [[0 for _ in range(7)] for _ in range(7)]
    draw_hollow(values, (0, 0, 4, 4), 1)
    draw_hollow(values, (2, 2, 6, 6), 2)
    source = as_grid(values)
    expected = as_grid([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    assert canonical_rectangular_layers(source, "colours") == expected
    assert_portable(({
        "op": "canonical_rectangular_layers", "object_mode": "colours",
    },), source, expected)


def test_blank_axis_supports_horizontal_and_vertical_roles():
    horizontal = as_grid([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    expected_h = as_grid([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [3, 3, 3, 3, 3],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    vertical = as_grid(list(map(list, zip(*horizontal))))
    expected_v = as_grid(list(map(list, zip(*expected_h))))
    assert fill_internal_blank_axis(horizontal, 3) == expected_h
    assert fill_internal_blank_axis(vertical, 3) == expected_v
    pipeline = ({"op": "fill_internal_blank_axis", "fill_colour": 3},)
    assert_portable(pipeline, horizontal, expected_h)
    assert_portable(pipeline, vertical, expected_v)


def test_multiple_corner_ray_motifs_execute_independently():
    values = [[0 for _ in range(8)] for _ in range(8)]
    for r, c in ((2, 3), (3, 2), (3, 3)):
        values[r][c] = 8
    for r, c in ((3, 5), (3, 6), (4, 5)):
        values[r][c] = 7
    source = as_grid(values)
    expected = [row[:] for row in values]
    expected[1][1] = expected[0][0] = 8
    expected[5][7] = 7
    expected_grid = as_grid(expected)
    assert extend_corner_marked_rays(source) == expected_grid
    assert_portable(({"op": "extend_corner_marked_rays"},), source, expected_grid)
