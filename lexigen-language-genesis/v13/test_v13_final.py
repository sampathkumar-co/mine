from __future__ import annotations

from latent_runtime_v13 import as_grid
from latent_runtime_v13_ext3 import execute_program
from portable_runtime_v13 import execute_portable


def assert_both(program, source, expected):
    source_grid = as_grid(source)
    expected_grid = as_grid(expected)
    assert execute_program(program, source_grid) == expected_grid
    assert execute_portable(program, source) == [list(row) for row in expected_grid]


def test_border_anchors_infer_horizontal_field():
    program = {
        "schema": "lexigen-latent-generator-v1",
        "operator": "periodic_axis_field",
        "parameters": {"background": 0},
    }
    source = [
        [0, 2, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    expected = [
        [0, 2, 0, 0, 3, 0, 0],
        [0, 2, 0, 0, 3, 0, 0],
        [0, 2, 0, 0, 3, 0, 0],
        [0, 2, 0, 0, 3, 0, 0],
    ]
    assert_both(program, source, expected)


def test_corruption_restores_periodic_lattice():
    program = {
        "schema": "lexigen-latent-generator-v1",
        "operator": "reconstruct_periodic_lattice",
        "parameters": {"noise_colour": 6},
    }
    source = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 6, 0, 4, 4, 0],
        [0, 4, 4, 0, 4, 6, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 0, 4, 4, 0],
        [0, 6, 4, 0, 4, 4, 6],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    expected = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 0, 4, 4, 0],
        [0, 4, 4, 0, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 0, 4, 4, 0],
        [0, 4, 4, 0, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    assert_both(program, source, expected)


def test_indexed_legend_broadcast_handles_reverse_axis():
    program = {
        "schema": "lexigen-latent-generator-v1",
        "operator": "indexed_legend_template_broadcast",
        "parameters": {"key_colour": 2, "index_stride": 2},
    }
    source = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 4, 0, 3, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    ]
    expected = [row[:] for row in source]
    template = [(4, 8), (4, 9), (5, 9), (6, 8), (6, 9)]
    legend = [(1, 7, 2), (1, 5, 3), (1, 3, 4), (1, 1, 8)]
    for index, (_, marker_col, colour) in enumerate(legend):
        shift = marker_col - 7 - 2 * index
        for row, col in template:
            expected[row][col + shift] = colour
    assert_both(program, source, expected)


def test_seed_propagation_recolours_connected_mask_only():
    program = {
        "schema": "lexigen-latent-generator-v1",
        "operator": "component_seed_propagation",
        "parameters": {"background": 0, "mask_colour": 5},
    }
    source = [
        [0, 2, 5, 5, 0, 0],
        [0, 0, 0, 5, 0, 3],
        [0, 5, 5, 5, 0, 5],
        [0, 5, 0, 0, 0, 5],
    ]
    expected = [
        [0, 0, 2, 2, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 3],
        [0, 2, 0, 0, 0, 3],
    ]
    assert_both(program, source, expected)
