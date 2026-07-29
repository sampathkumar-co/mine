from __future__ import annotations

from collections import Counter
from typing import Any

from latent_runtime_v13 import (
    Grid,
    LatentRuntimeError,
    _components,
    execute_program as execute_base,
    most_common_colour,
)


def execute_program(program: dict[str, Any], source: Grid) -> Grid:
    operator = str(program.get("operator"))
    parameters = dict(program.get("parameters", {}))
    if operator == "periodic_axis_field":
        return periodic_axis_field(source, parameters)
    if operator == "replace_colour_with_background":
        return replace_colour_with_background(source, parameters)
    if operator == "ordered_colour_summary":
        return ordered_colour_summary(source, parameters)
    if operator == "reflect_component_positions":
        return reflect_component_positions(source, parameters)
    if operator == "template_colour_broadcast_invariant":
        return template_colour_broadcast_invariant(source, parameters)
    return execute_base(program, source)


def periodic_axis_field(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    points = [(row, col, value) for row, line in enumerate(source) for col, value in enumerate(line) if value != background]
    if len(points) != 2:
        raise LatentRuntimeError("periodic axis field requires two anchors")
    height, width = len(source), len(source[0])
    first_point, second_point = points
    border_rows = all(row in {0, height - 1} for row, _, _ in points)
    border_cols = all(col in {0, width - 1} for _, col, _ in points)
    canvas = [[background for _ in range(width)] for _ in range(height)]
    if border_rows or first_point[0] == second_point[0]:
        anchors = sorted((col, colour) for _, col, colour in points)
        step = anchors[1][0] - anchors[0][0]
        if step <= 0:
            raise LatentRuntimeError("invalid column period")
        for index, col in enumerate(range(anchors[0][0], width, step)):
            colour = anchors[index % 2][1]
            for row in range(height):
                canvas[row][col] = colour
    elif border_cols or first_point[1] == second_point[1]:
        anchors = sorted((row, colour) for row, _, colour in points)
        step = anchors[1][0] - anchors[0][0]
        if step <= 0:
            raise LatentRuntimeError("invalid row period")
        for index, row in enumerate(range(anchors[0][0], height, step)):
            colour = anchors[index % 2][1]
            for col in range(width):
                canvas[row][col] = colour
    else:
        raise LatentRuntimeError("anchors do not identify a border frame axis")
    return tuple(tuple(row) for row in canvas)


def replace_colour_with_background(source: Grid, parameters: dict[str, Any]) -> Grid:
    old = int(parameters["old"])
    background = most_common_colour(source)
    return tuple(tuple(background if value == old else value for value in row) for row in source)


def ordered_colour_summary(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    output_height = int(parameters["output_height"])
    output_width = int(parameters["output_width"])
    by_colour: dict[int, list[tuple[int, int]]] = {}
    for row in range(len(source)):
        for col in range(len(source[0])):
            colour = source[row][col]
            if colour != background:
                by_colour.setdefault(colour, []).append((row, col))
    ordered = sorted(
        (
            min(row for row, _ in points),
            min(col for _, col in points),
            colour,
        )
        for colour, points in by_colour.items()
    )
    if len(ordered) > output_height:
        raise LatentRuntimeError("too many colour trajectories")
    canvas = [[background for _ in range(output_width)] for _ in range(output_height)]
    offset = output_height - len(ordered)
    for index, (_, _, colour) in enumerate(ordered):
        for col in range(output_width):
            canvas[offset + index][col] = colour
    return tuple(tuple(row) for row in canvas)


def reflect_component_positions(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    width = len(source[0])
    canvas = [[background for _ in row] for row in source]
    for component in _components(source, background):
        min_row = min(row for row, _ in component)
        min_col = min(col for _, col in component)
        max_col = max(col for _, col in component)
        component_width = max_col - min_col + 1
        target_left = width - (min_col + component_width)
        for row, col in component:
            target_row = row
            target_col = target_left + (col - min_col)
            canvas[target_row][target_col] = source[row][col]
    return tuple(tuple(row) for row in canvas)


def _legend_candidates(source: Grid, background: int):
    height, width = len(source), len(source[0])
    candidates = []
    for row in range(height):
        points = [(row, col, source[row][col]) for col in range(width) if source[row][col] != background]
        if len(points) >= 3 and len({value for _, _, value in points}) >= 2:
            candidates.append((len(points), "row", row, points))
    for col in range(width):
        points = [(row, col, source[row][col]) for row in range(height) if source[row][col] != background]
        if len(points) >= 3 and len({value for _, _, value in points}) >= 2:
            candidates.append((len(points), "col", col, points))
    return candidates


def template_colour_broadcast_invariant(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    key_colour = int(parameters["key_colour"])
    candidates = _legend_candidates(source, background)
    if not candidates:
        raise LatentRuntimeError("legend line not found")
    _, axis, axis_index, legend = max(candidates, key=lambda item: (item[0], -item[2]))
    key_markers = [(row, col) for row, col, colour in legend if colour == key_colour]
    if len(key_markers) != 1:
        raise LatentRuntimeError("key marker is ambiguous")
    key_marker = key_markers[0]
    legend_positions = {(row, col) for row, col, _ in legend}
    template = [
        (row, col)
        for row in range(len(source))
        for col in range(len(source[0]))
        if source[row][col] == key_colour and (row, col) not in legend_positions
    ]
    if not template:
        raise LatentRuntimeError("key template missing")
    canvas = [list(row) for row in source]
    for marker_row, marker_col, colour in legend:
        delta_row = marker_row - key_marker[0]
        delta_col = marker_col - key_marker[1]
        for row, col in template:
            target_row = row + delta_row
            target_col = col + delta_col
            if 0 <= target_row < len(source) and 0 <= target_col < len(source[0]):
                canvas[target_row][target_col] = colour
    return tuple(tuple(row) for row in canvas)
