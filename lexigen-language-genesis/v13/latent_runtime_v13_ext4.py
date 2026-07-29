from __future__ import annotations

from collections import Counter
from typing import Any

from latent_runtime_v13 import Grid, LatentRuntimeError
from latent_runtime_v13_ext2 import _components_of_colour, _legend_lines
from latent_runtime_v13_ext3 import execute_program as execute_previous


def execute_program(program: dict[str, Any], source: Grid) -> Grid:
    operator = str(program.get("operator"))
    parameters = dict(program.get("parameters", {}))
    if operator == "indexed_legend_template_broadcast":
        return indexed_legend_template_broadcast(source, parameters)
    return execute_previous(program, source)


def _attempt_candidate(
    source: Grid,
    legend: list[tuple[int, int, int]],
    key_colour: int,
    index_stride: int,
) -> Grid | None:
    legend_positions = {(row, col) for row, col, _ in legend}
    key_entries = [(row, col) for row, col, colour in legend if colour == key_colour]
    if len(key_entries) != 1:
        return None
    key_row, key_col = key_entries[0]

    components = _components_of_colour(source, key_colour, legend_positions)
    if not components:
        return None
    template = max(
        components,
        key=lambda component: (
            len(component),
            -min(row for row, _ in component),
            -min(col for _, col in component),
        ),
    )

    same_row = len({row for row, _, _ in legend}) == 1
    same_col = len({col for _, col, _ in legend}) == 1
    if same_row == same_col:
        return None
    ordered = sorted(legend, key=(lambda item: item[1]) if same_row else (lambda item: item[0]))
    key_index = next(
        (index for index, (row, col, _) in enumerate(ordered) if (row, col) == (key_row, key_col)),
        None,
    )
    if key_index is None or key_index not in {0, len(ordered) - 1}:
        return None

    canvas = [list(row) for row in source]
    for marker_index, (marker_row, marker_col, colour) in enumerate(ordered):
        distance = abs(marker_index - key_index)
        row_delta = marker_row - key_row
        col_delta = marker_col - key_col
        if same_row and distance:
            direction = 1 if col_delta > 0 else -1
            col_delta += direction * index_stride * distance
        elif same_col and distance:
            direction = 1 if row_delta > 0 else -1
            row_delta += direction * index_stride * distance
        for row, col in template:
            target_row = row + row_delta
            target_col = col + col_delta
            if not (0 <= target_row < len(source) and 0 <= target_col < len(source[0])):
                return None
            canvas[target_row][target_col] = colour
    return tuple(tuple(row) for row in canvas)


def indexed_legend_template_broadcast(source: Grid, parameters: dict[str, Any]) -> Grid:
    key_colour = int(parameters["key_colour"])
    index_stride = int(parameters["index_stride"])
    background = Counter(cell for row in source for cell in row).most_common(1)[0][0]
    candidates = sorted(_legend_lines(source, background), key=lambda item: item[0], reverse=True)
    for _, legend in candidates:
        result = _attempt_candidate(source, legend, key_colour, index_stride)
        if result is not None:
            return result
    raise LatentRuntimeError("no structurally valid indexed legend")
