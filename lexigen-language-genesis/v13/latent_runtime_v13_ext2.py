from __future__ import annotations

from collections import Counter, deque
from typing import Any

from latent_runtime_v13 import Grid, LatentRuntimeError
from latent_runtime_v13_ext import execute_program as execute_previous


def execute_program(program: dict[str, Any], source: Grid) -> Grid:
    operator = str(program.get("operator"))
    parameters = dict(program.get("parameters", {}))
    if operator == "reconstruct_periodic_lattice":
        return reconstruct_periodic_lattice(source, parameters)
    if operator == "template_colour_broadcast_invariant_v2":
        return template_colour_broadcast_invariant_v2(source, parameters)
    return execute_previous(program, source)


def reconstruct_periodic_lattice(source: Grid, parameters: dict[str, Any]) -> Grid:
    noise = int(parameters["noise_colour"])
    height, width = len(source), len(source[0])
    visible = [cell for row in source for cell in row if cell != noise]
    if len(set(visible)) != 2:
        raise LatentRuntimeError("periodic lattice requires exactly two non-noise colours")
    counts = Counter(visible)
    background = max(counts, key=lambda colour: (counts[colour], -colour))
    foreground = next(colour for colour in counts if colour != background)
    candidates: list[tuple[tuple[int, ...], Grid]] = []
    for length in range(1, min(4, height + 1, width + 1)):
        for spacing in range(length + 1, min(7, max(height, width) + 1)):
            for row_phase in range(spacing):
                for col_phase in range(spacing):
                    canvas = [[background for _ in range(width)] for _ in range(height)]
                    for top in range(row_phase, height, spacing):
                        for left in range(col_phase, width, spacing):
                            for row in range(top, min(top + length, height)):
                                for col in range(left, min(left + length, width)):
                                    canvas[row][col] = foreground
                    compatible = True
                    observed_foreground = 0
                    expected_foreground = 0
                    for row in range(height):
                        for col in range(width):
                            expected = canvas[row][col]
                            value = source[row][col]
                            if expected == foreground:
                                expected_foreground += 1
                            if value == foreground:
                                observed_foreground += 1
                            if value != noise and value != expected:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if not compatible or observed_foreground == 0:
                        continue
                    if expected_foreground < observed_foreground:
                        continue
                    missing = expected_foreground - observed_foreground
                    rank = (
                        missing,
                        spacing,
                        length,
                        row_phase,
                        col_phase,
                    )
                    candidates.append((rank, tuple(tuple(row) for row in canvas)))
    if not candidates:
        raise LatentRuntimeError("no compatible periodic lattice")
    return min(candidates, key=lambda item: item[0])[1]


def _components_of_colour(source: Grid, colour: int, excluded: set[tuple[int, int]]):
    points = {
        (row, col)
        for row in range(len(source))
        for col in range(len(source[0]))
        if source[row][col] == colour and (row, col) not in excluded
    }
    components = []
    while points:
        start = next(iter(points))
        points.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in points:
                    points.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _legend_lines(source: Grid, background: int):
    height, width = len(source), len(source[0])
    candidates = []
    for row in range(height):
        entries = [(row, col, source[row][col]) for col in range(width) if source[row][col] != background]
        diversity = len({colour for _, _, colour in entries})
        if len(entries) >= 3 and diversity >= 2:
            candidates.append(((diversity, len(entries), -row, 1), entries))
    for col in range(width):
        entries = [(row, col, source[row][col]) for row in range(height) if source[row][col] != background]
        diversity = len({colour for _, _, colour in entries})
        if len(entries) >= 3 and diversity >= 2:
            candidates.append(((diversity, len(entries), -col, 0), entries))
    return candidates


def template_colour_broadcast_invariant_v2(source: Grid, parameters: dict[str, Any]) -> Grid:
    key_colour = int(parameters["key_colour"])
    background = Counter(cell for row in source for cell in row).most_common(1)[0][0]
    candidates = _legend_lines(source, background)
    if not candidates:
        raise LatentRuntimeError("legend line not found")
    _, legend = max(candidates, key=lambda item: item[0])
    legend_positions = {(row, col) for row, col, _ in legend}
    key_markers = [(row, col) for row, col, colour in legend if colour == key_colour]
    if len(key_markers) != 1:
        raise LatentRuntimeError("key marker is ambiguous")
    components = _components_of_colour(source, key_colour, legend_positions)
    if not components:
        raise LatentRuntimeError("key template missing")
    template = max(components, key=lambda component: (len(component), -min(component)))
    anchor_row = min(row for row, _ in template)
    anchor_col = min(col for _, col in template)
    key_row, key_col = key_markers[0]
    canvas = [list(row) for row in source]
    for marker_row, marker_col, colour in legend:
        delta_row = marker_row - key_row
        delta_col = marker_col - key_col
        for row, col in template:
            target_row = row + delta_row
            target_col = col + delta_col
            if 0 <= target_row < len(source) and 0 <= target_col < len(source[0]):
                canvas[target_row][target_col] = colour
    return tuple(tuple(row) for row in canvas)
