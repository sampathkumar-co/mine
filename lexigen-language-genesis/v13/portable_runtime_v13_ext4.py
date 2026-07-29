from __future__ import annotations

from collections import Counter, deque
from typing import Any, Sequence

from portable_runtime_v13 import PortableGrid, PortableLatentError, as_portable, execute_portable as execute_previous

Point = tuple[int, int]


def execute_portable(program: dict[str, Any], value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = as_portable(value)
    if str(program.get("operator")) == "indexed_legend_template_broadcast":
        return indexed_broadcast(grid, dict(program.get("parameters", {})))
    return execute_previous(program, value)


def _legend_candidates(grid: PortableGrid, background: int):
    candidates = []
    for row in range(len(grid)):
        entries = [(row, col, grid[row][col]) for col in range(len(grid[0])) if grid[row][col] != background]
        diversity = len({value for _, _, value in entries})
        if len(entries) >= 3 and diversity >= 2:
            candidates.append(((diversity, len(entries), -row, 1), entries))
    for col in range(len(grid[0])):
        entries = [(row, col, grid[row][col]) for row in range(len(grid)) if grid[row][col] != background]
        diversity = len({value for _, _, value in entries})
        if len(entries) >= 3 and diversity >= 2:
            candidates.append(((diversity, len(entries), -col, 0), entries))
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def _components(grid: PortableGrid, colour: int, excluded: set[Point]):
    points = {
        (row, col)
        for row in range(len(grid))
        for col in range(len(grid[0]))
        if grid[row][col] == colour and (row, col) not in excluded
    }
    result = []
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
        result.append(component)
    return result


def _attempt(grid: PortableGrid, legend, key_colour: int, stride: int) -> PortableGrid | None:
    excluded = {(row, col) for row, col, _ in legend}
    key_entries = [(row, col) for row, col, colour in legend if colour == key_colour]
    if len(key_entries) != 1:
        return None
    key_row, key_col = key_entries[0]
    components = _components(grid, key_colour, excluded)
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

    output = [row[:] for row in grid]
    for index, (marker_row, marker_col, colour) in enumerate(ordered):
        distance = abs(index - key_index)
        row_delta = marker_row - key_row
        col_delta = marker_col - key_col
        if same_row and distance:
            col_delta += (1 if col_delta > 0 else -1) * stride * distance
        elif same_col and distance:
            row_delta += (1 if row_delta > 0 else -1) * stride * distance
        for row, col in template:
            target_row = row + row_delta
            target_col = col + col_delta
            if not (0 <= target_row < len(grid) and 0 <= target_col < len(grid[0])):
                return None
            output[target_row][target_col] = colour
    return output


def indexed_broadcast(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    key_colour = int(params["key_colour"])
    stride = int(params["index_stride"])
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    for _, legend in _legend_candidates(grid, background):
        result = _attempt(grid, legend, key_colour, stride)
        if result is not None:
            return result
    raise PortableLatentError("no structurally valid indexed legend")
