from __future__ import annotations

from collections import Counter
from typing import Any

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
Points = frozenset[Point]


class IndependentRuntimeError(RuntimeError):
    pass


def normalize_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0]:
        raise IndependentRuntimeError("invalid grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise IndependentRuntimeError("ragged grid")
    return grid


def inferred_background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def derived_color(grid: Grid, mode: str) -> int:
    counts = Counter(cell for row in grid for cell in row)
    background = inferred_background(grid)
    candidates = [color for color in counts if color != background]
    if not candidates:
        raise IndependentRuntimeError("no foreground color")
    if mode == "least_non_background":
        return min(candidates, key=lambda color: (counts[color], color))
    if mode == "most_non_background":
        return min(candidates, key=lambda color: (-counts[color], color))
    raise IndependentRuntimeError(f"unknown color mode: {mode}")


def points_of_color(grid: Grid, color: int) -> Points:
    return frozenset(
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell == color
    )


def foreground_points(grid: Grid) -> Points:
    background = inferred_background(grid)
    return frozenset(
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell != background
    )


def point_bounds(points: Points) -> tuple[int, int, int, int]:
    if not points:
        raise IndependentRuntimeError("empty point set")
    rows = [row for row, _ in points]
    cols = [col for _, col in points]
    return min(rows), max(rows), min(cols), max(cols)


def bounding_border(points: Points) -> Points:
    min_row, max_row, min_col, max_col = point_bounds(points)
    return frozenset(
        (row, col)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
        if row in (min_row, max_row) or col in (min_col, max_col)
    )


def dilate_cross(points: Points, height: int, width: int) -> Points:
    result = set(points)
    for row, col in points:
        for nr, nc in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if 0 <= nr < height and 0 <= nc < width:
                result.add((nr, nc))
    return frozenset(result)


def erode_cross(points: Points) -> Points:
    return frozenset(
        (row, col)
        for row, col in points
        if all(
            neighbor in points
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
        )
    )


def enclosed_holes(points: Points) -> Points:
    min_row, max_row, min_col, max_col = point_bounds(points)
    empty = {
        (row, col)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
        if (row, col) not in points
    }
    remaining = set(empty)
    result: set[Point] = set()
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        group = {start}
        touches_boundary = False
        while stack:
            row, col = stack.pop()
            if row in (min_row, max_row) or col in (min_col, max_col):
                touches_boundary = True
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    stack.append(neighbor)
        if not touches_boundary:
            result.update(group)
    return frozenset(result)


def solid_canvas(grid: Grid, color: int) -> Grid:
    return tuple(tuple(color for _ in row) for row in grid)


def paint_points(grid: Grid, points: Points, color: int) -> Grid:
    values = [list(row) for row in grid]
    height = len(values)
    width = len(values[0])
    for row, col in points:
        if 0 <= row < height and 0 <= col < width:
            values[row][col] = color
    return tuple(tuple(row) for row in values)


def crop_to_points(grid: Grid, points: Points) -> Grid:
    inside = frozenset(
        (row, col)
        for row, col in points
        if 0 <= row < len(grid) and 0 <= col < len(grid[0])
    )
    min_row, max_row, min_col, max_col = point_bounds(inside)
    return tuple(
        tuple(row[min_col : max_col + 1])
        for row in grid[min_row : max_row + 1]
    )


def evaluate_independent(
    ast: dict[str, Any],
    value: Any,
    parameters: dict[str, int] | None = None,
) -> Any:
    grid = normalize_grid(value)
    params = parameters or {}

    def evaluate(node: dict[str, Any]) -> Any:
        op = str(node["op"])
        if op == "param_color":
            return int(params[str(node["name"])])
        if op == "input_grid":
            return grid
        if op == "least_non_background":
            return derived_color(grid, op)
        if op == "most_non_background":
            return derived_color(grid, op)
        if op == "points_of_color":
            return points_of_color(grid, int(evaluate(node["colour"])))
        if op == "non_background_points":
            return foreground_points(grid)
        if op == "bbox_border":
            return bounding_border(frozenset(evaluate(node["points"])))
        if op == "dilate4":
            return dilate_cross(
                frozenset(evaluate(node["points"])),
                len(grid),
                len(grid[0]),
            )
        if op == "erode4":
            return erode_cross(frozenset(evaluate(node["points"])))
        if op == "holes":
            return enclosed_holes(frozenset(evaluate(node["points"])))
        if op == "canvas":
            return solid_canvas(grid, int(evaluate(node["colour"])))
        if op == "paint":
            return paint_points(
                normalize_grid(evaluate(node["grid"])),
                frozenset(evaluate(node["points"])),
                int(evaluate(node["colour"])),
            )
        if op == "crop_bbox":
            return crop_to_points(
                normalize_grid(evaluate(node["grid"])),
                frozenset(evaluate(node["points"])),
            )
        raise IndependentRuntimeError(f"unknown operation: {op}")

    return evaluate(ast)
