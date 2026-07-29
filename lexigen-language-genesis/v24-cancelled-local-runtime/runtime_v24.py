from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
PointSet = frozenset[Point]


class RuntimeV24Error(RuntimeError):
    pass


def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise RuntimeV24Error("invalid grid")
    return grid


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def colour_points(grid: Grid, colour: int) -> PointSet:
    return frozenset(
        (row, col)
        for row, values in enumerate(grid)
        for col, cell in enumerate(values)
        if cell == colour
    )


def components(points: PointSet) -> tuple[PointSet, ...]:
    remaining = set(points)
    groups: list[PointSet] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending = [start]
        group = {start}
        while pending:
            row, col = pending.pop()
            for neighbour in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    group.add(neighbour)
                    pending.append(neighbour)
        groups.append(frozenset(group))
    return tuple(sorted(groups, key=lambda group: (min(group), len(group))))


def filter_components(points: PointSet, mode: str) -> PointSet:
    groups = components(points)
    if not groups:
        raise RuntimeV24Error("empty selector")
    if mode == "all":
        selected = groups
    elif mode == "largest":
        size = max(map(len, groups))
        selected = tuple(group for group in groups if len(group) == size)
    elif mode == "smallest":
        size = min(map(len, groups))
        selected = tuple(group for group in groups if len(group) == size)
    elif mode == "singletons":
        selected = tuple(group for group in groups if len(group) == 1)
    elif mode == "non_singletons":
        selected = tuple(group for group in groups if len(group) > 1)
    else:
        raise RuntimeV24Error(f"unknown component filter: {mode}")
    result = frozenset(point for group in selected for point in group)
    if not result:
        raise RuntimeV24Error("component filter selected nothing")
    return result


def bounds(points: PointSet) -> tuple[int, int, int, int]:
    if not points:
        raise RuntimeV24Error("empty point set")
    rows = [row for row, _ in points]
    cols = [col for _, col in points]
    return min(rows), max(rows), min(cols), max(cols)


def transform_points(points: PointSet, name: str, height: int, width: int) -> PointSet:
    min_row, max_row, min_col, max_col = bounds(points)
    if name == "identity":
        mapped = points
    elif name == "grid_flip_h":
        mapped = {(row, width - 1 - col) for row, col in points}
    elif name == "grid_flip_v":
        mapped = {(height - 1 - row, col) for row, col in points}
    elif name == "grid_rotate180":
        mapped = {(height - 1 - row, width - 1 - col) for row, col in points}
    elif name == "bbox_reflect_left":
        mapped = {(row, 2 * min_col - 1 - col) for row, col in points}
    elif name == "bbox_reflect_right":
        mapped = {(row, 2 * max_col + 1 - col) for row, col in points}
    elif name == "bbox_reflect_top":
        mapped = {(2 * min_row - 1 - row, col) for row, col in points}
    elif name == "bbox_reflect_bottom":
        mapped = {(2 * max_row + 1 - row, col) for row, col in points}
    else:
        raise RuntimeV24Error(f"unknown transform: {name}")
    return frozenset(mapped)


def region_points(points: PointSet, mode: str) -> PointSet:
    if not points:
        raise RuntimeV24Error("empty transformed set")
    if mode == "points":
        return points
    min_row, max_row, min_col, max_col = bounds(points)
    if mode == "bbox_fill":
        return frozenset(
            (row, col)
            for row in range(min_row, max_row + 1)
            for col in range(min_col, max_col + 1)
        )
    if mode == "bbox_border":
        return frozenset(
            (row, col)
            for row in range(min_row, max_row + 1)
            for col in range(min_col, max_col + 1)
            if row in (min_row, max_row) or col in (min_col, max_col)
        )
    if mode == "row_span":
        by_row: dict[int, list[int]] = defaultdict(list)
        for row, col in points:
            by_row[row].append(col)
        return frozenset(
            (row, col)
            for row, cols in by_row.items()
            for col in range(min(cols), max(cols) + 1)
        )
    if mode == "col_span":
        by_col: dict[int, list[int]] = defaultdict(list)
        for row, col in points:
            by_col[col].append(row)
        return frozenset(
            (row, col)
            for col, rows in by_col.items()
            for row in range(min(rows), max(rows) + 1)
        )
    if mode == "connect_aligned":
        result = set(points)
        ordered = sorted(points)
        for first in ordered:
            for second in ordered:
                row1, col1 = first
                row2, col2 = second
                if row1 == row2:
                    result.update((row1, col) for col in range(min(col1, col2), max(col1, col2) + 1))
                if col1 == col2:
                    result.update((row, col1) for row in range(min(row1, row2), max(row1, row2) + 1))
        return frozenset(result)
    raise RuntimeV24Error(f"unknown region mode: {mode}")


def blank_like(grid: Grid) -> list[list[int]]:
    fill = background(grid)
    return [[fill for _ in row] for row in grid]


def paint(values: list[list[int]], points: Iterable[Point], colour: int) -> None:
    height = len(values)
    width = len(values[0])
    for row, col in points:
        if 0 <= row < height and 0 <= col < width:
            values[row][col] = int(colour)


def execute_paint_edit(program: dict[str, Any], source: Grid) -> Grid:
    height, width = len(source), len(source[0])
    source_colour = int(program["source_colour"])
    selected = filter_components(
        colour_points(source, source_colour),
        str(program["component_filter"]),
    )
    mapped = transform_points(selected, str(program["transform"]), height, width)
    region = region_points(mapped, str(program["region_mode"]))
    if program["combine_mode"] == "source_union_mapped":
        region = frozenset(set(region) | set(selected))
    elif program["combine_mode"] != "mapped_only":
        raise RuntimeV24Error("unknown combine mode")
    values = [list(row) for row in source] if program["base_mode"] == "input" else blank_like(source)
    if program["paint_mode"] == "source_colour":
        colour = source_colour
    elif program["paint_mode"] == "literal_colour":
        colour = int(program["paint_colour"])
    else:
        raise RuntimeV24Error("unknown paint mode")
    paint(values, region, colour)
    return tuple(tuple(row) for row in values)



def transform_single_point(point: Point, name: str, height: int, width: int, reference: PointSet) -> Point:
    row, col = point
    min_row, max_row, min_col, max_col = bounds(reference)
    if name == "identity": return point
    if name == "grid_flip_h": return row, width - 1 - col
    if name == "grid_flip_v": return height - 1 - row, col
    if name == "grid_rotate180": return height - 1 - row, width - 1 - col
    if name == "bbox_reflect_left": return row, 2 * min_col - 1 - col
    if name == "bbox_reflect_right": return row, 2 * max_col + 1 - col
    if name == "bbox_reflect_top": return 2 * min_row - 1 - row, col
    if name == "bbox_reflect_bottom": return 2 * max_row + 1 - row, col
    raise RuntimeV24Error(f"unknown transform: {name}")

def execute_relational_classify(program: dict[str, Any], source: Grid) -> Grid:
    height, width = len(source), len(source[0])
    source_colour = int(program["source_colour"])
    selected = filter_components(
        colour_points(source, source_colour),
        str(program["component_filter"]),
    )
    relation = str(program["relation"])
    values = [list(row) for row in source] if program["base_mode"] == "input" else blank_like(source)
    equal_colour = int(program["equal_colour"])
    unequal_colour = int(program["unequal_colour"])
    for point in selected:
        related_point = transform_single_point(point, relation, height, width, selected)
        values[point[0]][point[1]] = equal_colour if related_point in selected else unequal_colour
    return tuple(tuple(row) for row in values)


def packed_line(cells: list[int], size: int, direction: str, fill: int) -> list[int]:
    result = [fill] * size
    if direction == "start":
        result[: len(cells)] = cells
    elif direction == "end":
        result[size - len(cells) :] = cells
    else:
        raise RuntimeV24Error("unknown packing direction")
    return result


def execute_gravity_pack(program: dict[str, Any], source: Grid) -> Grid:
    height, width = len(source), len(source[0])
    fill = background(source)
    values = [list(row) for row in source] if program["base_mode"] == "input" else blank_like(source)
    axis = str(program["axis"])
    direction = str(program["direction"])
    if axis == "rows":
        for row in range(height):
            cells = [cell for cell in source[row] if cell != fill]
            packed = packed_line(cells, width, direction, fill)
            for col, cell in enumerate(packed):
                if program["base_mode"] == "background_canvas" or cell != fill:
                    values[row][col] = cell
    elif axis == "columns":
        for col in range(width):
            cells = [source[row][col] for row in range(height) if source[row][col] != fill]
            packed = packed_line(cells, height, direction, fill)
            for row, cell in enumerate(packed):
                if program["base_mode"] == "background_canvas" or cell != fill:
                    values[row][col] = cell
    else:
        raise RuntimeV24Error("unknown packing axis")
    return tuple(tuple(row) for row in values)


def execute(program: dict[str, Any], value: Any) -> Grid:
    source = as_grid(value)
    op = str(program.get("op"))
    if op == "paint_edit":
        return execute_paint_edit(program, source)
    if op == "relational_classify":
        return execute_relational_classify(program, source)
    if op == "gravity_pack":
        return execute_gravity_pack(program, source)
    raise RuntimeV24Error(f"unknown operation: {op}")
