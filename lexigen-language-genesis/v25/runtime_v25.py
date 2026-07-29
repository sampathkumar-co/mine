from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
PointSet = frozenset[Point]
ObjectSet = tuple[PointSet, ...]


class RuntimeV25Error(RuntimeError):
    pass


def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise RuntimeV25Error("invalid grid")
    return grid


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def colour_counts(grid: Grid) -> Counter[int]:
    return Counter(cell for row in grid for cell in row)


def background(grid: Grid) -> int:
    counts = colour_counts(grid)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def derived_colour(grid: Grid, mode: str) -> int:
    counts = colour_counts(grid)
    bg = background(grid)
    candidates = [colour for colour in counts if colour != bg]
    if not candidates:
        raise RuntimeV25Error("no non-background colour")
    if mode == "least_non_background":
        return min(candidates, key=lambda colour: (counts[colour], colour))
    if mode == "most_non_background":
        return min(candidates, key=lambda colour: (-counts[colour], colour))
    raise RuntimeV25Error(f"unknown derived colour: {mode}")


def colour_points(grid: Grid, colour: int) -> PointSet:
    return frozenset(
        (row, col)
        for row, values in enumerate(grid)
        for col, cell in enumerate(values)
        if cell == colour
    )


def non_background_points(grid: Grid) -> PointSet:
    return frozenset(
        (row, col)
        for row, values in enumerate(grid)
        for col, cell in enumerate(values)
        if cell != background(grid)
    )


def components4(points: PointSet) -> ObjectSet:
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


def bounds(points: PointSet) -> tuple[int, int, int, int]:
    if not points:
        raise RuntimeV25Error("empty point set")
    rows = [row for row, _ in points]
    cols = [col for _, col in points]
    return min(rows), max(rows), min(cols), max(cols)


def object_feature(points: PointSet, feature: str) -> int:
    min_row, max_row, min_col, max_col = bounds(points)
    if feature == "size":
        return len(points)
    if feature == "bbox_area":
        return (max_row - min_row + 1) * (max_col - min_col + 1)
    if feature == "bbox_width":
        return max_col - min_col + 1
    if feature == "bbox_height":
        return max_row - min_row + 1
    raise RuntimeV25Error(f"unknown object feature: {feature}")


def select_objects(objects: ObjectSet, feature: str, extremum: str) -> ObjectSet:
    if not objects:
        raise RuntimeV25Error("empty object set")
    if extremum == "all":
        return objects
    values = [object_feature(obj, feature) for obj in objects]
    target = min(values) if extremum == "minimum" else max(values) if extremum == "maximum" else None
    if target is None:
        raise RuntimeV25Error(f"unknown extremum: {extremum}")
    return tuple(obj for obj, value in zip(objects, values) if value == target)


def select_position(objects: ObjectSet, direction: str) -> ObjectSet:
    if not objects:
        raise RuntimeV25Error("empty object set")
    boxes = [bounds(obj) for obj in objects]
    if direction == "topmost":
        values = [box[0] for box in boxes]
        target = min(values)
    elif direction == "bottommost":
        values = [box[1] for box in boxes]
        target = max(values)
    elif direction == "leftmost":
        values = [box[2] for box in boxes]
        target = min(values)
    elif direction == "rightmost":
        values = [box[3] for box in boxes]
        target = max(values)
    else:
        raise RuntimeV25Error(f"unknown direction: {direction}")
    return tuple(obj for obj, value in zip(objects, values) if value == target)


def objects_to_points(objects: ObjectSet) -> PointSet:
    return frozenset(point for obj in objects for point in obj)


def transform_points(points: PointSet, mode: str, height: int, width: int) -> PointSet:
    min_row, max_row, min_col, max_col = bounds(points)
    if mode == "grid_flip_h":
        result = {(row, width - 1 - col) for row, col in points}
    elif mode == "grid_flip_v":
        result = {(height - 1 - row, col) for row, col in points}
    elif mode == "grid_rotate180":
        result = {(height - 1 - row, width - 1 - col) for row, col in points}
    elif mode == "bbox_reflect_left":
        result = {(row, 2 * min_col - 1 - col) for row, col in points}
    elif mode == "bbox_reflect_right":
        result = {(row, 2 * max_col + 1 - col) for row, col in points}
    elif mode == "bbox_reflect_top":
        result = {(2 * min_row - 1 - row, col) for row, col in points}
    elif mode == "bbox_reflect_bottom":
        result = {(2 * max_row + 1 - row, col) for row, col in points}
    else:
        raise RuntimeV25Error(f"unknown point transform: {mode}")
    return frozenset(result)


def bbox_fill(points: PointSet) -> PointSet:
    min_row, max_row, min_col, max_col = bounds(points)
    return frozenset(
        (row, col)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
    )


def bbox_border(points: PointSet) -> PointSet:
    min_row, max_row, min_col, max_col = bounds(points)
    return frozenset(
        (row, col)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
        if row in (min_row, max_row) or col in (min_col, max_col)
    )


def row_span(points: PointSet) -> PointSet:
    by_row: dict[int, list[int]] = defaultdict(list)
    for row, col in points:
        by_row[row].append(col)
    return frozenset(
        (row, col)
        for row, cols in by_row.items()
        for col in range(min(cols), max(cols) + 1)
    )


def col_span(points: PointSet) -> PointSet:
    by_col: dict[int, list[int]] = defaultdict(list)
    for row, col in points:
        by_col[col].append(row)
    return frozenset(
        (row, col)
        for col, rows in by_col.items()
        for row in range(min(rows), max(rows) + 1)
    )


def connect_aligned(points: PointSet) -> PointSet:
    result = set(points)
    ordered = sorted(points)
    for row1, col1 in ordered:
        for row2, col2 in ordered:
            if row1 == row2:
                result.update((row1, col) for col in range(min(col1, col2), max(col1, col2) + 1))
            if col1 == col2:
                result.update((row, col1) for row in range(min(row1, row2), max(row1, row2) + 1))
    return frozenset(result)


def holes(points: PointSet) -> PointSet:
    min_row, max_row, min_col, max_col = bounds(points)
    empty = {
        (row, col)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
        if (row, col) not in points
    }
    result: set[Point] = set()
    remaining = set(empty)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending = [start]
        group = {start}
        touches = False
        while pending:
            row, col = pending.pop()
            touches |= row in (min_row, max_row) or col in (min_col, max_col)
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    group.add(neighbour)
                    pending.append(neighbour)
        if not touches:
            result.update(group)
    return frozenset(result)


def outline4(points: PointSet) -> PointSet:
    return frozenset(
        (row, col)
        for row, col in points
        if any(
            neighbour not in points
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        )
    )


def dilate4(points: PointSet, height: int, width: int) -> PointSet:
    result = set(points)
    for row, col in points:
        for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            nr, nc = neighbour
            if 0 <= nr < height and 0 <= nc < width:
                result.add(neighbour)
    return frozenset(result)


def erode4(points: PointSet) -> PointSet:
    return frozenset(
        (row, col)
        for row, col in points
        if all(
            neighbour in points
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        )
    )


def canvas(grid: Grid, colour: int) -> Grid:
    return tuple(tuple(int(colour) for _ in row) for row in grid)


def paint(grid: Grid, points: PointSet, colour: int) -> Grid:
    values = [list(row) for row in grid]
    height, width = len(values), len(values[0])
    for row, col in points:
        if 0 <= row < height and 0 <= col < width:
            values[row][col] = int(colour)
    return tuple(tuple(row) for row in values)


def flip_grid_h(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def flip_grid_v(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def rotate_grid_180(grid: Grid) -> Grid:
    return flip_grid_v(flip_grid_h(grid))


def crop_bbox(grid: Grid, points: PointSet) -> Grid:
    in_bounds = frozenset(
        (row, col)
        for row, col in points
        if 0 <= row < len(grid) and 0 <= col < len(grid[0])
    )
    min_row, max_row, min_col, max_col = bounds(in_bounds)
    return tuple(tuple(row[min_col : max_col + 1]) for row in grid[min_row : max_row + 1])


def eval_ast(ast: dict[str, Any], value: Any, parameters: dict[str, int] | None = None) -> Any:
    grid = as_grid(value)
    params = parameters or {}

    def evaluate(node: dict[str, Any]) -> Any:
        op = str(node["op"])
        if op == "literal_color":
            return int(node["value"])
        if op == "param_color":
            return int(params[str(node["name"])])
        if op == "background":
            return background(grid)
        if op in {"least_non_background", "most_non_background"}:
            return derived_colour(grid, op)
        if op == "input_grid":
            return grid
        if op == "canvas":
            return canvas(grid, int(evaluate(node["colour"])))
        if op == "points_of_color":
            return colour_points(grid, int(evaluate(node["colour"])))
        if op == "non_background_points":
            return non_background_points(grid)
        if op == "components4":
            return components4(frozenset(evaluate(node["points"])))
        if op == "select_objects":
            return select_objects(
                tuple(evaluate(node["objects"])),
                str(node["feature"]),
                str(node["extremum"]),
            )
        if op == "select_position":
            return select_position(tuple(evaluate(node["objects"])), str(node["direction"]))
        if op == "objects_to_points":
            return objects_to_points(tuple(evaluate(node["objects"])))
        if op in {"union", "intersection", "difference"}:
            left = frozenset(evaluate(node["left"]))
            right = frozenset(evaluate(node["right"]))
            if op == "union":
                return frozenset(set(left) | set(right))
            if op == "intersection":
                return frozenset(set(left) & set(right))
            return frozenset(set(left) - set(right))
        points = frozenset(evaluate(node["points"])) if "points" in node else None
        if op == "bbox_fill":
            return bbox_fill(points)
        if op == "bbox_border":
            return bbox_border(points)
        if op == "row_span":
            return row_span(points)
        if op == "col_span":
            return col_span(points)
        if op == "connect_aligned":
            return connect_aligned(points)
        if op == "holes":
            return holes(points)
        if op == "outline4":
            return outline4(points)
        if op == "dilate4":
            return dilate4(points, len(grid), len(grid[0]))
        if op == "erode4":
            return erode4(points)
        if op in {
            "grid_flip_h_points",
            "grid_flip_v_points",
            "grid_rotate180_points",
            "bbox_reflect_left",
            "bbox_reflect_right",
            "bbox_reflect_top",
            "bbox_reflect_bottom",
        }:
            mapping = {
                "grid_flip_h_points": "grid_flip_h",
                "grid_flip_v_points": "grid_flip_v",
                "grid_rotate180_points": "grid_rotate180",
            }
            return transform_points(points, mapping.get(op, op), len(grid), len(grid[0]))
        if op == "paint":
            return paint(
                as_grid(evaluate(node["grid"])),
                frozenset(evaluate(node["points"])),
                int(evaluate(node["colour"])),
            )
        if op == "grid_flip_h":
            return flip_grid_h(as_grid(evaluate(node["grid"])))
        if op == "grid_flip_v":
            return flip_grid_v(as_grid(evaluate(node["grid"])))
        if op == "grid_rotate180":
            return rotate_grid_180(as_grid(evaluate(node["grid"])))
        if op == "crop_bbox":
            return crop_bbox(as_grid(evaluate(node["grid"])), frozenset(evaluate(node["points"])))
        raise RuntimeV25Error(f"unknown AST operation: {op}")

    return evaluate(ast)
