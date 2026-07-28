from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


class PortableRuntimeError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortableRuntimeError("invalid grid")
    return grid


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def _neighbours(point: Point, height: int, width: int) -> Iterable[Point]:
    row, col = point
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components(grid: Grid, colour: int) -> list[frozenset[Point]]:
    height, width = _shape(grid)
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == colour}
    result: list[frozenset[Point]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        points = {start}
        while queue:
            current = queue.popleft()
            for nxt in _neighbours(current, height, width):
                if nxt in unseen:
                    unseen.remove(nxt)
                    points.add(nxt)
                    queue.append(nxt)
        result.append(frozenset(points))
    return sorted(result, key=lambda points: (min(points), len(points)))


def _normalise(points: Iterable[Point]) -> frozenset[Point]:
    values = list(points)
    min_row = min(row for row, _ in values)
    min_col = min(col for _, col in values)
    return frozenset((row - min_row, col - min_col) for row, col in values)


def _variants(points: Iterable[Point]) -> set[frozenset[Point]]:
    current = list(_normalise(points))
    variants: set[frozenset[Point]] = set()
    for _ in range(4):
        variants.add(_normalise(current))
        variants.add(_normalise((row, -col) for row, col in current))
        current = [(col, -row) for row, col in current]
    return variants


def _bbox(points: frozenset[Point]) -> tuple[int, int]:
    rows = [row for row, _ in points]
    cols = [col for _, col in points]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _matches(left: frozenset[Point], right: frozenset[Point], predicate: dict[str, Any]) -> bool:
    feature = predicate["feature"]
    if feature == "area":
        return len(left) == len(right)
    if feature == "bbox":
        return _bbox(left) == _bbox(right)
    if feature == "normalised_points":
        if predicate.get("symmetry") == "dihedral":
            return _normalise(right) in _variants(left)
        return _normalise(left) == _normalise(right)
    raise PortableRuntimeError("unknown match feature")


def execute_portable(ast: dict[str, Any], grid: Grid) -> Grid:
    if ast.get("schema") != "lexigen-arc-relational-ast-v1":
        raise PortableRuntimeError("unsupported AST")

    height, width = _shape(grid)
    background = int(ast["scene"]["background_colour"])
    frame = int(ast["scene"]["frame_colour"])
    boundary_mode = str(ast["scene"]["hole_boundary"])
    exclude_frame = bool(ast["scene"]["exclude_frame_objects"])
    colour_role = str(ast["scene"].get("object_colour_role", "any"))

    holes: list[frozenset[Point]] = []
    for points in _components(grid, background):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in points):
            continue
        boundary = {
            nxt
            for point in points
            for nxt in _neighbours(point, height, width)
            if nxt not in points
        }
        values = [grid[r][c] for r, c in boundary]
        if not values:
            continue
        if boundary_mode == "all" and all(value == frame for value in values):
            holes.append(points)
        elif boundary_mode == "any" and any(value == frame for value in values):
            holes.append(points)
    holes.sort(key=lambda points: (min(points), len(points)))

    excluded = {background}
    if exclude_frame:
        excluded.add(frame)
    objects: list[tuple[int, frozenset[Point]]] = []
    for colour in sorted({cell for row in grid for cell in row} - excluded):
        components = _components(grid, colour)
        if colour_role == "single_component" and len(components) != 1:
            continue
        objects.extend((colour, points) for points in components)
    objects.sort(key=lambda item: (len(item[1]), item[0], min(item[1])))

    used: set[int] = set()
    pairs: list[tuple[int, frozenset[Point], frozenset[Point]]] = []
    for colour, source in objects:
        candidates = [
            index
            for index, hole in enumerate(holes)
            if index not in used and _matches(source, hole, ast["match"])
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: (min(holes[index]), len(holes[index])))
        used.add(chosen)
        pairs.append((colour, source, holes[chosen]))

    values = [list(row) for row in grid]
    for colour, source, destination in pairs:
        if bool(ast["render"]["erase_source"]):
            for row, col in source:
                values[row][col] = background
        for row, col in destination:
            values[row][col] = colour
    return tuple(tuple(row) for row in values)
