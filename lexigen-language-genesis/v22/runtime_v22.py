from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any

Grid = tuple[tuple[int, ...], ...]

class SketchError(RuntimeError):
    pass

def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(x) for x in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise SketchError("invalid grid")
    return grid

def background(grid: Grid) -> int:
    counts = Counter(x for row in grid for x in row)
    return min(counts, key=lambda x: (-counts[x], x))

def points_of(grid: Grid, colour: int):
    return [(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x == colour]

def non_background_colours(grid: Grid):
    bg = background(grid)
    return sorted({x for row in grid for x in row if x != bg})

def _paint(values, points, colour):
    for r, c in points:
        values[r][c] = int(colour)
def execute(program: dict[str, Any], source: Grid) -> Grid:
    op = program["op"]
    h, w = len(source), len(source[0])
    if op == "connect_aligned":
        marker = int(program["marker_colour"])
        paint = int(program["paint_colour"])
        bg = background(source)
        values = [list(row) for row in source]
        points = points_of(source, marker)
        for r1, c1 in points:
            for r2, c2 in points:
                if r1 == r2:
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        if values[r1][c] == bg: values[r1][c] = paint
                if c1 == c2:
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        if values[r][c1] == bg: values[r][c1] = paint
        return tuple(tuple(row) for row in values)
    if op == "classify_reflection":
        marker = int(program["marker_colour"])
        yes = int(program["equal_colour"])
        no = int(program["unequal_colour"])
        axis = str(program["axis"])
        values = [[0 for _ in range(w)] for _ in range(h)]
        occupied = set(points_of(source, marker))
        for r, c in occupied:
            rr, cc = (h - 1 - r, c) if axis == "vertical" else (r, w - 1 - c)
            values[r][c] = yes if (rr, cc) in occupied else no
        return tuple(tuple(row) for row in values)
    if op == "pack_columns":
        bg = background(source)
        values = [[bg for _ in range(w)] for _ in range(h)]
        for c in range(w):
            cells = [source[r][c] for r in range(h) if source[r][c] != bg]
            for offset, colour in enumerate(cells):
                values[h - 1 - offset][c] = colour
        return tuple(tuple(row) for row in values)
    raise SketchError(f"unknown op: {op}")


def canonical(program: dict[str, Any]) -> str:
    import json
    return json.dumps(program, sort_keys=True, separators=(",", ":"))
