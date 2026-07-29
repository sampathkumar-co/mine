from __future__ import annotations
from typing import Any

class PortableSketchError(RuntimeError):
    pass

def _bg(grid):
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda value: (-counts[value], value))[0]

def run(program: dict[str, Any], raw):
    grid = [list(map(int, row)) for row in raw]
    rows, cols = len(grid), len(grid[0])
    op = program["op"]
    if op == "connect_aligned":
        marker, paint, bg = int(program["marker_colour"]), int(program["paint_colour"]), _bg(grid)
        output = [row[:] for row in grid]
        pts = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == marker]
        for r, c in pts:
            for rr, cc in pts:
                if r == rr:
                    for x in range(min(c, cc), max(c, cc) + 1):
                        if output[r][x] == bg: output[r][x] = paint
                elif c == cc:
                    for y in range(min(r, rr), max(r, rr) + 1):
                        if output[y][c] == bg: output[y][c] = paint
        return output
    if op == "classify_reflection":
        marker = int(program["marker_colour"])
        yes, no = int(program["equal_colour"]), int(program["unequal_colour"])
        vertical = str(program["axis"]) == "vertical"
        pts = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == marker}
        output = [[0] * cols for _ in range(rows)]
        for r, c in pts:
            partner = (rows - 1 - r, c) if vertical else (r, cols - 1 - c)
            output[r][c] = yes if partner in pts else no
        return output
    if op == "pack_columns":
        bg = _bg(grid)
        output = [[bg] * cols for _ in range(rows)]
        for c in range(cols):
            stack = []
            for r in range(rows):
                if grid[r][c] != bg:
                    stack.append(grid[r][c])
            for i, value in enumerate(stack):
                output[rows - 1 - i][c] = value
        return output
    raise PortableSketchError(op)
