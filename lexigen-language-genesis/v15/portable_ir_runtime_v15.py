from __future__ import annotations

from collections import Counter
from typing import Any

Grid = tuple[tuple[int, ...], ...]


class PortableIRError(RuntimeError):
    pass


def _grid(value: Any) -> Grid:
    result = tuple(tuple(int(cell) for cell in row) for row in value)
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise PortableIRError("invalid grid")
    return result


def _background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return sorted(counts, key=lambda colour: (-counts[colour], colour))[0]


def _xform(grid: Grid, name: str) -> Grid:
    if name == "identity":
        return grid
    if name == "flip_h":
        return tuple(tuple(row[len(row) - 1 - c] for c in range(len(row))) for row in grid)
    if name == "flip_v":
        return tuple(grid[len(grid) - 1 - r] for r in range(len(grid)))
    if name == "transpose":
        return tuple(tuple(grid[r][c] for r in range(len(grid))) for c in range(len(grid[0])))
    if name == "rotate_180":
        return tuple(tuple(grid[len(grid) - 1 - r][len(grid[0]) - 1 - c] for c in range(len(grid[0]))) for r in range(len(grid)))
    raise PortableIRError("unknown transform")


def _bbox(points):
    points = list(points)
    if not points:
        raise PortableIRError("empty point set")
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _components(grid: Grid, colour: int):
    height, width = len(grid), len(grid[0])
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == colour}
    output = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                point = (r + dr, c + dc)
                if point in unseen:
                    unseen.remove(point)
                    component.add(point)
                    stack.append(point)
        output.append(component)
    return sorted(output, key=lambda item: (min(item), len(item)))


def _runs(length: int, separators):
    blocked = set(separators)
    result = []
    start = 0
    for index in range(length + 1):
        if index == length or index in blocked:
            if start < index:
                result.append((start, index))
            start = index + 1
    return result


def _separator(grid: Grid) -> int:
    height, width = len(grid), len(grid[0])
    candidates = []
    for colour in sorted({cell for row in grid for cell in row}):
        full_rows = [r for r in range(height) if all(cell == colour for cell in grid[r])]
        full_cols = [c for c in range(width) if all(grid[r][c] == colour for r in range(height))]
        row_runs = _runs(height, full_rows)
        col_runs = _runs(width, full_cols)
        if not full_rows or not full_cols or len(row_runs) < 2 or len(col_runs) < 2:
            continue
        if len({end - start for start, end in row_runs}) != 1:
            continue
        if len({end - start for start, end in col_runs}) != 1:
            continue
        candidates.append(colour)
    if len(candidates) != 1:
        raise PortableIRError("separator role is ambiguous")
    return candidates[0]


def _decode(grid: Grid, separator: int) -> Grid:
    height, width = len(grid), len(grid[0])
    row_blocks = _runs(height, [r for r in range(height) if all(value == separator for value in grid[r])])
    col_blocks = _runs(width, [c for c in range(width) if all(grid[r][c] == separator for r in range(height))])
    output = []
    for r0, r1 in row_blocks:
        row = []
        for c0, c1 in col_blocks:
            counts = Counter(
                grid[r][c]
                for r in range(r0, r1)
                for c in range(c0, c1)
                if grid[r][c] != separator
            )
            row.append(separator if not counts else sorted(counts, key=lambda value: (-counts[value], value))[0])
        output.append(tuple(row))
    return tuple(output)


def _tiles(grid: Grid, rows: int, cols: int):
    height, width = len(grid), len(grid[0])
    if height % rows or width % cols:
        raise PortableIRError("uneven partition")
    tile_height, tile_width = height // rows, width // cols
    output = []
    for tile_row in range(rows):
        for tile_col in range(cols):
            output.append(tuple(
                tuple(grid[tile_row * tile_height + r][tile_col * tile_width + c] for c in range(tile_width))
                for r in range(tile_height)
            ))
    return output


def _overlay(tiles, order, background):
    height, width = len(tiles[0]), len(tiles[0][0])
    result = [[background for _ in range(width)] for _ in range(height)]
    for index in order:
        for r in range(height):
            for c in range(width):
                if tiles[index][r][c] != background:
                    result[r][c] = tiles[index][r][c]
    return tuple(tuple(row) for row in result)


def _rect_objects(grid: Grid, mode: str):
    colours = sorted({cell for row in grid for cell in row if cell != 0})
    output = []
    if mode == "colours":
        groups = [
            (colour, {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour})
            for colour in colours
        ]
    elif mode == "components":
        groups = [(colour, component) for colour in colours for component in _components(grid, colour)]
    else:
        raise PortableIRError("unknown rectangle mode")
    for colour, points in groups:
        box = _bbox(points)
        if box[2] - box[0] + 1 >= 3 and box[3] - box[1] + 1 >= 3:
            output.append({"colour": colour, "box": box})
    if len(output) < 2:
        raise PortableIRError("not enough rectangles")
    return output


def _border(box):
    r0, c0, r1, c1 = box
    points = set()
    for c in range(c0, c1 + 1):
        points.add((r0, c))
        points.add((r1, c))
    for r in range(r0, r1 + 1):
        points.add((r, c0))
        points.add((r, c1))
    return points


def _inside(left, right):
    return left != right and left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]
