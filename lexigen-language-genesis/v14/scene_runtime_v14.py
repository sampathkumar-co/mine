from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


class SceneRuntimeError(RuntimeError):
    pass


def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise SceneRuntimeError("grid must be non-empty and rectangular")
    return grid


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda value: (-counts[value], value))


def transform(grid: Grid, name: str) -> Grid:
    if name == "identity":
        return grid
    if name == "flip_h":
        return tuple(tuple(reversed(row)) for row in grid)
    if name == "flip_v":
        return tuple(reversed(grid))
    if name == "transpose":
        return tuple(tuple(grid[r][c] for r in range(len(grid))) for c in range(len(grid[0])))
    if name == "rotate_180":
        return tuple(tuple(reversed(row)) for row in reversed(grid))
    raise SceneRuntimeError(f"unknown transform: {name}")


def components(grid: Grid, colour: int) -> list[set[Point]]:
    height, width = shape(grid)
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == colour}
    found: list[set[Point]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        comp = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    queue.append(q)
        found.append(comp)
    return sorted(found, key=lambda comp: (len(comp), min(comp)))


def recolour(grid: Grid, mapping: dict[int, int]) -> Grid:
    return tuple(tuple(mapping.get(cell, cell) for cell in row) for row in grid)


def move_singleton_towards(grid: Grid, source_colour: int, target_colour: int) -> Grid:
    source = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == source_colour]
    target = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == target_colour]
    if len(source) != 1 or len(target) != 1:
        raise SceneRuntimeError("relative singleton motion requires one source and one target")
    (sr, sc), (tr, tc) = source[0], target[0]
    nr = sr + (1 if tr > sr else -1 if tr < sr else 0)
    nc = sc + (1 if tc > sc else -1 if tc < sc else 0)
    values = [list(row) for row in grid]
    values[sr][sc] = background(grid)
    values[nr][nc] = source_colour
    return tuple(tuple(row) for row in values)


def edge_project(grid: Grid, radius: int = 1, fill_colour: int | None = None) -> Grid:
    if radius != 1:
        raise SceneRuntimeError("v14 currently supports one-cell edge projection")
    height, width = shape(grid)
    bg = background(grid) if fill_colour is None else int(fill_colour)
    values = [[bg for _ in range(width + 2)] for _ in range(height + 2)]
    for r in range(height):
        for c in range(width):
            values[r + 1][c + 1] = grid[r][c]
    for c in range(width):
        values[0][c + 1] = grid[0][c]
        values[height + 1][c + 1] = grid[height - 1][c]
    for r in range(height):
        values[r + 1][0] = grid[r][0]
        values[r + 1][width + 1] = grid[r][width - 1]
    return tuple(tuple(row) for row in values)


def _full_colour_rows(grid: Grid, colour: int) -> list[int]:
    return [r for r, row in enumerate(grid) if all(value == colour for value in row)]


def _full_colour_cols(grid: Grid, colour: int) -> list[int]:
    height, width = shape(grid)
    return [c for c in range(width) if all(grid[r][c] == colour for r in range(height))]


def _runs(length: int, separators: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(length + 1):
        if index == length or index in separators:
            if start < index:
                runs.append((start, index))
            start = index + 1
    return runs


def decode_regular_linegrid(grid: Grid, line_colour: int | None, final_transform: str) -> Grid:
    height, width = shape(grid)
    if line_colour is None:
        candidates = []
        for colour in sorted({value for row in grid for value in row}):
            rows0 = _full_colour_rows(grid, colour)
            cols0 = _full_colour_cols(grid, colour)
            if rows0 and cols0:
                candidates.append((-(len(rows0) + len(cols0)), colour))
        if not candidates:
            raise SceneRuntimeError("no separator colour inferred")
        line_colour = min(candidates)[1]
    rows = _runs(height, set(_full_colour_rows(grid, line_colour)))
    cols = _runs(width, set(_full_colour_cols(grid, line_colour)))
    if len(rows) < 2 or len(cols) < 2:
        raise SceneRuntimeError("not a regular line grid")
    row_sizes = {end - start for start, end in rows}
    col_sizes = {end - start for start, end in cols}
    if len(row_sizes) != 1 or len(col_sizes) != 1:
        raise SceneRuntimeError("line-grid cells are not regular")
    decoded = []
    for r0, r1 in rows:
        row = []
        for c0, c1 in cols:
            values = [grid[r][c] for r in range(r0, r1) for c in range(c0, c1)]
            counts = Counter(value for value in values if value != line_colour)
            if not counts:
                row.append(line_colour)
            else:
                row.append(min(counts, key=lambda value: (-counts[value], value)))
        decoded.append(tuple(row))
    return transform(tuple(decoded), final_transform)


def overlay_equal_tiles(
    grid: Grid,
    tile_rows: int,
    tile_cols: int,
    order: tuple[int, ...],
) -> Grid:
    height, width = shape(grid)
    if height % tile_rows or width % tile_cols:
        raise SceneRuntimeError("grid does not divide into equal tiles")
    tile_height, tile_width = height // tile_rows, width // tile_cols
    count = tile_rows * tile_cols
    if sorted(order) != list(range(count)):
        raise SceneRuntimeError("tile order is not a permutation")
    bg = background(grid)
    output = [[bg for _ in range(tile_width)] for _ in range(tile_height)]
    for index in order:
        tile_row, tile_col = divmod(index, tile_cols)
        for r in range(tile_height):
            for c in range(tile_width):
                value = grid[tile_row * tile_height + r][tile_col * tile_width + c]
                if value != bg:
                    output[r][c] = value
    return tuple(tuple(row) for row in output)


def execute_stage(stage: dict[str, Any], grid: Grid) -> Grid:
    op = stage["op"]
    if op == "recolour":
        mapping = {int(k): int(v) for k, v in stage["mapping"].items()}
        return recolour(grid, mapping)
    if op == "move_singleton_towards":
        return move_singleton_towards(
            grid,
            int(stage["source_colour"]),
            int(stage["target_colour"]),
        )
    if op == "edge_project":
        return edge_project(grid, int(stage.get("radius", 1)), stage.get("fill_colour"))
    if op == "decode_regular_linegrid":
        return decode_regular_linegrid(
            grid,
            int(stage["line_colour"]),
            str(stage["transform"]),
        )
    if op == "overlay_equal_tiles":
        return overlay_equal_tiles(
            grid,
            int(stage["tile_rows"]),
            int(stage["tile_cols"]),
            tuple(int(value) for value in stage["order"]),
        )
    if op == "transform":
        return transform(grid, str(stage["name"]))
    raise SceneRuntimeError(f"unknown scene stage: {op}")


def execute_pipeline(pipeline: Iterable[dict[str, Any]], grid: Grid) -> Grid:
    current = grid
    for stage in pipeline:
        current = execute_stage(stage, current)
    return current



# Generic v14 canvas-colour-aware edge projection.
def edge_project(grid: Grid, radius: int = 1, fill_colour: int | None = None) -> Grid:
    if radius != 1:
        raise SceneRuntimeError("v14 currently supports one-cell edge projection")
    height, width = shape(grid)
    fill = background(grid) if fill_colour is None else int(fill_colour)
    values = [[fill for _ in range(width + 2)] for _ in range(height + 2)]
    for r in range(height):
        for c in range(width):
            values[r + 1][c + 1] = grid[r][c]
    for c in range(width):
        values[0][c + 1] = grid[0][c]
        values[height + 1][c + 1] = grid[height - 1][c]
    for r in range(height):
        values[r + 1][0] = grid[r][0]
        values[r + 1][width + 1] = grid[r][width - 1]
    return tuple(tuple(row) for row in values)


def detect_regular_line_colour(grid: Grid) -> int:
    height, width = shape(grid)
    candidates: list[int] = []
    for colour in sorted({cell for row in grid for cell in row}):
        row_separators = set(_full_colour_rows(grid, colour))
        col_separators = set(_full_colour_cols(grid, colour))
        if not row_separators or not col_separators:
            continue
        rows = _runs(height, row_separators)
        cols = _runs(width, col_separators)
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len({end - start for start, end in rows}) != 1:
            continue
        if len({end - start for start, end in cols}) != 1:
            continue
        candidates.append(colour)
    if len(candidates) != 1:
        raise SceneRuntimeError("regular line colour is not uniquely identifiable")
    return candidates[0]


_decode_regular_linegrid_numeric = decode_regular_linegrid


def decode_regular_linegrid(grid: Grid, line_colour: int | str, final_transform: str) -> Grid:
    selected = detect_regular_line_colour(grid) if line_colour == "structural" else int(line_colour)
    return _decode_regular_linegrid_numeric(grid, selected, final_transform)


_execute_stage_before_structural_roles = execute_stage


def execute_stage(stage: dict[str, Any], grid: Grid) -> Grid:
    if stage.get("op") == "decode_regular_linegrid":
        return decode_regular_linegrid(
            grid,
            stage["line_colour"],
            str(stage["transform"]),
        )
    return _execute_stage_before_structural_roles(stage, grid)
