from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]


class CompositionalRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Lattice:
    values: tuple[tuple[int, ...], ...]
    background: int

    @property
    def height(self) -> int:
        return len(self.values)

    @property
    def width(self) -> int:
        return len(self.values[0])


@dataclass(frozen=True)
class Layout:
    output_height: int
    output_width: int
    margin: int
    gap: int
    tile_height: int
    tile_width: int


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise CompositionalRuntimeError("grid must be a non-empty rectangle")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _nonblank_indices(lines: Sequence[Sequence[int]], background: int) -> list[int]:
    return [index for index, line in enumerate(lines) if any(value != background for value in line)]


def extract_lattice(grid: Grid, stage: dict[str, Any]) -> Lattice:
    background = int(stage["background"])
    mode = str(stage["mode"])
    if mode == "compress_blank_axes":
        row_indices = _nonblank_indices(grid, background)
        columns = tuple(zip(*grid))
        col_indices = _nonblank_indices(columns, background)
        if not row_indices or not col_indices:
            raise CompositionalRuntimeError("lattice has no nonblank rows or columns")
        values = tuple(
            tuple(grid[row][col] for col in col_indices)
            for row in row_indices
        )
    elif mode == "sample_regular_stride":
        row_offset = int(stage["row_offset"])
        col_offset = int(stage["col_offset"])
        row_stride = int(stage["row_stride"])
        col_stride = int(stage["col_stride"])
        values = tuple(
            tuple(grid[row][col] for col in range(col_offset, len(grid[0]), col_stride))
            for row in range(row_offset, len(grid), row_stride)
        )
    else:
        raise CompositionalRuntimeError(f"unsupported lattice extraction mode: {mode}")
    if not values or not values[0] or any(len(row) != len(values[0]) for row in values):
        raise CompositionalRuntimeError("extracted lattice is not rectangular")
    if any(all(value == background for value in row) for row in values):
        raise CompositionalRuntimeError("extracted lattice contains a blank row")
    if any(all(values[row][col] == background for row in range(len(values))) for col in range(len(values[0]))):
        raise CompositionalRuntimeError("extracted lattice contains a blank column")
    return Lattice(values, background)


def allocate_layout(lattice: Lattice, stage: dict[str, Any]) -> Layout:
    output_height = int(stage["output_height"])
    output_width = int(stage["output_width"])
    margin = int(stage["margin"])
    gap = int(stage["gap"])
    available_height = output_height - 2 * margin - gap * (lattice.height - 1)
    available_width = output_width - 2 * margin - gap * (lattice.width - 1)
    if available_height <= 0 or available_width <= 0:
        raise CompositionalRuntimeError("layout has no drawable area")
    if available_height % lattice.height or available_width % lattice.width:
        raise CompositionalRuntimeError("layout does not divide into uniform tiles")
    tile_height = available_height // lattice.height
    tile_width = available_width // lattice.width
    if min(tile_height, tile_width) <= 0:
        raise CompositionalRuntimeError("layout tile is empty")
    return Layout(output_height, output_width, margin, gap, tile_height, tile_width)


def _tile_bounds(layout: Layout, row: int, col: int) -> tuple[int, int, int, int]:
    r0 = layout.margin + row * (layout.tile_height + layout.gap)
    c0 = layout.margin + col * (layout.tile_width + layout.gap)
    return r0, r0 + layout.tile_height - 1, c0, c0 + layout.tile_width - 1


def build_relations(lattice: Lattice, stage: dict[str, Any]) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    predicate = str(stage["predicate"])
    horizontal = []
    vertical = []

    def related(left: int, right: int) -> bool:
        if predicate == "equal_nonbackground":
            return left == right and left != lattice.background
        if predicate == "equal":
            return left == right
        if predicate == "both_nonbackground":
            return left != lattice.background and right != lattice.background
        raise CompositionalRuntimeError(f"unsupported relation predicate: {predicate}")

    for row in range(lattice.height):
        for col in range(lattice.width - 1):
            if related(lattice.values[row][col], lattice.values[row][col + 1]):
                horizontal.append(((row, col), (row, col + 1)))
    for row in range(lattice.height - 1):
        for col in range(lattice.width):
            if related(lattice.values[row][col], lattice.values[row + 1][col]):
                vertical.append(((row, col), (row + 1, col)))
    return {"horizontal": horizontal, "vertical": vertical}


def apply_precedence(
    relations: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]],
    stage: dict[str, Any],
) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    mode = str(stage["mode"])
    horizontal = relations["horizontal"]
    vertical = relations["vertical"]
    if mode == "all_edges":
        return [("horizontal", *edge) for edge in horizontal] + [("vertical", *edge) for edge in vertical]
    if mode == "horizontal_then_vertical_unclaimed":
        claimed = {point for edge in horizontal for point in edge}
        kept_vertical = [edge for edge in vertical if edge[0] not in claimed and edge[1] not in claimed]
        return [("horizontal", *edge) for edge in horizontal] + [("vertical", *edge) for edge in kept_vertical]
    if mode == "vertical_then_horizontal_unclaimed":
        claimed = {point for edge in vertical for point in edge}
        kept_horizontal = [edge for edge in horizontal if edge[0] not in claimed and edge[1] not in claimed]
        return [("vertical", *edge) for edge in vertical] + [("horizontal", *edge) for edge in kept_horizontal]
    raise CompositionalRuntimeError(f"unsupported precedence mode: {mode}")


def render_pipeline(
    lattice: Lattice,
    layout: Layout,
    edges: list[tuple[str, tuple[int, int], tuple[int, int]]],
    stage: dict[str, Any],
) -> Grid:
    canvas_background = int(stage["canvas_background"])
    skip_background_tiles = bool(stage["skip_background_tiles"])
    canvas = [
        [canvas_background for _ in range(layout.output_width)]
        for _ in range(layout.output_height)
    ]
    for row in range(lattice.height):
        for col in range(lattice.width):
            colour = lattice.values[row][col]
            if skip_background_tiles and colour == lattice.background:
                continue
            r0, r1, c0, c1 = _tile_bounds(layout, row, col)
            for out_row in range(r0, r1 + 1):
                for out_col in range(c0, c1 + 1):
                    canvas[out_row][out_col] = colour

    for orientation, first, second in edges:
        colour = lattice.values[first[0]][first[1]]
        first_bounds = _tile_bounds(layout, *first)
        second_bounds = _tile_bounds(layout, *second)
        if orientation == "horizontal":
            r0, r1 = first_bounds[0], first_bounds[1]
            c0, c1 = first_bounds[3] + 1, second_bounds[2] - 1
        else:
            r0, r1 = first_bounds[1] + 1, second_bounds[0] - 1
            c0, c1 = first_bounds[2], first_bounds[3]
        for out_row in range(r0, r1 + 1):
            for out_col in range(c0, c1 + 1):
                canvas[out_row][out_col] = colour
    return tuple(tuple(row) for row in canvas)


def execute_pipeline(program: dict[str, Any], grid: Grid) -> Grid:
    if program.get("schema") != "lexigen-compositional-pipeline-v1":
        raise CompositionalRuntimeError("unsupported compositional pipeline schema")
    stages = program.get("stages")
    if not isinstance(stages, list) or [stage.get("kind") for stage in stages] != [
        "extract_lattice",
        "allocate_layout",
        "build_relations",
        "apply_precedence",
        "render",
    ]:
        raise CompositionalRuntimeError("v11 pipeline stages are malformed")
    lattice = extract_lattice(grid, stages[0])
    layout = allocate_layout(lattice, stages[1])
    relations = build_relations(lattice, stages[2])
    edges = apply_precedence(relations, stages[3])
    return render_pipeline(lattice, layout, edges, stages[4])
