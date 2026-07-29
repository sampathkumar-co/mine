from __future__ import annotations

from typing import Any, Sequence

PortableGrid = tuple[tuple[int, ...], ...]


class PortablePipelineError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortablePipelineError("invalid grid")
    return grid


def execute_portable(program: dict[str, Any], grid: PortableGrid) -> PortableGrid:
    if program.get("schema") != "lexigen-compositional-pipeline-v1":
        raise PortablePipelineError("invalid pipeline schema")
    extraction, allocation, relation_stage, precedence, render = program["stages"]
    background = int(extraction["background"])
    if extraction["mode"] == "compress_blank_axes":
        rows = [index for index, row in enumerate(grid) if any(value != background for value in row)]
        cols = [
            col for col in range(len(grid[0]))
            if any(grid[row][col] != background for row in range(len(grid)))
        ]
        lattice = tuple(tuple(grid[row][col] for col in cols) for row in rows)
    elif extraction["mode"] == "sample_regular_stride":
        lattice = tuple(
            tuple(
                grid[row][col]
                for col in range(
                    int(extraction["col_offset"]),
                    len(grid[0]),
                    int(extraction["col_stride"]),
                )
            )
            for row in range(
                int(extraction["row_offset"]),
                len(grid),
                int(extraction["row_stride"]),
            )
        )
    else:
        raise PortablePipelineError("invalid extraction mode")
    if not lattice or not lattice[0]:
        raise PortablePipelineError("empty lattice")
    lattice_height, lattice_width = len(lattice), len(lattice[0])
    out_h = int(allocation["output_height"])
    out_w = int(allocation["output_width"])
    margin = int(allocation["margin"])
    gap = int(allocation["gap"])
    drawable_h = out_h - 2 * margin - gap * (lattice_height - 1)
    drawable_w = out_w - 2 * margin - gap * (lattice_width - 1)
    if drawable_h <= 0 or drawable_w <= 0 or drawable_h % lattice_height or drawable_w % lattice_width:
        raise PortablePipelineError("invalid layout division")
    tile_h, tile_w = drawable_h // lattice_height, drawable_w // lattice_width

    def bounds(row: int, col: int):
        r0 = margin + row * (tile_h + gap)
        c0 = margin + col * (tile_w + gap)
        return r0, r0 + tile_h - 1, c0, c0 + tile_w - 1

    def related(first: int, second: int) -> bool:
        predicate = relation_stage["predicate"]
        if predicate == "equal_nonbackground":
            return first == second and first != background
        if predicate == "equal":
            return first == second
        if predicate == "both_nonbackground":
            return first != background and second != background
        raise PortablePipelineError("invalid relation predicate")

    horizontal = []
    vertical = []
    for row in range(lattice_height):
        for col in range(lattice_width - 1):
            if related(lattice[row][col], lattice[row][col + 1]):
                horizontal.append(((row, col), (row, col + 1)))
    for row in range(lattice_height - 1):
        for col in range(lattice_width):
            if related(lattice[row][col], lattice[row + 1][col]):
                vertical.append(((row, col), (row + 1, col)))
    mode = precedence["mode"]
    if mode == "all_edges":
        edges = [("horizontal", *edge) for edge in horizontal] + [("vertical", *edge) for edge in vertical]
    elif mode == "horizontal_then_vertical_unclaimed":
        claimed = {point for edge in horizontal for point in edge}
        edges = [("horizontal", *edge) for edge in horizontal]
        edges += [("vertical", *edge) for edge in vertical if edge[0] not in claimed and edge[1] not in claimed]
    elif mode == "vertical_then_horizontal_unclaimed":
        claimed = {point for edge in vertical for point in edge}
        edges = [("vertical", *edge) for edge in vertical]
        edges += [("horizontal", *edge) for edge in horizontal if edge[0] not in claimed and edge[1] not in claimed]
    else:
        raise PortablePipelineError("invalid precedence mode")

    canvas_background = int(render["canvas_background"])
    canvas = [[canvas_background for _ in range(out_w)] for _ in range(out_h)]
    for row in range(lattice_height):
        for col in range(lattice_width):
            colour = lattice[row][col]
            if render["skip_background_tiles"] and colour == background:
                continue
            r0, r1, c0, c1 = bounds(row, col)
            for target_row in range(r0, r1 + 1):
                for target_col in range(c0, c1 + 1):
                    canvas[target_row][target_col] = colour
    for orientation, first, second in edges:
        colour = lattice[first[0]][first[1]]
        first_box, second_box = bounds(*first), bounds(*second)
        if orientation == "horizontal":
            r0, r1, c0, c1 = first_box[0], first_box[1], first_box[3] + 1, second_box[2] - 1
        else:
            r0, r1, c0, c1 = first_box[1] + 1, second_box[0] - 1, first_box[2], first_box[3]
        for target_row in range(r0, r1 + 1):
            for target_col in range(c0, c1 + 1):
                canvas[target_row][target_col] = colour
    return tuple(tuple(row) for row in canvas)
