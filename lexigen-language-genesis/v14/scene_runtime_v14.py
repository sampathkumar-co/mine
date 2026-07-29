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
        height, width = shape(grid)
        return tuple(tuple(grid[row][col] for row in range(height)) for col in range(width))
    if name == "rotate_180":
        return tuple(tuple(reversed(row)) for row in reversed(grid))
    raise SceneRuntimeError(f"unknown transform: {name}")


def components(grid: Grid, colour: int) -> list[set[Point]]:
    height, width = shape(grid)
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == colour}
    result: list[set[Point]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (r + dr, c + dc)
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        result.append(component)
    return sorted(result, key=lambda item: (len(item), min(item)))


def recolour(grid: Grid, mapping: dict[int, int]) -> Grid:
    return tuple(tuple(mapping.get(value, value) for value in row) for row in grid)


def move_singleton_towards(grid: Grid, source_colour: int, target_colour: int) -> Grid:
    source = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == source_colour]
    target = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == target_colour]
    if len(source) != 1 or len(target) != 1:
        raise SceneRuntimeError("relative singleton motion requires one source and one target")
    (sr, sc), (tr, tc) = source[0], target[0]
    dr = (tr > sr) - (tr < sr)
    dc = (tc > sc) - (tc < sc)
    values = [list(row) for row in grid]
    values[sr][sc] = background(grid)
    values[sr + dr][sc + dc] = source_colour
    return tuple(tuple(row) for row in values)


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


def _full_colour_rows(grid: Grid, colour: int) -> list[int]:
    return [r for r, row in enumerate(grid) if all(value == colour for value in row)]


def _full_colour_cols(grid: Grid, colour: int) -> list[int]:
    height, width = shape(grid)
    return [c for c in range(width) if all(grid[r][c] == colour for r in range(height))]


def _runs(length: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    blocked = set(separators)
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(length + 1):
        if index == length or index in blocked:
            if start < index:
                runs.append((start, index))
            start = index + 1
    return runs


def detect_regular_line_colour(grid: Grid) -> int:
    height, width = shape(grid)
    matches: list[int] = []
    for colour in sorted({cell for row in grid for cell in row}):
        row_separators = _full_colour_rows(grid, colour)
        col_separators = _full_colour_cols(grid, colour)
        if not row_separators or not col_separators:
            continue
        rows = _runs(height, row_separators)
        cols = _runs(width, col_separators)
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len({end - start for start, end in rows}) == 1 and len({end - start for start, end in cols}) == 1:
            matches.append(colour)
    if len(matches) != 1:
        raise SceneRuntimeError("regular line colour is not uniquely identifiable")
    return matches[0]


def decode_regular_linegrid(grid: Grid, line_colour: int | str, final_transform: str) -> Grid:
    selected = detect_regular_line_colour(grid) if line_colour == "structural" else int(line_colour)
    height, width = shape(grid)
    rows = _runs(height, _full_colour_rows(grid, selected))
    cols = _runs(width, _full_colour_cols(grid, selected))
    if len(rows) < 2 or len(cols) < 2:
        raise SceneRuntimeError("not a regular line grid")
    decoded = []
    for r0, r1 in rows:
        row = []
        for c0, c1 in cols:
            counts = Counter(
                grid[r][c]
                for r in range(r0, r1)
                for c in range(c0, c1)
                if grid[r][c] != selected
            )
            if not counts:
                row.append(selected)
            else:
                row.append(min(counts, key=lambda value: (-counts[value], value)))
        decoded.append(tuple(row))
    return transform(tuple(decoded), final_transform)


def overlay_equal_tiles(grid: Grid, tile_rows: int, tile_cols: int, order: tuple[int, ...]) -> Grid:
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


def _bbox(points: Iterable[Point]) -> tuple[int, int, int, int]:
    values = list(points)
    return (
        min(r for r, _ in values),
        min(c for _, c in values),
        max(r for r, _ in values),
        max(c for _, c in values),
    )


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer != inner
        and outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _boundary(box: tuple[int, int, int, int]) -> set[Point]:
    r0, c0, r1, c1 = box
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _layer_objects(grid: Grid, object_mode: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    colours = sorted({cell for row in grid for cell in row if cell != 0})
    if object_mode == "components":
        for colour in colours:
            for component in components(grid, colour):
                box = _bbox(component)
                if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                    objects.append({"colour": colour, "box": box})
    elif object_mode == "colours":
        for colour in colours:
            points = {
                (r, c)
                for r, row in enumerate(grid)
                for c, value in enumerate(row)
                if value == colour
            }
            box = _bbox(points)
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                objects.append({"colour": colour, "box": box})
    else:
        raise SceneRuntimeError("unknown rectangular-layer object mode")
    if len(objects) < 2:
        raise SceneRuntimeError("fewer than two rectangular layers")
    return objects


def _layer_edges(grid: Grid, objects: list[dict[str, Any]], object_mode: str) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if object_mode == "components":
        for i, left in enumerate(objects):
            for j, right in enumerate(objects):
                if i != j and _contains(left["box"], right["box"]):
                    edges.add((i, j))
        return edges

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            shared = _boundary(objects[i]["box"]) & _boundary(objects[j]["box"])
            observed = {
                grid[r][c]
                for r, c in shared
                if grid[r][c] in {objects[i]["colour"], objects[j]["colour"]}
            }
            if observed == {objects[i]["colour"]}:
                edges.add((j, i))
            elif observed == {objects[j]["colour"]}:
                edges.add((i, j))
    return edges


def _topological_order(objects: list[dict[str, Any]], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {index: set() for index in range(len(objects))}
    indegree = [0] * len(objects)
    for before, after in edges:
        if after not in outgoing[before]:
            outgoing[before].add(after)
            indegree[after] += 1
    ready = [index for index, value in enumerate(indegree) if value == 0]
    order: list[int] = []
    while ready:
        ready.sort(
            key=lambda index: (
                -_box_area(objects[index]["box"]),
                objects[index]["colour"],
                objects[index]["box"],
            )
        )
        current = ready.pop(0)
        order.append(current)
        for neighbour in sorted(outgoing[current]):
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                ready.append(neighbour)
    if len(order) != len(objects):
        raise SceneRuntimeError("rectangular precedence graph contains a cycle")
    return order


def canonical_rectangular_layers(grid: Grid, object_mode: str) -> Grid:
    objects = _layer_objects(grid, object_mode)
    order = _topological_order(objects, _layer_edges(grid, objects, object_mode))
    size = 2 * len(order) - 1
    output = [[0 for _ in range(size)] for _ in range(size)]
    for depth, index in enumerate(order):
        colour = int(objects[index]["colour"])
        for r in range(depth, size - depth):
            for c in range(depth, size - depth):
                output[r][c] = colour
    return tuple(tuple(row) for row in output)


def fill_internal_blank_axis(grid: Grid, fill_colour: int) -> Grid:
    bg = 0 if any(cell == 0 for row in grid for cell in row) else background(grid)
    points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != bg]
    if not points:
        raise SceneRuntimeError("blank-axis inference requires a foreground object")
    r0, c0, r1, c1 = _bbox(points)
    blank_rows = [r for r in range(r0, r1 + 1) if all(value == bg for value in grid[r])]
    blank_cols = [c for c in range(c0, c1 + 1) if all(grid[r][c] == bg for r in range(len(grid)))]
    if len(blank_rows) + len(blank_cols) != 1:
        raise SceneRuntimeError("internal blank axis is not unique")
    values = [list(row) for row in grid]
    if blank_rows:
        for c in range(len(grid[0])):
            values[blank_rows[0]][c] = int(fill_colour)
    else:
        for r in range(len(grid)):
            values[r][blank_cols[0]] = int(fill_colour)
    return tuple(tuple(row) for row in values)


def extend_corner_marked_rays(grid: Grid) -> Grid:
    height, width = shape(grid)
    motifs: dict[Point, tuple[int, int, int]] = {}
    for r in range(height - 1):
        for c in range(width - 1):
            entries = (
                (r, c, grid[r][c]),
                (r, c + 1, grid[r][c + 1]),
                (r + 1, c, grid[r + 1][c]),
                (r + 1, c + 1, grid[r + 1][c + 1]),
            )
            zeros = [(rr, cc) for rr, cc, value in entries if value == 0]
            coloured = [value for _, _, value in entries if value != 0]
            if len(zeros) == 1 and len(coloured) == 3 and len(set(coloured)) == 1:
                rr, cc = zeros[0]
                motifs[(rr, cc)] = (
                    coloured[0],
                    -1 if rr == r else 1,
                    -1 if cc == c else 1,
                )
    if not motifs:
        raise SceneRuntimeError("no corner-marked ray motifs found")
    values = [list(row) for row in grid]
    for (r, c), (colour, dr, dc) in sorted(motifs.items()):
        r += dr
        c += dc
        while 0 <= r < height and 0 <= c < width:
            values[r][c] = colour
            r += dr
            c += dc
    return tuple(tuple(row) for row in values)


def execute_stage(stage: dict[str, Any], grid: Grid) -> Grid:
    op = stage["op"]
    if op == "recolour":
        return recolour(grid, {int(k): int(v) for k, v in stage["mapping"].items()})
    if op == "move_singleton_towards":
        return move_singleton_towards(grid, int(stage["source_colour"]), int(stage["target_colour"]))
    if op == "edge_project":
        return edge_project(grid, int(stage.get("radius", 1)), stage.get("fill_colour"))
    if op == "decode_regular_linegrid":
        return decode_regular_linegrid(grid, stage["line_colour"], str(stage["transform"]))
    if op == "overlay_equal_tiles":
        return overlay_equal_tiles(
            grid,
            int(stage["tile_rows"]),
            int(stage["tile_cols"]),
            tuple(int(value) for value in stage["order"]),
        )
    if op == "canonical_rectangular_layers":
        return canonical_rectangular_layers(grid, str(stage["object_mode"]))
    if op == "fill_internal_blank_axis":
        return fill_internal_blank_axis(grid, int(stage["fill_colour"]))
    if op == "extend_corner_marked_rays":
        return extend_corner_marked_rays(grid)
    if op == "transform":
        return transform(grid, str(stage["name"]))
    raise SceneRuntimeError(f"unknown scene stage: {op}")


def execute_pipeline(pipeline: Iterable[dict[str, Any]], grid: Grid) -> Grid:
    current = grid
    for stage in pipeline:
        current = execute_stage(stage, current)
        if len(current) > 60 or len(current[0]) > 60:
            raise SceneRuntimeError("intermediate grid exceeds v14 size budget")
    return current
