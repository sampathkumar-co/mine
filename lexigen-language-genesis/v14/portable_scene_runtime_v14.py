from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable

PortableGrid = tuple[tuple[int, ...], ...]
Coord = tuple[int, int]


class PortableSceneError(RuntimeError):
    pass


def gridify(value: Any) -> PortableGrid:
    result = tuple(tuple(int(cell) for cell in row) for row in value)
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise PortableSceneError("invalid grid")
    return result


def dims(grid: PortableGrid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def modal_colour(grid: PortableGrid) -> int:
    tally = Counter(cell for row in grid for cell in row)
    return sorted(tally, key=lambda colour: (-tally[colour], colour))[0]


def geometric_transform(grid: PortableGrid, name: str) -> PortableGrid:
    if name == "identity":
        return grid
    if name == "flip_h":
        return tuple(tuple(row[::-1]) for row in grid)
    if name == "flip_v":
        return tuple(grid[::-1])
    if name == "transpose":
        h, w = dims(grid)
        return tuple(tuple(grid[r][c] for r in range(h)) for c in range(w))
    if name == "rotate_180":
        return tuple(tuple(row[::-1]) for row in grid[::-1])
    raise PortableSceneError("unsupported transform")


def paint_recolour(grid: PortableGrid, mapping: dict[int, int]) -> PortableGrid:
    return tuple(tuple(mapping.get(value, value) for value in row) for row in grid)


def singleton_step(grid: PortableGrid, source_colour: int, target_colour: int) -> PortableGrid:
    source = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == source_colour]
    target = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == target_colour]
    if len(source) != 1 or len(target) != 1:
        raise PortableSceneError("singleton role mismatch")
    sr, sc = source[0]
    tr, tc = target[0]
    dr = (tr > sr) - (tr < sr)
    dc = (tc > sc) - (tc < sc)
    mutable = [list(row) for row in grid]
    mutable[sr][sc] = modal_colour(grid)
    mutable[sr + dr][sc + dc] = source_colour
    return tuple(tuple(row) for row in mutable)


def project_edges(grid: PortableGrid, fill_colour: int) -> PortableGrid:
    height, width = dims(grid)
    out = [[int(fill_colour) for _ in range(width + 2)] for _ in range(height + 2)]
    for r in range(height):
        for c in range(width):
            out[r + 1][c + 1] = grid[r][c]
    for c in range(width):
        out[0][c + 1] = grid[0][c]
        out[-1][c + 1] = grid[-1][c]
    for r in range(height):
        out[r + 1][0] = grid[r][0]
        out[r + 1][-1] = grid[r][-1]
    return tuple(tuple(row) for row in out)


def solid_rows(grid: PortableGrid, colour: int) -> list[int]:
    return [r for r, row in enumerate(grid) if all(value == colour for value in row)]


def solid_cols(grid: PortableGrid, colour: int) -> list[int]:
    h, w = dims(grid)
    return [c for c in range(w) if all(grid[r][c] == colour for r in range(h))]


def intervals(length: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    blocked = set(separators)
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(length + 1):
        if index == length or index in blocked:
            if start < index:
                result.append((start, index))
            start = index + 1
    return result


def structural_separator(grid: PortableGrid) -> int:
    h, w = dims(grid)
    matches = []
    for colour in sorted({value for row in grid for value in row}):
        rows = intervals(h, solid_rows(grid, colour))
        cols = intervals(w, solid_cols(grid, colour))
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len({b - a for a, b in rows}) == 1 and len({b - a for a, b in cols}) == 1:
            if solid_rows(grid, colour) and solid_cols(grid, colour):
                matches.append(colour)
    if len(matches) != 1:
        raise PortableSceneError("separator role is ambiguous")
    return matches[0]


def decode_line_cells(grid: PortableGrid, line_colour: int | str, transform_name: str) -> PortableGrid:
    selected = structural_separator(grid) if line_colour == "structural" else int(line_colour)
    h, w = dims(grid)
    row_intervals = intervals(h, solid_rows(grid, selected))
    col_intervals = intervals(w, solid_cols(grid, selected))
    if len(row_intervals) < 2 or len(col_intervals) < 2:
        raise PortableSceneError("line-grid partition missing")
    decoded = []
    for r0, r1 in row_intervals:
        row = []
        for c0, c1 in col_intervals:
            tally = Counter(
                grid[r][c]
                for r in range(r0, r1)
                for c in range(c0, c1)
                if grid[r][c] != selected
            )
            if not tally:
                row.append(selected)
            else:
                row.append(sorted(tally, key=lambda colour: (-tally[colour], colour))[0])
        decoded.append(tuple(row))
    return geometric_transform(tuple(decoded), transform_name)


def reduce_tiles(grid: PortableGrid, tile_rows: int, tile_cols: int, order: tuple[int, ...]) -> PortableGrid:
    h, w = dims(grid)
    if h % tile_rows or w % tile_cols:
        raise PortableSceneError("unequal tile partition")
    th, tw = h // tile_rows, w // tile_cols
    if sorted(order) != list(range(tile_rows * tile_cols)):
        raise PortableSceneError("invalid tile order")
    bg = modal_colour(grid)
    out = [[bg for _ in range(tw)] for _ in range(th)]
    for index in order:
        tr, tc = divmod(index, tile_cols)
        for r in range(th):
            for c in range(tw):
                value = grid[tr * th + r][tc * tw + c]
                if value != bg:
                    out[r][c] = value
    return tuple(tuple(row) for row in out)


def flood_components(grid: PortableGrid, colour: int) -> list[set[Coord]]:
    h, w = dims(grid)
    unseen = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == colour}
    result = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        group = {root}
        queue = deque([root])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    group.add(nxt)
                    queue.append(nxt)
        result.append(group)
    return result


def bounds(points: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(points)
    return min(r for r, _ in pts), min(c for _, c in pts), max(r for r, _ in pts), max(c for _, c in pts)


def boundary_points(box: tuple[int, int, int, int]) -> set[Coord]:
    r0, c0, r1, c1 = box
    result = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
    result |= {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    return result


def area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _layer_objects(grid: PortableGrid, mode: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    colours = sorted({cell for row in grid for cell in row if cell != 0})
    if mode == "components":
        for colour in colours:
            for comp in flood_components(grid, colour):
                box = bounds(comp)
                if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                    objects.append({"colour": colour, "box": box})
    elif mode == "colours":
        for colour in colours:
            pts = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour}
            box = bounds(pts)
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                objects.append({"colour": colour, "box": box})
    else:
        raise PortableSceneError("unknown layer mode")
    return objects


def _contains_box(a, b) -> bool:
    return a != b and a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _topological_layer_order(grid: PortableGrid, objects: list[dict[str, Any]], mode: str) -> list[int]:
    edges: set[tuple[int, int]] = set()
    if mode == "components":
        for i, left in enumerate(objects):
            for j, right in enumerate(objects):
                if i != j and _contains_box(left["box"], right["box"]):
                    edges.add((i, j))
    else:
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                common = boundary_points(objects[i]["box"]) & boundary_points(objects[j]["box"])
                seen = {grid[r][c] for r, c in common if grid[r][c] in {objects[i]["colour"], objects[j]["colour"]}}
                if seen == {objects[i]["colour"]}:
                    edges.add((j, i))
                elif seen == {objects[j]["colour"]}:
                    edges.add((i, j))

    outgoing = {index: set() for index in range(len(objects))}
    indegree = [0 for _ in objects]
    for before, after in edges:
        if after not in outgoing[before]:
            outgoing[before].add(after)
            indegree[after] += 1
    ready = [index for index, value in enumerate(indegree) if value == 0]
    order: list[int] = []
    while ready:
        ready.sort(key=lambda index: (-area(objects[index]["box"]), objects[index]["colour"], objects[index]["box"]))
        index = ready.pop(0)
        order.append(index)
        for after in sorted(outgoing[index]):
            indegree[after] -= 1
            if indegree[after] == 0:
                ready.append(after)
    if len(order) != len(objects):
        raise PortableSceneError("layer order cycle")
    return order


def render_layers(grid: PortableGrid, mode: str) -> PortableGrid:
    objects = _layer_objects(grid, mode)
    if len(objects) < 2:
        raise PortableSceneError("insufficient layers")
    order = _topological_layer_order(grid, objects, mode)
    size = 2 * len(order) - 1
    out = [[0 for _ in range(size)] for _ in range(size)]
    for depth, index in enumerate(order):
        colour = objects[index]["colour"]
        for r in range(depth, size - depth):
            for c in range(depth, size - depth):
                out[r][c] = colour
    return tuple(tuple(row) for row in out)


def paint_internal_axis(grid: PortableGrid, fill_colour: int) -> PortableGrid:
    bg = 0 if any(cell == 0 for row in grid for cell in row) else modal_colour(grid)
    foreground = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != bg]
    if not foreground:
        raise PortableSceneError("no foreground")
    r0, c0, r1, c1 = bounds(foreground)
    rows = [r for r in range(r0, r1 + 1) if all(value == bg for value in grid[r])]
    cols = [c for c in range(c0, c1 + 1) if all(grid[r][c] == bg for r in range(len(grid)))]
    if len(rows) + len(cols) != 1:
        raise PortableSceneError("axis role is ambiguous")
    out = [list(row) for row in grid]
    if rows:
        for c in range(len(grid[0])):
            out[rows[0]][c] = int(fill_colour)
    else:
        for r in range(len(grid)):
            out[r][cols[0]] = int(fill_colour)
    return tuple(tuple(row) for row in out)


def ray_motifs(grid: PortableGrid) -> PortableGrid:
    h, w = dims(grid)
    motifs: dict[Coord, tuple[int, int, int]] = {}
    for r in range(h - 1):
        for c in range(w - 1):
            entries = [(r, c, grid[r][c]), (r, c + 1, grid[r][c + 1]),
                       (r + 1, c, grid[r + 1][c]), (r + 1, c + 1, grid[r + 1][c + 1])]
            zeros = [(rr, cc) for rr, cc, value in entries if value == 0]
            coloured = [value for _, _, value in entries if value != 0]
            if len(zeros) == 1 and len(coloured) == 3 and len(set(coloured)) == 1:
                rr, cc = zeros[0]
                motifs[(rr, cc)] = (coloured[0], -1 if rr == r else 1, -1 if cc == c else 1)
    if not motifs:
        raise PortableSceneError("no ray motifs")
    out = [list(row) for row in grid]
    for (r, c), (colour, dr, dc) in sorted(motifs.items()):
        r += dr; c += dc
        while 0 <= r < h and 0 <= c < w:
            out[r][c] = colour
            r += dr; c += dc
    return tuple(tuple(row) for row in out)


def execute_portable_stage(stage: dict[str, Any], grid: PortableGrid) -> PortableGrid:
    op = stage["op"]
    if op == "recolour":
        return paint_recolour(grid, {int(k): int(v) for k, v in stage["mapping"].items()})
    if op == "move_singleton_towards":
        return singleton_step(grid, int(stage["source_colour"]), int(stage["target_colour"]))
    if op == "edge_project":
        return project_edges(grid, int(stage["fill_colour"]))
    if op == "decode_regular_linegrid":
        return decode_line_cells(grid, stage["line_colour"], str(stage["transform"]))
    if op == "overlay_equal_tiles":
        return reduce_tiles(grid, int(stage["tile_rows"]), int(stage["tile_cols"]), tuple(stage["order"]))
    if op == "canonical_rectangular_layers":
        return render_layers(grid, str(stage["object_mode"]))
    if op == "fill_internal_blank_axis":
        return paint_internal_axis(grid, int(stage["fill_colour"]))
    if op == "extend_corner_marked_rays":
        return ray_motifs(grid)
    if op == "transform":
        return geometric_transform(grid, str(stage["name"]))
    raise PortableSceneError(f"unknown portable stage: {op}")


def execute_portable_pipeline(pipeline: Iterable[dict[str, Any]], grid: Any) -> PortableGrid:
    current = gridify(grid)
    for stage in pipeline:
        current = execute_portable_stage(stage, current)
    return current
