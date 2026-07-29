from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
AST = dict[str, Any]


class IRRuntimeError(RuntimeError):
    pass


def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise IRRuntimeError("invalid grid")
    return grid


def _mode(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def _transform(grid: Grid, name: str) -> Grid:
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
    raise IRRuntimeError(f"unknown transform: {name}")


def _components(grid: Grid, colour: int) -> list[set[Point]]:
    height, width = len(grid), len(grid[0])
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
                point = (r + dr, c + dc)
                if point in unseen:
                    unseen.remove(point)
                    component.add(point)
                    queue.append(point)
        result.append(component)
    return sorted(result, key=lambda points: (min(points), len(points)))


def _bbox(points: Iterable[Point]) -> tuple[int, int, int, int]:
    pts = list(points)
    if not pts:
        raise IRRuntimeError("empty point set")
    return (
        min(r for r, _ in pts),
        min(c for _, c in pts),
        max(r for r, _ in pts),
        max(c for _, c in pts),
    )


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _boundary(box: tuple[int, int, int, int]) -> set[Point]:
    r0, c0, r1, c1 = box
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _contains(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return left != right and left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


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


def _separator_colour(grid: Grid) -> int:
    height, width = len(grid), len(grid[0])
    matches: list[int] = []
    for colour in sorted({cell for row in grid for cell in row}):
        rows = [r for r, row in enumerate(grid) if all(value == colour for value in row)]
        cols = [c for c in range(width) if all(grid[r][c] == colour for r in range(height))]
        row_runs, col_runs = _runs(height, rows), _runs(width, cols)
        regular = (
            rows and cols and len(row_runs) >= 2 and len(col_runs) >= 2
            and len({b - a for a, b in row_runs}) == 1
            and len({b - a for a, b in col_runs}) == 1
        )
        if regular:
            matches.append(colour)
    if len(matches) != 1:
        raise IRRuntimeError("separator role is ambiguous")
    return matches[0]


def _decode_cells(grid: Grid, separator: int) -> Grid:
    height, width = len(grid), len(grid[0])
    rows = _runs(height, [r for r, row in enumerate(grid) if all(v == separator for v in row)])
    cols = _runs(width, [c for c in range(width) if all(grid[r][c] == separator for r in range(height))])
    decoded: list[tuple[int, ...]] = []
    for r0, r1 in rows:
        row: list[int] = []
        for c0, c1 in cols:
            counts = Counter(
                grid[r][c]
                for r in range(r0, r1)
                for c in range(c0, c1)
                if grid[r][c] != separator
            )
            row.append(separator if not counts else min(counts, key=lambda value: (-counts[value], value)))
        decoded.append(tuple(row))
    return tuple(decoded)


def _partition(grid: Grid, rows: int, cols: int) -> list[Grid]:
    height, width = len(grid), len(grid[0])
    if height % rows or width % cols:
        raise IRRuntimeError("grid does not divide into equal tiles")
    tile_h, tile_w = height // rows, width // cols
    return [
        tuple(tuple(grid[tr * tile_h + r][tc * tile_w + c] for c in range(tile_w)) for r in range(tile_h))
        for tr in range(rows)
        for tc in range(cols)
    ]


def _overlay(tiles: list[Grid], order: list[int], background: int) -> Grid:
    if sorted(order) != list(range(len(tiles))):
        raise IRRuntimeError("invalid overlay order")
    height, width = len(tiles[0]), len(tiles[0][0])
    output = [[background for _ in range(width)] for _ in range(height)]
    for index in order:
        tile = tiles[index]
        for r in range(height):
            for c in range(width):
                if tile[r][c] != background:
                    output[r][c] = tile[r][c]
    return tuple(tuple(row) for row in output)


def _rect_objects(grid: Grid, mode: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    colours = sorted({cell for row in grid for cell in row if cell != 0})
    if mode == "colours":
        for colour in colours:
            points = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour}
            box = _bbox(points)
            if box[2] - box[0] + 1 >= 3 and box[3] - box[1] + 1 >= 3:
                objects.append({"colour": colour, "box": box})
    elif mode == "components":
        for colour in colours:
            for component in _components(grid, colour):
                box = _bbox(component)
                if box[2] - box[0] + 1 >= 3 and box[3] - box[1] + 1 >= 3:
                    objects.append({"colour": colour, "box": box})
    else:
        raise IRRuntimeError("unknown rectangle extraction mode")
    if len(objects) < 2:
        raise IRRuntimeError("fewer than two rectangular objects")
    return objects


def _rect_order(grid: Grid, objects: list[dict[str, Any]], mode: str) -> list[int]:
    edges: set[tuple[int, int]] = set()
    if mode == "components":
        for i, left in enumerate(objects):
            for j, right in enumerate(objects):
                if i != j and _contains(left["box"], right["box"]):
                    edges.add((i, j))
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            shared = _boundary(objects[i]["box"]) & _boundary(objects[j]["box"])
            colours = {objects[i]["colour"], objects[j]["colour"]}
            observed = {grid[r][c] for r, c in shared if grid[r][c] in colours}
            if observed == {objects[i]["colour"]}:
                edges.add((j, i))
            elif observed == {objects[j]["colour"]}:
                edges.add((i, j))

    indegree = [0] * len(objects)
    outgoing = {index: set() for index in range(len(objects))}
    for before, after in edges:
        if after not in outgoing[before]:
            outgoing[before].add(after)
            indegree[after] += 1
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        ready.sort(key=lambda index: (-_box_area(objects[index]["box"]), objects[index]["colour"], objects[index]["box"]))
        current = ready.pop(0)
        order.append(current)
        for neighbour in sorted(outgoing[current]):
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                ready.append(neighbour)
    if len(order) != len(objects):
        raise IRRuntimeError("rectangle precedence graph contains a cycle")
    return order


def _render_concentric(objects: list[dict[str, Any]], order: list[int]) -> Grid:
    size = 2 * len(order) - 1
    output = [[0 for _ in range(size)] for _ in range(size)]
    for depth, index in enumerate(order):
        colour = int(objects[index]["colour"])
        for r in range(depth, size - depth):
            for c in range(depth, size - depth):
                output[r][c] = colour
    return tuple(tuple(row) for row in output)


def _blank_axis(grid: Grid) -> list[Point]:
    height, width = len(grid), len(grid[0])
    bg = 0 if any(cell == 0 for row in grid for cell in row) else _mode(grid)
    foreground = [(r, c) for r in range(height) for c in range(width) if grid[r][c] != bg]
    if not foreground:
        raise IRRuntimeError("blank-axis inference requires foreground")
    r0, c0, r1, c1 = _bbox(foreground)
    blank_rows = [r for r in range(r0, r1 + 1) if all(value == bg for value in grid[r])]
    blank_cols = [c for c in range(c0, c1 + 1) if all(grid[r][c] == bg for r in range(height))]
    if len(blank_rows) + len(blank_cols) != 1:
        raise IRRuntimeError("internal blank axis is ambiguous")
    if blank_rows:
        return [(blank_rows[0], c) for c in range(width)]
    return [(r, blank_cols[0]) for r in range(height)]


def _corner_motifs(grid: Grid) -> list[dict[str, Any]]:
    height, width = len(grid), len(grid[0])
    bg = _mode(grid)
    motifs: list[dict[str, Any]] = []
    for colour in sorted({cell for row in grid for cell in row if cell != bg}):
        for r in range(height - 1):
            for c in range(width - 1):
                cells = {(dr, dc): grid[r + dr][c + dc] for dr in (0, 1) for dc in (0, 1)}
                filled = [offset for offset, value in cells.items() if value == colour]
                empty = [offset for offset, value in cells.items() if value == bg]
                if len(filled) == 3 and len(empty) == 1:
                    missing = empty[0]
                    motifs.append({"colour": colour, "row": r, "col": c, "missing": missing})
    unique = {(m["colour"], m["row"], m["col"], tuple(m["missing"])): m for m in motifs}
    return [unique[key] for key in sorted(unique)]


def _ray_points(grid: Grid, motifs: list[dict[str, Any]]) -> list[tuple[Point, int]]:
    height, width = len(grid), len(grid[0])
    points: list[tuple[Point, int]] = []
    for motif in motifs:
        mr, mc = motif["missing"]
        dr = -1 if mr == 0 else 1
        dc = -1 if mc == 0 else 1
        row = motif["row"] + mr
        col = motif["col"] + mc
        while True:
            row += dr
            col += dc
            if not (0 <= row < height and 0 <= col < width):
                break
            points.append(((row, col), int(motif["colour"])))
    return points


def _paint(grid: Grid, points: Iterable[Any], colour: int | None = None) -> Grid:
    values = [list(row) for row in grid]
    height, width = len(values), len(values[0])
    for item in points:
        if colour is None:
            (r, c), item_colour = item
        else:
            r, c = item
            item_colour = colour
        if 0 <= r < height and 0 <= c < width:
            values[r][c] = int(item_colour)
    return tuple(tuple(row) for row in values)


def _edge_project(grid: Grid, fill: int) -> Grid:
    height, width = len(grid), len(grid[0])
    output = [[fill for _ in range(width + 2)] for _ in range(height + 2)]
    for r in range(height):
        for c in range(width):
            output[r + 1][c + 1] = grid[r][c]
    for c in range(width):
        output[0][c + 1] = grid[0][c]
        output[-1][c + 1] = grid[-1][c]
    for r in range(height):
        output[r + 1][0] = grid[r][0]
        output[r + 1][-1] = grid[r][-1]
    return tuple(tuple(row) for row in output)


def evaluate(node: Any, input_grid: Grid) -> Any:
    if isinstance(node, (int, str, bool)) or node is None:
        return node
    if isinstance(node, list):
        return [evaluate(item, input_grid) for item in node]
    if not isinstance(node, dict) or "op" not in node:
        return node
    op = node["op"]
    if op == "input":
        return input_grid
    if op == "background":
        return _mode(evaluate(node["grid"], input_grid))
    if op == "singleton":
        grid = evaluate(node["grid"], input_grid)
        colour = int(evaluate(node["colour"], input_grid))
        points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour]
        if len(points) != 1:
            raise IRRuntimeError("singleton role is ambiguous")
        return points[0]
    if op == "unit_step_towards":
        source = evaluate(node["source"], input_grid)
        target = evaluate(node["target"], input_grid)
        return (
            1 if target[0] > source[0] else -1 if target[0] < source[0] else 0,
            1 if target[1] > source[1] else -1 if target[1] < source[1] else 0,
        )
    if op == "recolour":
        grid = evaluate(node["grid"], input_grid)
        mapping = {int(key): int(value) for key, value in node["mapping"].items()}
        return tuple(tuple(mapping.get(value, value) for value in row) for row in grid)
    if op == "move_point":
        grid = evaluate(node["grid"], input_grid)
        point = evaluate(node["point"], input_grid)
        delta = evaluate(node["delta"], input_grid)
        erase = int(evaluate(node["erase"], input_grid))
        colour = int(evaluate(node["colour"], input_grid))
        values = [list(row) for row in grid]
        values[point[0]][point[1]] = erase
        target = (point[0] + delta[0], point[1] + delta[1])
        values[target[0]][target[1]] = colour
        return tuple(tuple(row) for row in values)
    if op == "edge_project":
        return _edge_project(
            evaluate(node["grid"], input_grid),
            int(evaluate(node["fill"], input_grid)),
        )
    if op == "separator_role":
        return _separator_colour(evaluate(node["grid"], input_grid))
    if op == "decode_cells":
        return _decode_cells(
            evaluate(node["grid"], input_grid),
            int(evaluate(node["separator"], input_grid)),
        )
    if op == "transform":
        return _transform(evaluate(node["grid"], input_grid), str(node["name"]))
    if op == "partition":
        return _partition(
            evaluate(node["grid"], input_grid),
            int(node["rows"]),
            int(node["cols"]),
        )
    if op == "overlay":
        return _overlay(
            evaluate(node["tiles"], input_grid),
            [int(value) for value in node["order"]],
            int(evaluate(node["background"], input_grid)),
        )
    if op == "rect_objects":
        return _rect_objects(evaluate(node["grid"], input_grid), str(node["mode"]))
    if op == "rect_order":
        grid = evaluate(node["grid"], input_grid)
        objects = evaluate(node["objects"], input_grid)
        return _rect_order(grid, objects, str(node["mode"]))
    if op == "render_concentric":
        objects = evaluate(node["objects"], input_grid)
        order = evaluate(node["order"], input_grid)
        return _render_concentric(objects, order)
    if op == "blank_axis":
        return _blank_axis(evaluate(node["grid"], input_grid))
    if op == "corner_motifs":
        return _corner_motifs(evaluate(node["grid"], input_grid))
    if op == "ray_points":
        grid = evaluate(node["grid"], input_grid)
        motifs = evaluate(node["motifs"], input_grid)
        return _ray_points(grid, motifs)
    if op == "paint":
        grid = evaluate(node["grid"], input_grid)
        points = evaluate(node["points"], input_grid)
        colour_node = node.get("colour")
        colour = None if colour_node is None else int(evaluate(colour_node, input_grid))
        return _paint(grid, points, colour)
    raise IRRuntimeError(f"unknown IR opcode: {op}")


def execute(ast: AST, grid: Grid) -> Grid:
    result = evaluate(ast, as_grid(grid))
    return as_grid(result)

