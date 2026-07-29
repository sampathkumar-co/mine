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


def _area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _rect_order(grid: Grid, objects, mode: str):
    arrows = set()
    if mode == "components":
        for left_index, left in enumerate(objects):
            for right_index, right in enumerate(objects):
                if left_index != right_index and _inside(left["box"], right["box"]):
                    arrows.add((left_index, right_index))
    for left_index in range(len(objects)):
        for right_index in range(left_index + 1, len(objects)):
            common = _border(objects[left_index]["box"]) & _border(objects[right_index]["box"])
            palette = {objects[left_index]["colour"], objects[right_index]["colour"]}
            visible = {grid[r][c] for r, c in common if grid[r][c] in palette}
            if visible == {objects[left_index]["colour"]}:
                arrows.add((right_index, left_index))
            elif visible == {objects[right_index]["colour"]}:
                arrows.add((left_index, right_index))

    indegree = [0 for _ in objects]
    outgoing = {index: set() for index in range(len(objects))}
    for before, after in arrows:
        if after not in outgoing[before]:
            outgoing[before].add(after)
            indegree[after] += 1
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    ordered = []
    while ready:
        ready.sort(key=lambda index: (-_area(objects[index]["box"]), objects[index]["colour"], objects[index]["box"]))
        current = ready.pop(0)
        ordered.append(current)
        for neighbour in sorted(outgoing[current]):
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                ready.append(neighbour)
    if len(ordered) != len(objects):
        raise PortableIRError("rectangle precedence cycle")
    return ordered


def _concentric(objects, order):
    size = 2 * len(order) - 1
    result = [[0 for _ in range(size)] for _ in range(size)]
    for depth, object_index in enumerate(order):
        colour = int(objects[object_index]["colour"])
        for r in range(depth, size - depth):
            for c in range(depth, size - depth):
                result[r][c] = colour
    return tuple(tuple(row) for row in result)


def _blank_axis(grid: Grid):
    height, width = len(grid), len(grid[0])
    background = 0 if any(cell == 0 for row in grid for cell in row) else _background(grid)
    foreground = [(r, c) for r in range(height) for c in range(width) if grid[r][c] != background]
    if not foreground:
        raise PortableIRError("blank-axis object missing")
    r0, c0, r1, c1 = _bbox(foreground)
    rows = [r for r in range(r0, r1 + 1) if all(grid[r][c] == background for c in range(width))]
    cols = [c for c in range(c0, c1 + 1) if all(grid[r][c] == background for r in range(height))]
    if len(rows) + len(cols) != 1:
        raise PortableIRError("blank axis is ambiguous")
    if rows:
        return [(rows[0], c) for c in range(width)]
    return [(r, cols[0]) for r in range(height)]


def _corner_motifs(grid: Grid):
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    found = {}
    for colour in sorted({cell for row in grid for cell in row if cell != background}):
        for row in range(height - 1):
            for col in range(width - 1):
                samples = [((dr, dc), grid[row + dr][col + dc]) for dr in (0, 1) for dc in (0, 1)]
                filled = [offset for offset, value in samples if value == colour]
                empty = [offset for offset, value in samples if value == background]
                if len(filled) == 3 and len(empty) == 1:
                    key = (colour, row, col, empty[0])
                    found[key] = {"colour": colour, "row": row, "col": col, "missing": empty[0]}
    return [found[key] for key in sorted(found)]


def _ray_points(grid: Grid, motifs):
    height, width = len(grid), len(grid[0])
    output = []
    for motif in motifs:
        missing_row, missing_col = motif["missing"]
        dr = -1 if missing_row == 0 else 1
        dc = -1 if missing_col == 0 else 1
        row = motif["row"] + missing_row
        col = motif["col"] + missing_col
        while True:
            row += dr
            col += dc
            if row < 0 or row >= height or col < 0 or col >= width:
                break
            output.append(((row, col), int(motif["colour"])))
    return output


def _paint(grid: Grid, points, colour=None):
    result = [list(row) for row in grid]
    height, width = len(result), len(result[0])
    for item in points:
        if colour is None:
            (row, col), selected = item
        else:
            row, col = item
            selected = colour
        if 0 <= row < height and 0 <= col < width:
            result[row][col] = int(selected)
    return tuple(tuple(row) for row in result)


def _edge_project(grid: Grid, fill: int):
    height, width = len(grid), len(grid[0])
    result = [[fill for _ in range(width + 2)] for _ in range(height + 2)]
    for row in range(height):
        for col in range(width):
            result[row + 1][col + 1] = grid[row][col]
    for col in range(width):
        result[0][col + 1] = grid[0][col]
        result[-1][col + 1] = grid[-1][col]
    for row in range(height):
        result[row + 1][0] = grid[row][0]
        result[row + 1][-1] = grid[row][-1]
    return tuple(tuple(row) for row in result)


def _evaluate(node: Any, input_grid: Grid):
    if isinstance(node, (int, str, bool)) or node is None:
        return node
    if isinstance(node, list):
        return [_evaluate(item, input_grid) for item in node]
    if not isinstance(node, dict) or "op" not in node:
        return node
    opcode = str(node["op"])
    if opcode == "input":
        return input_grid
    if opcode == "background":
        return _background(_evaluate(node["grid"], input_grid))
    if opcode == "singleton":
        grid = _evaluate(node["grid"], input_grid)
        colour = int(_evaluate(node["colour"], input_grid))
        points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour]
        if len(points) != 1:
            raise PortableIRError("singleton role is ambiguous")
        return points[0]
    if opcode == "unit_step_towards":
        source = _evaluate(node["source"], input_grid)
        target = _evaluate(node["target"], input_grid)
        return (
            1 if target[0] > source[0] else -1 if target[0] < source[0] else 0,
            1 if target[1] > source[1] else -1 if target[1] < source[1] else 0,
        )
    if opcode == "recolour":
        grid = _evaluate(node["grid"], input_grid)
        mapping = {int(key): int(value) for key, value in node["mapping"].items()}
        return tuple(tuple(mapping.get(value, value) for value in row) for row in grid)
    if opcode == "move_point":
        grid = _evaluate(node["grid"], input_grid)
        point = _evaluate(node["point"], input_grid)
        delta = _evaluate(node["delta"], input_grid)
        erased = int(_evaluate(node["erase"], input_grid))
        colour = int(_evaluate(node["colour"], input_grid))
        result = [list(row) for row in grid]
        result[point[0]][point[1]] = erased
        row, col = point[0] + delta[0], point[1] + delta[1]
        result[row][col] = colour
        return tuple(tuple(line) for line in result)
    if opcode == "edge_project":
        return _edge_project(_evaluate(node["grid"], input_grid), int(_evaluate(node["fill"], input_grid)))
    if opcode == "separator_role":
        return _separator(_evaluate(node["grid"], input_grid))
    if opcode == "decode_cells":
        return _decode(
            _evaluate(node["grid"], input_grid),
            int(_evaluate(node["separator"], input_grid)),
        )
    if opcode == "transform":
        return _xform(_evaluate(node["grid"], input_grid), str(node["name"]))
    if opcode == "partition":
        return _tiles(
            _evaluate(node["grid"], input_grid),
            int(node["rows"]),
            int(node["cols"]),
        )
    if opcode == "overlay":
        return _overlay(
            _evaluate(node["tiles"], input_grid),
            [int(value) for value in node["order"]],
            int(_evaluate(node["background"], input_grid)),
        )
    if opcode == "rect_objects":
        return _rect_objects(_evaluate(node["grid"], input_grid), str(node["mode"]))
    if opcode == "rect_order":
        grid = _evaluate(node["grid"], input_grid)
        objects = _evaluate(node["objects"], input_grid)
        return _rect_order(grid, objects, str(node["mode"]))
    if opcode == "render_concentric":
        objects = _evaluate(node["objects"], input_grid)
        order = _evaluate(node["order"], input_grid)
        return _concentric(objects, order)
    if opcode == "blank_axis":
        return _blank_axis(_evaluate(node["grid"], input_grid))
    if opcode == "corner_motifs":
        return _corner_motifs(_evaluate(node["grid"], input_grid))
    if opcode == "ray_points":
        grid = _evaluate(node["grid"], input_grid)
        motifs = _evaluate(node["motifs"], input_grid)
        return _ray_points(grid, motifs)
    if opcode == "paint":
        grid = _evaluate(node["grid"], input_grid)
        points = _evaluate(node["points"], input_grid)
        colour_node = node.get("colour")
        colour = None if colour_node is None else int(_evaluate(colour_node, input_grid))
        return _paint(grid, points, colour)
    raise PortableIRError(f"unknown opcode: {opcode}")


def execute_portable_ir(ast: dict[str, Any], grid: Any) -> Grid:
    input_grid = _grid(grid)
    return _grid(_evaluate(ast, input_grid))
