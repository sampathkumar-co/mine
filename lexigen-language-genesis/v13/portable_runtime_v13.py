from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable, Sequence

PortableGrid = list[list[int]]
Point = tuple[int, int]


class PortableLatentError(RuntimeError):
    pass


def as_portable(value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = [[int(cell) for cell in row] for row in value]
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortableLatentError("invalid grid")
    return grid


def execute_portable(program: dict[str, Any], value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = as_portable(value)
    if program.get("schema") != "lexigen-latent-generator-v1":
        raise PortableLatentError("wrong schema")
    operator = str(program["operator"])
    params = dict(program.get("parameters", {}))
    table = {
        "periodic_axis_field": periodic_axis,
        "modular_separator_lattice": modular_lattice,
        "recover_periodic_patch": periodic_patch,
        "masked_tensor_expansion": tensor_expand,
        "broadcast_reference_tile": tile_broadcast,
        "enclosure_interior_classification": enclosure_classify,
        "legend_palette_permutation": legend_permute,
        "ordered_colour_summary": colour_summary,
        "reflect_component_positions": reflect_positions,
        "reconstruct_periodic_lattice": reconstruct_lattice,
        "marker_parameterised_cross_shift": shifted_cross,
        "indexed_legend_template_broadcast": indexed_broadcast,
        "component_seed_propagation": seed_propagation,
        "finite_state_component_recolour": state_recolour,
    }
    try:
        function = table[operator]
    except KeyError as exc:
        raise PortableLatentError(f"unsupported operator: {operator}") from exc
    return function(grid, params)


def _dominant(grid: PortableGrid) -> int:
    counts = Counter(value for row in grid for value in row)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def periodic_axis(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    background = int(params["background"])
    points = [(r, c, grid[r][c]) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != background]
    if len(points) != 2:
        raise PortableLatentError("two anchors required")
    height, width = len(grid), len(grid[0])
    output = [[background] * width for _ in range(height)]
    border_rows = all(row in {0, height - 1} for row, _, _ in points)
    border_cols = all(col in {0, width - 1} for _, col, _ in points)
    if border_rows or points[0][0] == points[1][0]:
        anchors = sorted((col, colour) for _, col, colour in points)
        step = anchors[1][0] - anchors[0][0]
        if step <= 0:
            raise PortableLatentError("bad period")
        for index, col in enumerate(range(anchors[0][0], width, step)):
            colour = anchors[index % 2][1]
            for row in range(height):
                output[row][col] = colour
    elif border_cols or points[0][1] == points[1][1]:
        anchors = sorted((row, colour) for row, _, colour in points)
        step = anchors[1][0] - anchors[0][0]
        if step <= 0:
            raise PortableLatentError("bad period")
        for index, row in enumerate(range(anchors[0][0], height, step)):
            colour = anchors[index % 2][1]
            output[row] = [colour] * width
    else:
        raise PortableLatentError("axis unknown")
    return output


def modular_lattice(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    height = int(params["output_height"])
    width = int(params["output_width"])
    background = int(params["background"])
    size = len(grid)
    if size != len(grid[0]) or len({value for row in grid for value in row}) != 1:
        raise PortableLatentError("uniform square seed required")
    colour = grid[0][0]
    modulus = size + 1
    return [
        [colour if row % modulus == size or col % modulus == size else background for col in range(width)]
        for row in range(height)
    ]


def _period_model(grid: PortableGrid, background: int):
    best = None
    height, width = len(grid), len(grid[0])
    for row_period in range(1, min(6, height) + 1):
        for col_period in range(1, min(6, width) + 1):
            slots: dict[tuple[int, int], set[int]] = {}
            for row in range(height):
                for col in range(width):
                    value = grid[row][col]
                    if value != background:
                        slots.setdefault((row % row_period, col % col_period), set()).add(value)
            if len(slots) != row_period * col_period or any(len(values) != 1 for values in slots.values()):
                continue
            model = {key: next(iter(values)) for key, values in slots.items()}
            rank = (row_period * col_period, row_period + col_period, row_period, col_period)
            if best is None or rank < best[0]:
                best = (rank, row_period, col_period, model)
    if best is None:
        raise PortableLatentError("no periodic model")
    return best[1], best[2], best[3]


def periodic_patch(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    background = int(params["background"])
    row_period, col_period, model = _period_model(grid, background)
    missing = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            expected = model[(row % row_period, col % col_period)]
            if grid[row][col] == background and expected != background:
                missing.append((row, col, expected))
    if not missing:
        raise PortableLatentError("nothing missing")
    top = min(row for row, _, _ in missing)
    left = min(col for _, col, _ in missing)
    bottom = max(row for row, _, _ in missing)
    right = max(col for _, col, _ in missing)
    output = [[background for _ in range(right - left + 1)] for _ in range(bottom - top + 1)]
    for row, col, value in missing:
        output[row - top][col - left] = value
    return output


def tensor_expand(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    trigger = int(params["trigger"])
    background = int(params["background"])
    height, width = len(grid), len(grid[0])
    output = [[background for _ in range(width * width)] for _ in range(height * height)]
    for outer_row in range(height):
        for outer_col in range(width):
            if grid[outer_row][outer_col] != trigger:
                continue
            for inner_row in range(height):
                for inner_col in range(width):
                    output[outer_row * height + inner_row][outer_col * width + inner_col] = grid[inner_row][inner_col]
    return output


def _line_cuts(grid: PortableGrid, colour: int, axis: str) -> list[int]:
    if axis == "row":
        return [i for i, row in enumerate(grid) if all(value == colour for value in row)]
    return [col for col in range(len(grid[0])) if all(row[col] == colour for row in grid)]


def _ranges(size: int, cuts: Iterable[int]) -> list[tuple[int, int]]:
    result = []
    start = 0
    for cut in sorted(set(cuts)):
        if cut > start:
            result.append((start, cut))
        start = cut + 1
    if start < size:
        result.append((start, size))
    return result


def tile_broadcast(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    separator = int(params["separator"])
    background = int(params["background"])
    rows = _ranges(len(grid), _line_cuts(grid, separator, "row"))
    cols = _ranges(len(grid[0]), _line_cuts(grid, separator, "col"))
    candidates = []
    for row_index, (r0, r1) in enumerate(rows):
        for col_index, (c0, c1) in enumerate(cols):
            pattern = [(r - r0, c - c0, grid[r][c]) for r in range(r0, r1) for c in range(c0, c1) if grid[r][c] != background]
            candidates.append((len(pattern), -row_index, -col_index, pattern))
    count, _, _, pattern = max(candidates)
    if count == 0:
        raise PortableLatentError("empty tiles")
    output = [row[:] for row in grid]
    for r0, r1 in rows:
        for c0, c1 in cols:
            for local_row, local_col, value in pattern:
                row, col = r0 + local_row, c0 + local_col
                if row < r1 and col < c1:
                    output[row][col] = value
    return output


def _squares(grid: PortableGrid, colour: int):
    height, width = len(grid), len(grid[0])
    candidates = []
    for top in range(height):
        for left in range(width):
            for size in range(3, min(height - top, width - left) + 1):
                bottom, right = top + size - 1, left + size - 1
                border = [(top, col) for col in range(left, right + 1)] + [(bottom, col) for col in range(left, right + 1)] + [(row, left) for row in range(top + 1, bottom)] + [(row, right) for row in range(top + 1, bottom)]
                if all(grid[row][col] == colour for row, col in border):
                    candidates.append((top, left, size))
    maximal = []
    for item in candidates:
        top, left, size = item
        contained = any(
            other != item
            and other[0] <= top
            and other[1] <= left
            and other[0] + other[2] >= top + size
            and other[1] + other[2] >= left + size
            for other in candidates
        )
        if not contained:
            maximal.append(item)
    return maximal


def enclosure_classify(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    border = int(params["border_colour"])
    mapping = {int(key): int(value) for key, value in params["thickness_to_colour"].items()}
    output = [row[:] for row in grid]
    for top, left, size in _squares(grid, border):
        thickness = (size - 3) // 2
        if thickness not in mapping:
            raise PortableLatentError("unmapped thickness")
        fill = mapping[thickness]
        for row in range(top + 1, top + size - 1):
            for col in range(left + 1, left + size - 1):
                if grid[row][col] != border:
                    output[row][col] = fill
    return output


def legend_permute(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    rows = int(params.get("legend_rows", 2))
    cols = int(params.get("legend_cols", 2))
    legend = [[grid[row][col] for col in range(cols)] for row in range(rows)]
    mapping = {legend[0][0]: legend[0][1], legend[0][1]: legend[0][0], legend[1][0]: legend[1][1], legend[1][1]: legend[1][0]}
    output = [row[:] for row in grid]
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if row < rows and col < cols:
                continue
            if grid[row][col] in mapping:
                output[row][col] = mapping[grid[row][col]]
    return output


def colour_summary(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    background = int(params["background"])
    height = int(params["output_height"])
    width = int(params["output_width"])
    positions: dict[int, list[Point]] = {}
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            value = grid[row][col]
            if value != background:
                positions.setdefault(value, []).append((row, col))
    ordered = sorted((min(row for row, _ in points), min(col for _, col in points), colour) for colour, points in positions.items())
    if len(ordered) > height:
        raise PortableLatentError("too many colours")
    output = [[background] * width for _ in range(height)]
    offset = height - len(ordered)
    for index, (_, _, colour) in enumerate(ordered):
        output[offset + index] = [colour] * width
    return output


def _components(grid: PortableGrid, background: int):
    points = {(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if grid[row][col] != background}
    result = []
    while points:
        start = next(iter(points))
        points.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in points:
                    points.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        result.append(component)
    return result


def reflect_positions(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    background = int(params["background"])
    width = len(grid[0])
    output = [[background for _ in row] for row in grid]
    for component in _components(grid, background):
        left = min(col for _, col in component)
        right = max(col for _, col in component)
        target_left = width - (right + 1)
        for row, col in component:
            output[row][target_left + (col - left)] = grid[row][col]
    return output


def reconstruct_lattice(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    noise = int(params["noise_colour"])
    height, width = len(grid), len(grid[0])
    visible = [value for row in grid for value in row if value != noise]
    if len(set(visible)) != 2:
        raise PortableLatentError("two visible colours required")
    counts = Counter(visible)
    background = max(counts, key=lambda colour: (counts[colour], -colour))
    foreground = next(colour for colour in counts if colour != background)
    candidates = []
    for length in range(1, min(4, height + 1, width + 1)):
        for spacing in range(length + 1, min(7, max(height, width) + 1)):
            for row_phase in range(spacing):
                for col_phase in range(spacing):
                    output = [[background] * width for _ in range(height)]
                    for top in range(row_phase, height, spacing):
                        for left in range(col_phase, width, spacing):
                            for row in range(top, min(top + length, height)):
                                for col in range(left, min(left + length, width)):
                                    output[row][col] = foreground
                    compatible = all(grid[row][col] == noise or grid[row][col] == output[row][col] for row in range(height) for col in range(width))
                    if not compatible:
                        continue
                    observed = sum(value == foreground for row in grid for value in row)
                    expected = sum(value == foreground for row in output for value in row)
                    if observed == 0 or expected < observed:
                        continue
                    candidates.append(((expected - observed, spacing, length, row_phase, col_phase), output))
    if not candidates:
        raise PortableLatentError("no lattice")
    return min(candidates, key=lambda item: item[0])[1]


def shifted_cross(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    marker = int(params["marker"])
    background = int(params["background"])
    count = sum(value == marker for row in grid for value in row)
    colours = [value for row in grid for value in row if value not in {marker, background}]
    if not colours:
        raise PortableLatentError("cross colour absent")
    colour = Counter(colours).most_common(1)[0][0]
    rows = [row for row in range(len(grid)) if all(value == colour for value in grid[row])]
    cols = [col for col in range(len(grid[0])) if all(grid[row][col] == colour for row in range(len(grid)))]
    if len(rows) != 1 or len(cols) != 1:
        raise PortableLatentError("cross ambiguous")
    target_row, target_col = rows[0] + count, cols[0] - count
    output = [[background for _ in row] for row in grid]
    for col in range(len(grid[0])):
        output[target_row][col] = colour
    for row in range(len(grid)):
        output[row][target_col] = colour
    return output


def _legend(grid: PortableGrid, background: int):
    candidates = []
    for row in range(len(grid)):
        entries = [(row, col, grid[row][col]) for col in range(len(grid[0])) if grid[row][col] != background]
        diversity = len({value for _, _, value in entries})
        if len(entries) >= 3 and diversity >= 2:
            candidates.append(((diversity, len(entries), -row, 1), entries))
    for col in range(len(grid[0])):
        entries = [(row, col, grid[row][col]) for row in range(len(grid)) if grid[row][col] != background]
        diversity = len({value for _, _, value in entries})
        if len(entries) >= 3 and diversity >= 2:
            candidates.append(((diversity, len(entries), -col, 0), entries))
    if not candidates:
        raise PortableLatentError("legend missing")
    return max(candidates, key=lambda item: item[0])[1]


def _colour_components(grid: PortableGrid, colour: int, excluded: set[Point]):
    points = {(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if grid[row][col] == colour and (row, col) not in excluded}
    result = []
    while points:
        start = next(iter(points))
        points.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in points:
                    points.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        result.append(component)
    return result


def indexed_broadcast(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    key_colour = int(params["key_colour"])
    stride = int(params["index_stride"])
    background = _dominant(grid)
    legend = _legend(grid, background)
    excluded = {(row, col) for row, col, _ in legend}
    key_entries = [(row, col) for row, col, colour in legend if colour == key_colour]
    if len(key_entries) != 1:
        raise PortableLatentError("key ambiguous")
    key_row, key_col = key_entries[0]
    components = _colour_components(grid, key_colour, excluded)
    if not components:
        raise PortableLatentError("template absent")
    template = max(components, key=lambda component: (len(component), -min(row for row, _ in component), -min(col for _, col in component)))
    same_row = len({row for row, _, _ in legend}) == 1
    same_col = len({col for _, col, _ in legend}) == 1
    if same_row == same_col:
        raise PortableLatentError("axis ambiguous")
    ordered = sorted(legend, key=(lambda item: item[1]) if same_row else (lambda item: item[0]))
    key_index = next(index for index, (row, col, _) in enumerate(ordered) if (row, col) == (key_row, key_col))
    if key_index not in {0, len(ordered) - 1}:
        raise PortableLatentError("key not endpoint")
    output = [row[:] for row in grid]
    for index, (marker_row, marker_col, colour) in enumerate(ordered):
        distance = abs(index - key_index)
        row_delta, col_delta = marker_row - key_row, marker_col - key_col
        if same_row and distance:
            col_delta += (1 if col_delta > 0 else -1) * stride * distance
        if same_col and distance:
            row_delta += (1 if row_delta > 0 else -1) * stride * distance
        for row, col in template:
            output[row + row_delta][col + col_delta] = colour
    return output


def seed_propagation(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    background = int(params["background"])
    mask = int(params["mask_colour"])
    height, width = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    seeds = [(row, col, grid[row][col]) for row in range(height) for col in range(width) if grid[row][col] not in {background, mask}]
    for row, col, colour in seeds:
        neighbours = [(row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        starts = [(r, c) for r, c in neighbours if 0 <= r < height and 0 <= c < width and grid[r][c] == mask]
        output[row][col] = background
        if len(starts) != 1:
            continue
        queue = deque(starts)
        seen = set(starts)
        while queue:
            r, c = queue.popleft()
            output[r][c] = colour
            for neighbour in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                nr, nc = neighbour
                if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == mask and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
    return output


def _bands(grid: PortableGrid, background: int):
    bands = []
    row = 0
    while row < len(grid):
        cells = [(col, value) for col, value in enumerate(grid[row]) if value != background]
        if not cells:
            row += 1
            continue
        colour = Counter(value for _, value in cells).most_common(1)[0][0]
        interval = (min(col for col, _ in cells), max(col for col, _ in cells) + 1)
        stop = row + 1
        while stop < len(grid):
            next_cells = [(col, value) for col, value in enumerate(grid[stop]) if value != background]
            if not next_cells:
                break
            next_colour = Counter(value for _, value in next_cells).most_common(1)[0][0]
            next_interval = (min(col for col, _ in next_cells), max(col for col, _ in next_cells) + 1)
            if next_colour != colour or next_interval != interval:
                break
            stop += 1
        bands.append((row, stop, colour))
        row = stop
    return bands


def state_recolour(grid: PortableGrid, params: dict[str, Any]) -> PortableGrid:
    background = int(params["background"])
    cycles = {int(key): [int(value) for value in values] for key, values in params["cycles"].items()}
    bands = _bands(grid, background)
    if not bands or bands[0][2] not in cycles:
        raise PortableLatentError("cycle unavailable")
    cycle = cycles[bands[0][2]]
    output = [row[:] for row in grid]
    for index, (start, stop, _) in enumerate(bands):
        colour = cycle[index % len(cycle)]
        for row in range(start, stop):
            for col in range(len(grid[0])):
                if grid[row][col] != background:
                    output[row][col] = colour
    return output
