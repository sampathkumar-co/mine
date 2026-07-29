from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


class LatentRuntimeError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise LatentRuntimeError("grid must be a non-empty rectangle")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def most_common_colour(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    maximum = max(counts.values())
    return min(colour for colour, count in counts.items() if count == maximum)


def execute_program(program: dict[str, Any], source: Grid) -> Grid:
    if program.get("schema") != "lexigen-latent-generator-v1":
        raise LatentRuntimeError("unsupported latent-generator schema")
    operator = str(program["operator"])
    parameters = dict(program.get("parameters", {}))
    dispatch = {
        "periodic_axis_field": periodic_axis_field,
        "modular_separator_lattice": modular_separator_lattice,
        "recover_periodic_patch": recover_periodic_patch,
        "broadcast_reference_tile": broadcast_reference_tile,
        "legend_palette_permutation": legend_palette_permutation,
        "replace_colour": replace_colour,
        "marker_parameterised_cross_shift": marker_parameterised_cross_shift,
        "masked_tensor_expansion": masked_tensor_expansion,
        "enclosure_interior_classification": enclosure_interior_classification,
        "ordered_component_summary": ordered_component_summary,
        "horizontal_reflection": horizontal_reflection,
        "component_seed_propagation": component_seed_propagation,
        "template_colour_broadcast": template_colour_broadcast,
        "finite_state_component_recolour": finite_state_component_recolour,
        "fill_rectangular_interiors": fill_rectangular_interiors,
    }
    if operator not in dispatch:
        raise LatentRuntimeError(f"unknown operator: {operator}")
    return dispatch[operator](source, parameters)


def periodic_axis_field(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    points = [(r, c, value) for r, row in enumerate(source) for c, value in enumerate(row) if value != background]
    if len(points) != 2:
        raise LatentRuntimeError("periodic axis field requires two anchors")
    (r1, c1, colour1), (r2, c2, colour2) = sorted(points)
    height, width = len(source), len(source[0])
    canvas = [[background for _ in range(width)] for _ in range(height)]
    if r1 == r2 and c1 != c2:
        first, second = sorted(((c1, colour1), (c2, colour2)))
        step = second[0] - first[0]
        if step <= 0:
            raise LatentRuntimeError("invalid periodic step")
        index = 0
        for col in range(first[0], width, step):
            colour = (first[1], second[1])[index % 2]
            for row in range(height):
                canvas[row][col] = colour
            index += 1
    elif c1 == c2 and r1 != r2:
        first, second = sorted(((r1, colour1), (r2, colour2)))
        step = second[0] - first[0]
        if step <= 0:
            raise LatentRuntimeError("invalid periodic step")
        index = 0
        for row in range(first[0], height, step):
            colour = (first[1], second[1])[index % 2]
            for col in range(width):
                canvas[row][col] = colour
            index += 1
    else:
        raise LatentRuntimeError("anchors do not define one axis")
    return tuple(tuple(row) for row in canvas)


def modular_separator_lattice(source: Grid, parameters: dict[str, Any]) -> Grid:
    height = int(parameters["output_height"])
    width = int(parameters["output_width"])
    background = int(parameters["background"])
    size = len(source)
    if size != len(source[0]):
        raise LatentRuntimeError("lattice seed must be square")
    colours = {cell for row in source for cell in row}
    if len(colours) != 1:
        raise LatentRuntimeError("lattice seed must be uniform")
    colour = next(iter(colours))
    canvas = [[background for _ in range(width)] for _ in range(height)]
    modulus = size + 1
    for row in range(height):
        for col in range(width):
            if row % modulus == size or col % modulus == size:
                canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)


def _infer_period(source: Grid, background: int, maximum: int = 6) -> tuple[int, int, dict[tuple[int, int], int]]:
    height, width = len(source), len(source[0])
    best = None
    for row_period in range(1, min(maximum, height) + 1):
        for col_period in range(1, min(maximum, width) + 1):
            residues: dict[tuple[int, int], set[int]] = {}
            for row in range(height):
                for col in range(width):
                    value = source[row][col]
                    if value == background:
                        continue
                    residues.setdefault((row % row_period, col % col_period), set()).add(value)
            if not residues or any(len(values) != 1 for values in residues.values()):
                continue
            if len(residues) < row_period * col_period:
                continue
            model = {key: next(iter(values)) for key, values in residues.items()}
            rank = (row_period * col_period, row_period + col_period, row_period, col_period)
            if best is None or rank < best[0]:
                best = (rank, row_period, col_period, model)
    if best is None:
        raise LatentRuntimeError("no periodic model")
    return best[1], best[2], best[3]


def recover_periodic_patch(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    row_period, col_period, model = _infer_period(source, background)
    missing = []
    for row in range(len(source)):
        for col in range(len(source[0])):
            expected = model[(row % row_period, col % col_period)]
            if source[row][col] == background and expected != background:
                missing.append((row, col, expected))
    if not missing:
        raise LatentRuntimeError("periodic model has no missing patch")
    min_row = min(row for row, _, _ in missing)
    max_row = max(row for row, _, _ in missing)
    min_col = min(col for _, col, _ in missing)
    max_col = max(col for _, col, _ in missing)
    canvas = [[background for _ in range(max_col - min_col + 1)] for _ in range(max_row - min_row + 1)]
    for row, col, value in missing:
        canvas[row - min_row][col - min_col] = value
    return tuple(tuple(row) for row in canvas)


def _full_line_indices(source: Grid, colour: int, axis: str) -> list[int]:
    if axis == "row":
        return [index for index, row in enumerate(source) if all(value == colour for value in row)]
    return [index for index in range(len(source[0])) if all(row[index] == colour for row in source)]


def _intervals(size: int, cuts: Iterable[int]) -> list[tuple[int, int]]:
    result = []
    start = 0
    for cut in sorted(set(cuts)):
        if cut > start:
            result.append((start, cut))
        start = cut + 1
    if start < size:
        result.append((start, size))
    return result


def broadcast_reference_tile(source: Grid, parameters: dict[str, Any]) -> Grid:
    separator = int(parameters["separator"])
    background = int(parameters["background"])
    rows = _intervals(len(source), _full_line_indices(source, separator, "row"))
    cols = _intervals(len(source[0]), _full_line_indices(source, separator, "col"))
    if not rows or not cols:
        raise LatentRuntimeError("no tile lattice")
    candidates = []
    for row_index, (r0, r1) in enumerate(rows):
        for col_index, (c0, c1) in enumerate(cols):
            cells = [(r - r0, c - c0, source[r][c]) for r in range(r0, r1) for c in range(c0, c1) if source[r][c] != background]
            candidates.append((len(cells), row_index, col_index, cells))
    count, _, _, pattern = max(candidates)
    if count == 0:
        raise LatentRuntimeError("all tiles empty")
    canvas = [list(row) for row in source]
    for r0, r1 in rows:
        for c0, c1 in cols:
            for local_row, local_col, value in pattern:
                row, col = r0 + local_row, c0 + local_col
                if row < r1 and col < c1:
                    canvas[row][col] = value
    return tuple(tuple(row) for row in canvas)


def legend_palette_permutation(source: Grid, parameters: dict[str, Any]) -> Grid:
    legend_rows = int(parameters.get("legend_rows", 2))
    legend_cols = int(parameters.get("legend_cols", 2))
    if len(source) < legend_rows or len(source[0]) < legend_cols:
        raise LatentRuntimeError("legend does not fit")
    legend = [[source[row][col] for col in range(legend_cols)] for row in range(legend_rows)]
    mapping = {
        legend[0][0]: legend[0][1],
        legend[0][1]: legend[0][0],
        legend[1][0]: legend[1][1],
        legend[1][1]: legend[1][0],
    }
    canvas = [list(row) for row in source]
    for row in range(len(source)):
        for col in range(len(source[0])):
            if row < legend_rows and col < legend_cols:
                continue
            if source[row][col] in mapping:
                canvas[row][col] = mapping[source[row][col]]
    return tuple(tuple(row) for row in canvas)


def replace_colour(source: Grid, parameters: dict[str, Any]) -> Grid:
    old = int(parameters["old"])
    new = int(parameters["new"])
    return tuple(tuple(new if value == old else value for value in row) for row in source)


def marker_parameterised_cross_shift(source: Grid, parameters: dict[str, Any]) -> Grid:
    marker = int(parameters["marker"])
    background = int(parameters["background"])
    marker_count = sum(value == marker for row in source for value in row)
    non_marker = [value for row in source for value in row if value not in {background, marker}]
    if not non_marker:
        raise LatentRuntimeError("cross colour missing")
    colour = Counter(non_marker).most_common(1)[0][0]
    rows = [row for row in range(len(source)) if sum(source[row][col] == colour for col in range(len(source[0]))) == len(source[0])]
    cols = [col for col in range(len(source[0])) if sum(source[row][col] == colour for row in range(len(source))) == len(source)]
    if len(rows) != 1 or len(cols) != 1:
        raise LatentRuntimeError("cross axes are ambiguous")
    target_row = rows[0] + marker_count
    target_col = cols[0] - marker_count
    if not (0 <= target_row < len(source) and 0 <= target_col < len(source[0])):
        raise LatentRuntimeError("shifted cross leaves canvas")
    canvas = [[background for _ in row] for row in source]
    for col in range(len(source[0])):
        canvas[target_row][col] = colour
    for row in range(len(source)):
        canvas[row][target_col] = colour
    return tuple(tuple(row) for row in canvas)


def masked_tensor_expansion(source: Grid, parameters: dict[str, Any]) -> Grid:
    trigger = int(parameters["trigger"])
    background = int(parameters["background"])
    height, width = len(source), len(source[0])
    canvas = [[background for _ in range(width * width)] for _ in range(height * height)]
    for outer_row in range(height):
        for outer_col in range(width):
            if source[outer_row][outer_col] != trigger:
                continue
            for inner_row in range(height):
                for inner_col in range(width):
                    canvas[outer_row * height + inner_row][outer_col * width + inner_col] = source[inner_row][inner_col]
    return tuple(tuple(row) for row in canvas)


def _bordered_squares(source: Grid, border_colour: int) -> list[tuple[int, int, int]]:
    height, width = len(source), len(source[0])
    results = []
    for top in range(height):
        for left in range(width):
            if source[top][left] != border_colour:
                continue
            for size in range(3, min(height - top, width - left) + 1):
                bottom, right = top + size - 1, left + size - 1
                border = [(top, col) for col in range(left, right + 1)] + [(bottom, col) for col in range(left, right + 1)] + [(row, left) for row in range(top + 1, bottom)] + [(row, right) for row in range(top + 1, bottom)]
                if all(source[row][col] == border_colour for row, col in border):
                    results.append((top, left, size))
    maximal = []
    for candidate in results:
        top, left, size = candidate
        if not any(other != candidate and other[0] <= top and other[1] <= left and other[0] + other[2] >= top + size and other[1] + other[2] >= left + size for other in results):
            maximal.append(candidate)
    return maximal


def enclosure_interior_classification(source: Grid, parameters: dict[str, Any]) -> Grid:
    border_colour = int(parameters["border_colour"])
    mapping = {int(key): int(value) for key, value in parameters["thickness_to_colour"].items()}
    canvas = [list(row) for row in source]
    for top, left, size in _bordered_squares(source, border_colour):
        thickness = (size - 3) // 2
        if thickness not in mapping:
            raise LatentRuntimeError("unknown enclosure thickness")
        fill = mapping[thickness]
        for row in range(top + 1, top + size - 1):
            for col in range(left + 1, left + size - 1):
                if source[row][col] != border_colour:
                    canvas[row][col] = fill
    return tuple(tuple(row) for row in canvas)


def _components(source: Grid, background: int) -> list[set[Point]]:
    points = {(row, col) for row in range(len(source)) for col in range(len(source[0])) if source[row][col] != background}
    components = []
    while points:
        start = next(iter(points))
        component = {start}
        queue = deque([start])
        points.remove(start)
        while queue:
            row, col = queue.popleft()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in points:
                    points.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def ordered_component_summary(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    output_height = int(parameters["output_height"])
    output_width = int(parameters["output_width"])
    components = _components(source, background)
    coloured = []
    for component in components:
        colours = [source[row][col] for row, col in component]
        colour = Counter(colours).most_common(1)[0][0]
        coloured.append((min(row for row, _ in component), min(col for _, col in component), colour))
    coloured.sort()
    if len(coloured) > output_height:
        raise LatentRuntimeError("too many components")
    canvas = [[background for _ in range(output_width)] for _ in range(output_height)]
    offset = output_height - len(coloured)
    for index, (_, _, colour) in enumerate(coloured):
        for col in range(output_width):
            canvas[offset + index][col] = colour
    return tuple(tuple(row) for row in canvas)


def horizontal_reflection(source: Grid, parameters: dict[str, Any]) -> Grid:
    return tuple(tuple(reversed(row)) for row in source)


def component_seed_propagation(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    mask_colour = int(parameters["mask_colour"])
    canvas = [list(row) for row in source]
    height, width = len(source), len(source[0])
    seeds = [(row, col, source[row][col]) for row in range(height) for col in range(width) if source[row][col] not in {background, mask_colour}]
    for row, col, colour in seeds:
        neighbours = [(row + dr, col + dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        mask_neighbours = [(r, c) for r, c in neighbours if 0 <= r < height and 0 <= c < width and source[r][c] == mask_colour]
        canvas[row][col] = background
        if len(mask_neighbours) != 1:
            continue
        queue = deque(mask_neighbours)
        seen = set(mask_neighbours)
        while queue:
            r, c = queue.popleft()
            canvas[r][c] = colour
            for nr, nc in ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)):
                if 0 <= nr < height and 0 <= nc < width and source[nr][nc] == mask_colour and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return tuple(tuple(row) for row in canvas)


def template_colour_broadcast(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    key_colour = int(parameters["key_colour"])
    non_background = [(r, c, source[r][c]) for r in range(len(source)) for c in range(len(source[0])) if source[r][c] != background]
    legend_rows = Counter(r for r, _, _ in non_background)
    legend_row = min(row for row, count in legend_rows.items() if count == max(legend_rows.values()))
    legend = sorted((col, colour) for row, col, colour in non_background if row == legend_row)
    key_entries = [(col, colour) for col, colour in legend if colour == key_colour]
    if len(key_entries) != 1:
        raise LatentRuntimeError("template key is ambiguous")
    key_col = key_entries[0][0]
    pattern = [(r, c - key_col) for r, c, value in non_background if r != legend_row and value == key_colour]
    if not pattern:
        raise LatentRuntimeError("template pattern missing")
    canvas = [list(row) for row in source]
    for anchor_col, colour in legend:
        for row, delta_col in pattern:
            col = anchor_col + delta_col
            if 0 <= col < len(source[0]):
                canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)


def _row_bands(source: Grid, background: int) -> list[tuple[int, int, int]]:
    bands = []
    start = 0
    while start < len(source):
        nonzero = [(col, value) for col, value in enumerate(source[start]) if value != background]
        if not nonzero:
            start += 1
            continue
        colour = Counter(value for _, value in nonzero).most_common(1)[0][0]
        interval = (min(col for col, _ in nonzero), max(col for col, _ in nonzero) + 1)
        stop = start + 1
        while stop < len(source):
            row_cells = [(col, value) for col, value in enumerate(source[stop]) if value != background]
            if not row_cells:
                break
            row_colour = Counter(value for _, value in row_cells).most_common(1)[0][0]
            row_interval = (min(col for col, _ in row_cells), max(col for col, _ in row_cells) + 1)
            if row_colour != colour or row_interval != interval:
                break
            stop += 1
        bands.append((start, stop, colour))
        start = stop
    return bands


def finite_state_component_recolour(source: Grid, parameters: dict[str, Any]) -> Grid:
    background = int(parameters["background"])
    cycles = {int(key): tuple(int(value) for value in values) for key, values in parameters["cycles"].items()}
    bands = _row_bands(source, background)
    if not bands:
        raise LatentRuntimeError("no sequence components")
    first_colour = bands[0][2]
    if first_colour not in cycles:
        raise LatentRuntimeError("unknown sequence phase")
    cycle = cycles[first_colour]
    canvas = [list(row) for row in source]
    for index, (start, stop, _) in enumerate(bands):
        colour = cycle[index % len(cycle)]
        for row in range(start, stop):
            for col in range(len(source[0])):
                if source[row][col] != background:
                    canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)


def fill_rectangular_interiors(source: Grid, parameters: dict[str, Any]) -> Grid:
    outline = int(parameters["outline"])
    fill = int(parameters["fill"])
    canvas = [list(row) for row in source]
    for top, left, size in _bordered_squares(source, outline):
        for row in range(top + 1, top + size - 1):
            for col in range(left + 1, left + size - 1):
                if source[row][col] == outline:
                    canvas[row][col] = fill
    return tuple(tuple(row) for row in canvas)
