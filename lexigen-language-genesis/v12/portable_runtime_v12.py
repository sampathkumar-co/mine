from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Sequence

PortableGrid = list[list[int]]


class PortableRuntimeError(RuntimeError):
    pass


def as_portable(value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = [[int(cell) for cell in row] for row in value]
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortableRuntimeError("invalid grid")
    return grid


def _cuts(size: int, separators: list[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(set(separators)):
        if cut > start:
            result.append((start, cut))
        start = cut + 1
    if start < size:
        result.append((start, size))
    return result


def _partition(grid: PortableGrid, colour: int):
    row_cuts = [i for i, row in enumerate(grid) if all(value == colour for value in row)]
    col_cuts = [c for c in range(len(grid[0])) if all(row[c] == colour for row in grid)]
    rows, cols = _cuts(len(grid), row_cuts), _cuts(len(grid[0]), col_cuts)
    if not rows or not cols:
        raise PortableRuntimeError("empty partition")
    return rows, cols


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def _reduce(grid: PortableGrid, rows, cols, mode: str, border: int, canvas: int):
    summary: list[list[int]] = []
    for r0, r1 in rows:
        line: list[int] = []
        for c0, c1 in cols:
            cells = [grid[r][c] for r in range(r0, r1) for c in range(c0, c1)]
            if mode == "mode":
                value = _mode(cells)
            elif mode == "min":
                value = min(cells)
            elif mode == "max":
                value = max(cells)
            else:
                raise PortableRuntimeError("unknown reducer")
            line.append(value)
        summary.append(line)
    height = len(summary) + border * 2
    width = len(summary[0]) + border * 2
    output = [[canvas for _ in range(width)] for _ in range(height)]
    for r, line in enumerate(summary):
        for c, value in enumerate(line):
            output[r + border][c + border] = value
    return output


def _complete(grid: PortableGrid, rows, cols, cell_background: int, same: bool):
    output = [row[:] for row in grid]
    tokens: dict[tuple[int, int, int, int], int] = {}
    for outer_r, (r0, r1) in enumerate(rows):
        for outer_c, (c0, c1) in enumerate(cols):
            for r in range(r0, r1):
                for c in range(c0, c1):
                    value = grid[r][c]
                    if value == cell_background:
                        continue
                    tokens[(outer_r, outer_c, r - r0, c - c0)] = value
    inferred: dict[tuple[int, int, int, int], int] = {}
    items = list(tokens.items())
    for left_key, left_colour in items:
        ar, ac, lr, lc = left_key
        for right_key, right_colour in items:
            br, bc, lr2, lc2 = right_key
            if (ar, ac) >= (br, bc) or (lr, lc) != (lr2, lc2):
                continue
            if same and left_colour != right_colour:
                continue
            if ar == br and bc - ac == 2:
                inferred.setdefault((ar, ac + 1, lr, lc), left_colour)
            if ac == bc and br - ar == 2:
                inferred.setdefault((ar + 1, ac, lr, lc), left_colour)
    for (outer_r, outer_c, local_r, local_c), colour in inferred.items():
        if outer_r >= len(rows) or outer_c >= len(cols):
            continue
        r0, r1 = rows[outer_r]
        c0, c1 = cols[outer_c]
        row, col = r0 + local_r, c0 + local_c
        if row < r1 and col < c1 and output[row][col] == cell_background:
            output[row][col] = colour
    return output


def _segments(grid: PortableGrid, background: int):
    result: list[tuple[int, int]] = []
    start = None
    for col in range(len(grid[0])):
        occupied = any(row[col] != background for row in grid)
        if occupied and start is None:
            start = col
        elif not occupied and start is not None:
            result.append((start, col))
            start = None
    if start is not None:
        result.append((start, len(grid[0])))
    return result


def _connected(points: set[tuple[int, int]]) -> bool:
    if not points:
        return False
    pending = [next(iter(points))]
    seen = set(pending)
    while pending:
        row, col = pending.pop()
        for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if point in points and point not in seen:
                seen.add(point)
                pending.append(point)
    return seen == points


def _align(grid: PortableGrid, background: int, marker: int, rank_mode: str):
    intervals = _segments(grid, background)
    if len(intervals) < 2:
        raise PortableRuntimeError("not a segment chain")
    height = len(grid)
    widths = [stop - start for start, stop in intervals]
    colours: list[int] = []
    point_sets: list[set[tuple[int, int]]] = []
    for start, stop in intervals:
        visible = [
            grid[r][c]
            for r in range(height)
            for c in range(start, stop)
            if grid[r][c] not in {background, marker}
        ]
        if not visible:
            raise PortableRuntimeError("ambiguous segment colour")
        colours.append(_mode(visible))
        point_sets.append({
            (r, c - start)
            for r in range(height)
            for c in range(start, stop)
            if grid[r][c] != background
        })
    output_width = sum(widths)
    candidates = []
    choices = [range(-(height - 1), height) for _ in intervals[1:]]
    for tail in itertools.product(*choices):
        offsets = (0,) + tuple(tail)
        output = [[background for _ in range(output_width)] for _ in range(height)]
        all_points: set[tuple[int, int]] = set()
        boundary_pairs = []
        cursor = 0
        previous_right: set[tuple[int, int]] = set()
        valid = True
        for index, ((start, stop), width, colour, offset, points) in enumerate(
            zip(intervals, widths, colours, offsets, point_sets)
        ):
            left: set[tuple[int, int]] = set()
            right: set[tuple[int, int]] = set()
            for row, local_col in points:
                out_row = row - offset
                out_col = cursor + local_col
                if not 0 <= out_row < height:
                    valid = False
                    break
                source = grid[row][start + local_col]
                value = colour if source == marker else source
                if output[out_row][out_col] not in {background, value}:
                    valid = False
                    break
                output[out_row][out_col] = value
                all_points.add((out_row, out_col))
                if source == marker and local_col == 0:
                    left.add((out_row, out_col))
                if source == marker and local_col == width - 1:
                    right.add((out_row, out_col))
            if not valid:
                break
            if index:
                boundary_pairs.append((previous_right, left))
            previous_right = right
            cursor += width
        if not valid:
            continue
        if any(
            not any(a[0] == b[0] and b[1] - a[1] == 1 for a in left for b in right)
            for left, right in boundary_pairs
        ):
            continue
        if any(not any(point[1] == col for point in all_points) for col in range(output_width)):
            continue
        if not _connected(all_points):
            continue
        shift_cost = sum(abs(value) for value in offsets)
        if rank_mode == "min_shift":
            rank = (shift_cost, offsets)
        elif rank_mode == "lexicographic_offsets":
            rank = (offsets, shift_cost)
        else:
            raise PortableRuntimeError("unknown rank")
        candidates.append((rank, output))
    if not candidates:
        raise PortableRuntimeError("no valid alignment")
    return min(candidates, key=lambda item: item[0])[1]


def execute_portable(program: dict[str, Any], value: Sequence[Sequence[int]]):
    grid = as_portable(value)
    if program.get("schema") != "lexigen-hierarchical-scene-v1":
        raise PortableRuntimeError("wrong schema")
    partition, transform, assemble = program["stages"]
    partition_mode = partition["mode"]
    transform_mode = transform["mode"]
    assemble_mode = assemble["mode"]
    if partition_mode == "separator_lines":
        rows, cols = _partition(grid, int(partition["separator_colour"]))
        if transform_mode == "complete_local_midpoints" and assemble_mode == "preserve_canvas":
            return _complete(
                grid,
                rows,
                cols,
                int(transform["cell_background"]),
                bool(transform["require_same_colour"]),
            )
        if transform_mode.startswith("reduce_") and assemble_mode == "summary_grid":
            return _reduce(
                grid,
                rows,
                cols,
                transform_mode.removeprefix("reduce_"),
                int(assemble["border"]),
                int(assemble["canvas_colour"]),
            )
    if partition_mode == "marker_gap_chain":
        if transform_mode == "align_local_frames" and assemble_mode == "concatenate_segments":
            return _align(
                grid,
                int(partition["background"]),
                int(partition["marker"]),
                str(transform["rank_mode"]),
            )
    raise PortableRuntimeError("unsupported program")
