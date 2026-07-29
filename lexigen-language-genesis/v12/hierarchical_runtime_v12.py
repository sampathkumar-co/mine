from __future__ import annotations

import itertools
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


class HierarchicalRuntimeError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0]:
        raise HierarchicalRuntimeError("grid must be non-empty")
    if any(len(row) != len(grid[0]) for row in grid):
        raise HierarchicalRuntimeError("grid must be rectangular")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _full_line_indices(grid: Grid, colour: int, axis: str) -> tuple[int, ...]:
    if axis == "row":
        return tuple(i for i, row in enumerate(grid) if all(v == colour for v in row))
    if axis == "col":
        columns = tuple(zip(*grid))
        return tuple(i for i, col in enumerate(columns) if all(v == colour for v in col))
    raise HierarchicalRuntimeError(f"unsupported axis: {axis}")


def _intervals(size: int, separators: Iterable[int]) -> tuple[tuple[int, int], ...]:
    cuts = sorted(set(int(value) for value in separators))
    result: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        if cut > start:
            result.append((start, cut))
        start = cut + 1
    if start < size:
        result.append((start, size))
    return tuple(result)


@dataclass(frozen=True)
class ContainerGrid:
    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]
    separator_colour: int

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.cols)


def partition_by_separator_lines(grid: Grid, separator_colour: int) -> ContainerGrid:
    rows = _intervals(len(grid), _full_line_indices(grid, separator_colour, "row"))
    cols = _intervals(len(grid[0]), _full_line_indices(grid, separator_colour, "col"))
    if not rows or not cols:
        raise HierarchicalRuntimeError("separator partition is empty")
    return ContainerGrid(rows, cols, separator_colour)


def _mode(values: Iterable[int]) -> int:
    items = tuple(values)
    if not items:
        raise HierarchicalRuntimeError("cannot reduce an empty container")
    counts = Counter(items)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def reduce_containers(
    grid: Grid,
    partition: ContainerGrid,
    reducer: str,
    border: int,
    canvas_colour: int,
) -> Grid:
    values: list[list[int]] = []
    for r0, r1 in partition.rows:
        row: list[int] = []
        for c0, c1 in partition.cols:
            cells = [grid[r][c] for r in range(r0, r1) for c in range(c0, c1)]
            if reducer == "mode":
                value = _mode(cells)
            elif reducer == "min":
                value = min(cells)
            elif reducer == "max":
                value = max(cells)
            else:
                raise HierarchicalRuntimeError(f"unsupported reducer: {reducer}")
            row.append(value)
        values.append(row)
    height = len(values) + 2 * border
    width = len(values[0]) + 2 * border
    canvas = [[canvas_colour for _ in range(width)] for _ in range(height)]
    for row, line in enumerate(values):
        for col, value in enumerate(line):
            canvas[row + border][col + border] = value
    return tuple(tuple(line) for line in canvas)


def complete_local_midpoints(
    grid: Grid,
    partition: ContainerGrid,
    cell_background: int,
    require_same_colour: bool,
) -> Grid:
    canvas = [list(row) for row in grid]
    tokens: dict[tuple[int, int, int, int], int] = {}
    for outer_row, (r0, r1) in enumerate(partition.rows):
        for outer_col, (c0, c1) in enumerate(partition.cols):
            for row in range(r0, r1):
                for col in range(c0, c1):
                    colour = grid[row][col]
                    if colour in {cell_background, partition.separator_colour}:
                        continue
                    tokens[(outer_row, outer_col, row - r0, col - c0)] = colour

    inferred: dict[tuple[int, int, int, int], int] = {}
    items = list(tokens.items())
    for (ar, ac, lr, lc), colour in items:
        for (br, bc, lr2, lc2), other in items:
            if (ar, ac) >= (br, bc) or (lr, lc) != (lr2, lc2):
                continue
            if require_same_colour and colour != other:
                continue
            if ar == br and bc - ac == 2:
                inferred.setdefault((ar, ac + 1, lr, lc), colour)
            if ac == bc and br - ar == 2:
                inferred.setdefault((ar + 1, ac, lr, lc), colour)

    for (outer_row, outer_col, local_row, local_col), colour in inferred.items():
        if not (0 <= outer_row < partition.height and 0 <= outer_col < partition.width):
            continue
        r0, r1 = partition.rows[outer_row]
        c0, c1 = partition.cols[outer_col]
        row, col = r0 + local_row, c0 + local_col
        if row < r1 and col < c1 and canvas[row][col] == cell_background:
            canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)


def _nonblank_column_intervals(grid: Grid, background: int) -> tuple[tuple[int, int], ...]:
    columns = tuple(zip(*grid))
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for col, values in enumerate(columns):
        occupied = any(value != background for value in values)
        if occupied and start is None:
            start = col
        elif not occupied and start is not None:
            intervals.append((start, col))
            start = None
    if start is not None:
        intervals.append((start, len(columns)))
    return tuple(intervals)


def _connected(points: set[Point]) -> bool:
    if not points:
        return False
    seen = {next(iter(points))}
    queue = deque(seen)
    while queue:
        row, col = queue.popleft()
        for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if point in points and point not in seen:
                seen.add(point)
                queue.append(point)
    return seen == points


def _segment_colours(
    grid: Grid,
    intervals: tuple[tuple[int, int], ...],
    background: int,
    marker: int,
) -> tuple[int, ...]:
    global_visible = [v for row in grid for v in row if v not in {background, marker}]
    colours: list[int] = []
    for start, stop in intervals:
        visible = [
            grid[row][col]
            for row in range(len(grid))
            for col in range(start, stop)
            if grid[row][col] not in {background, marker}
        ]
        if visible:
            colours.append(_mode(visible))
        elif global_visible and len(set(global_visible)) == 1:
            colours.append(global_visible[0])
        else:
            raise HierarchicalRuntimeError("segment colour is ambiguous")
    return tuple(colours)


def align_marker_segments(
    grid: Grid,
    background: int,
    marker: int,
    rank_mode: str,
) -> Grid:
    intervals = _nonblank_column_intervals(grid, background)
    if len(intervals) < 2:
        raise HierarchicalRuntimeError("marker chain needs at least two segments")
    colours = _segment_colours(grid, intervals, background, marker)
    height = len(grid)
    widths = tuple(stop - start for start, stop in intervals)
    output_width = sum(widths)
    point_sets: list[set[Point]] = []
    for start, stop in intervals:
        points = {
            (row, col - start)
            for row in range(height)
            for col in range(start, stop)
            if grid[row][col] != background
        }
        if not points:
            raise HierarchicalRuntimeError("empty marker segment")
        point_sets.append(points)

    candidates: list[tuple[tuple[Any, ...], Grid]] = []
    offset_options = [range(-(height - 1), height) for _ in intervals[1:]]
    for tail in itertools.product(*offset_options):
        offsets = (0,) + tuple(int(value) for value in tail)
        canvas = [[background for _ in range(output_width)] for _ in range(height)]
        all_points: set[Point] = set()
        boundaries: list[tuple[set[Point], set[Point]]] = []
        cursor = 0
        valid = True
        previous_right: set[Point] = set()
        for index, ((start, stop), width, colour, offset, points) in enumerate(
            zip(intervals, widths, colours, offsets, point_sets)
        ):
            left_markers: set[Point] = set()
            right_markers: set[Point] = set()
            for row, local_col in points:
                out_row = row - offset
                out_col = cursor + local_col
                if not (0 <= out_row < height):
                    valid = False
                    break
                source_value = grid[row][start + local_col]
                value = colour if source_value == marker else source_value
                existing = canvas[out_row][out_col]
                if existing != background and existing != value:
                    valid = False
                    break
                canvas[out_row][out_col] = value
                all_points.add((out_row, out_col))
                if source_value == marker and local_col == 0:
                    left_markers.add((out_row, out_col))
                if source_value == marker and local_col == width - 1:
                    right_markers.add((out_row, out_col))
            if not valid:
                break
            if index:
                boundaries.append((previous_right, left_markers))
            previous_right = right_markers
            cursor += width
        if not valid:
            continue
        if any(
            not any(left[0] == right[0] and right[1] - left[1] == 1 for left in lefts for right in rights)
            for lefts, rights in boundaries
        ):
            continue
        if any(not any(point[1] == col for point in all_points) for col in range(output_width)):
            continue
        if not _connected(all_points):
            continue
        output = tuple(tuple(row) for row in canvas)
        shift_cost = sum(abs(value) for value in offsets)
        if rank_mode == "min_shift":
            rank = (shift_cost, offsets)
        elif rank_mode == "lexicographic_offsets":
            rank = (offsets, shift_cost)
        else:
            raise HierarchicalRuntimeError(f"unsupported alignment rank: {rank_mode}")
        candidates.append((rank, output))
    if not candidates:
        raise HierarchicalRuntimeError("no valid aligned segment interpretation")
    return min(candidates, key=lambda item: item[0])[1]


def execute_program(program: dict[str, Any], grid: Grid) -> Grid:
    if program.get("schema") != "lexigen-hierarchical-scene-v1":
        raise HierarchicalRuntimeError("unsupported hierarchical schema")
    stages = program.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise HierarchicalRuntimeError("hierarchical program requires three stages")
    partition_stage, transform_stage, assemble_stage = stages
    partition_mode = str(partition_stage["mode"])
    transform_mode = str(transform_stage["mode"])
    assemble_mode = str(assemble_stage["mode"])

    if partition_mode == "separator_lines":
        partition = partition_by_separator_lines(grid, int(partition_stage["separator_colour"]))
        if transform_mode == "complete_local_midpoints" and assemble_mode == "preserve_canvas":
            return complete_local_midpoints(
                grid,
                partition,
                int(transform_stage["cell_background"]),
                bool(transform_stage["require_same_colour"]),
            )
        if transform_mode.startswith("reduce_") and assemble_mode == "summary_grid":
            return reduce_containers(
                grid,
                partition,
                transform_mode.removeprefix("reduce_"),
                int(assemble_stage["border"]),
                int(assemble_stage["canvas_colour"]),
            )
    if partition_mode == "marker_gap_chain":
        if transform_mode == "align_local_frames" and assemble_mode == "concatenate_segments":
            return align_marker_segments(
                grid,
                int(partition_stage["background"]),
                int(partition_stage["marker"]),
                str(transform_stage["rank_mode"]),
            )
    raise HierarchicalRuntimeError(
        f"unsupported composition: {partition_mode}/{transform_mode}/{assemble_mode}"
    )
