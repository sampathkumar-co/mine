from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


class ObjectMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Region:
    colour: int
    points: frozenset[Point]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        rows = [point[0] for point in self.points]
        cols = [point[1] for point in self.points]
        return min(rows), max(rows), min(cols), max(cols)


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ObjectMotionError("grid must be a non-empty rectangle")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def neighbours4(point: Point, height: int, width: int) -> Iterable[Point]:
    row, col = point
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        candidate = row + dr, col + dc
        if 0 <= candidate[0] < height and 0 <= candidate[1] < width:
            yield candidate


def components_for_colour(grid: Grid, colour: int) -> list[Region]:
    height, width = len(grid), len(grid[0])
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] == colour
    }
    regions: list[Region] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        points = {start}
        while queue:
            current = queue.popleft()
            for neighbour in neighbours4(current, height, width):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    points.add(neighbour)
                    queue.append(neighbour)
        regions.append(Region(colour, frozenset(points)))
    return sorted(regions, key=lambda region: min(region.points))


def extract_objects(grid: Grid, background: int, marker_colour: int) -> list[Region]:
    colours = sorted({value for row in grid for value in row} - {background, marker_colour})
    return [region for colour in colours for region in components_for_colour(grid, colour)]


def marker_candidates(
    grid: Grid,
    region: Region,
    marker_colour: int,
    mode: str,
) -> list[Point]:
    height, width = len(grid), len(grid[0])
    r0, r1, c0, c1 = region.bounds
    markers = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] == marker_colour
    }
    if mode == "inside_bbox":
        selected = {point for point in markers if r0 <= point[0] <= r1 and c0 <= point[1] <= c1}
    elif mode == "touches_component":
        selected = {
            point
            for point in markers
            if any(neighbour in region.points for neighbour in neighbours4(point, height, width))
        }
    elif mode == "inside_or_touches":
        selected = {
            point
            for point in markers
            if (r0 <= point[0] <= r1 and c0 <= point[1] <= c1)
            or any(neighbour in region.points for neighbour in neighbours4(point, height, width))
        }
    else:
        raise ObjectMotionError(f"unsupported marker association mode: {mode}")
    return sorted(selected)


def rectangle_points(bounds: tuple[int, int, int, int]) -> frozenset[Point]:
    r0, r1, c0, c1 = bounds
    return frozenset((row, col) for row in range(r0, r1 + 1) for col in range(c0, c1 + 1))


def completed_shape(region: Region, marker: Point, mode: str) -> frozenset[Point]:
    if mode == "bbox_fill":
        return rectangle_points(region.bounds)
    if mode == "component_plus_marker":
        return frozenset(set(region.points) | {marker})
    if mode == "component_only":
        return region.points
    raise ObjectMotionError(f"unsupported completed-shape mode: {mode}")


def axis_delta(marker_value: int, low: int, high: int, mode: str) -> int:
    if marker_value == low:
        outward = -1
    elif marker_value == high:
        outward = 1
    else:
        raise ObjectMotionError("marker is not on the required object extreme")
    if mode == "outward":
        return outward
    if mode == "inward":
        return -outward
    if mode == "zero":
        return 0
    raise ObjectMotionError(f"unsupported displacement mode: {mode}")


def execute_extension(extension: dict[str, Any], grid: Grid) -> Grid:
    if extension.get("schema") != "lexigen-object-motion-extension-v1":
        raise ObjectMotionError("unsupported v9 extension schema")
    scene = extension["scene"]
    background = int(scene["background_colour"])
    marker_colour = int(scene["marker_colour"])
    association = str(extension["association"]["mode"])
    shape_mode = str(extension["shape"]["mode"])
    row_mode = str(extension["displacement"]["row"])
    col_mode = str(extension["displacement"]["col"])
    erase_source = bool(extension["render"]["erase_source"])

    canvas = [list(row) for row in grid]
    render_jobs: list[tuple[int, frozenset[Point], frozenset[Point]]] = []
    used_markers: set[Point] = set()
    for region in extract_objects(grid, background, marker_colour):
        markers = [
            point
            for point in marker_candidates(grid, region, marker_colour, association)
            if point not in used_markers
        ]
        if len(markers) != 1:
            raise ObjectMotionError("each object must resolve to exactly one unused marker")
        marker = markers[0]
        used_markers.add(marker)
        shape = completed_shape(region, marker, shape_mode)
        r0, r1, c0, c1 = region.bounds
        dr = axis_delta(marker[0], r0, r1, row_mode)
        dc = axis_delta(marker[1], c0, c1, col_mode)
        shifted = frozenset((row + dr, col + dc) for row, col in shape)
        height, width = len(grid), len(grid[0])
        if any(not (0 <= row < height and 0 <= col < width) for row, col in shifted):
            raise ObjectMotionError("translated object leaves the grid")
        render_jobs.append((region.colour, frozenset(set(region.points) | {marker}), shifted))

    if len(used_markers) != sum(value == marker_colour for row in grid for value in row):
        raise ObjectMotionError("not all markers were assigned")
    if erase_source:
        for _, source_points, _ in render_jobs:
            for row, col in source_points:
                canvas[row][col] = background
    for colour, _, destination in render_jobs:
        for row, col in destination:
            if canvas[row][col] != background:
                raise ObjectMotionError("translated objects overlap preserved content")
            canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)
