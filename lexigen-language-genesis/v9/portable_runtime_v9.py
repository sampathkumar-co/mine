from __future__ import annotations

from typing import Any, Sequence

PortableGrid = tuple[tuple[int, ...], ...]
PortablePoint = tuple[int, int]


class PortableMotionError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortableMotionError("invalid grid")
    return grid


def _connected(points: set[PortablePoint]) -> list[set[PortablePoint]]:
    remaining = set(points)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        members = {root}
        while stack:
            row, col = stack.pop()
            for candidate in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
                if candidate in remaining:
                    remaining.remove(candidate)
                    members.add(candidate)
                    stack.append(candidate)
        result.append(members)
    return sorted(result, key=min)


def _axis(marker: int, low: int, high: int, mode: str) -> int:
    if marker not in (low, high):
        raise PortableMotionError("marker is not on an object extreme")
    outward = -1 if marker == low else 1
    if mode == "outward":
        return outward
    if mode == "inward":
        return -outward
    if mode == "zero":
        return 0
    raise PortableMotionError("invalid axis mode")


def execute_portable(extension: dict[str, Any], grid: PortableGrid) -> PortableGrid:
    if extension.get("schema") != "lexigen-object-motion-extension-v1":
        raise PortableMotionError("invalid extension schema")
    background = int(extension["scene"]["background_colour"])
    marker_colour = int(extension["scene"]["marker_colour"])
    marker_points = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == marker_colour
    }
    colours = sorted({value for row in grid for value in row} - {background, marker_colour})
    regions = [
        (colour, component)
        for colour in colours
        for component in _connected(
            {
                (row, col)
                for row, values in enumerate(grid)
                for col, value in enumerate(values)
                if value == colour
            }
        )
    ]
    jobs = []
    assigned: set[PortablePoint] = set()
    height, width = len(grid), len(grid[0])
    for colour, component in regions:
        rows = [point[0] for point in component]
        cols = [point[1] for point in component]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        mode = extension["association"]["mode"]
        candidates = []
        for marker in sorted(marker_points - assigned):
            inside = r0 <= marker[0] <= r1 and c0 <= marker[1] <= c1
            touching = any(
                (marker[0] + dr, marker[1] + dc) in component
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if (mode == "inside_bbox" and inside) or (mode == "touches_component" and touching) or (
                mode == "inside_or_touches" and (inside or touching)
            ):
                candidates.append(marker)
        if len(candidates) != 1:
            raise PortableMotionError("object-marker association is not unique")
        marker = candidates[0]
        assigned.add(marker)
        shape_mode = extension["shape"]["mode"]
        if shape_mode == "bbox_fill":
            shape = {(row, col) for row in range(r0, r1 + 1) for col in range(c0, c1 + 1)}
        elif shape_mode == "component_plus_marker":
            shape = set(component) | {marker}
        elif shape_mode == "component_only":
            shape = set(component)
        else:
            raise PortableMotionError("invalid shape mode")
        dr = _axis(marker[0], r0, r1, extension["displacement"]["row"])
        dc = _axis(marker[1], c0, c1, extension["displacement"]["col"])
        destination = {(row + dr, col + dc) for row, col in shape}
        if any(not (0 <= row < height and 0 <= col < width) for row, col in destination):
            raise PortableMotionError("translation leaves grid")
        jobs.append((colour, set(component) | {marker}, destination))
    if assigned != marker_points:
        raise PortableMotionError("unassigned markers remain")
    canvas = [list(row) for row in grid]
    if extension["render"]["erase_source"]:
        for _, source, _ in jobs:
            for row, col in source:
                canvas[row][col] = background
    for colour, _, destination in jobs:
        for row, col in destination:
            if canvas[row][col] != background:
                raise PortableMotionError("translated objects overlap")
            canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)
