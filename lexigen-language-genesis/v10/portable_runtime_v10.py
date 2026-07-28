from __future__ import annotations

from typing import Any, Sequence

PortableGrid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
Vector = tuple[int, int]


class PortableStateError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortableStateError("invalid grid")
    return grid


def _inside(point: Point, height: int, width: int) -> bool:
    return 0 <= point[0] < height and 0 <= point[1] < width


def execute_portable(machine: dict[str, Any], grid: PortableGrid) -> PortableGrid:
    if machine.get("schema") != "lexigen-sensor-state-machine-v1":
        raise PortableStateError("invalid state-machine schema")
    background = int(machine["colours"]["background"])
    obstacle = int(machine["colours"]["obstacle"])
    trace = int(machine["colours"]["trace"])
    height, width = len(grid), len(grid[0])
    seed = sorted(
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == trace
    )
    if len(seed) != 2 or abs(seed[0][0] - seed[1][0]) + abs(seed[0][1] - seed[1][1]) != 1:
        raise PortableStateError("invalid two-cell seed")

    def boundary(point: Point) -> bool:
        return point[0] in (0, height - 1) or point[1] in (0, width - 1)

    boundary_cells = [point for point in seed if boundary(point)]
    if len(boundary_cells) != 1:
        raise PortableStateError("seed boundary endpoint is not unique")
    outside_endpoint = boundary_cells[0]
    inside_endpoint = seed[1] if seed[0] == outside_endpoint else seed[0]
    heading = (
        inside_endpoint[0] - outside_endpoint[0],
        inside_endpoint[1] - outside_endpoint[1],
    )
    if machine["seed"]["direction_mode"] == "toward_boundary":
        heading = -heading[0], -heading[1]
    current = inside_endpoint[0] + heading[0], inside_endpoint[1] + heading[1]
    canvas = [list(row) for row in grid]
    seen = set()
    budget = int(machine["execution"]["max_steps_factor"]) * height * width
    for _ in range(budget):
        if not _inside(current, height, width):
            return tuple(tuple(row) for row in canvas)
        state = current, heading
        if state in seen:
            raise PortableStateError("cycle detected")
        seen.add(state)
        if canvas[current[0]][current[1]] not in {background, trace}:
            raise PortableStateError("protected cell paint")
        canvas[current[0]][current[1]] = trace
        forward = current[0] + heading[0], current[1] + heading[1]
        forward_value = grid[forward[0]][forward[1]] if _inside(forward, height, width) else background
        trigger = (
            forward_value == obstacle
            if machine["transition"]["trigger"] == "forward_is_obstacle"
            else forward_value != background
        )
        if trigger:
            left = -heading[1], heading[0]
            right = heading[1], -heading[0]
            left_point = current[0] + left[0], current[1] + left[1]
            right_point = current[0] + right[0], current[1] + right[1]
            left_hit = _inside(left_point, height, width) and grid[left_point[0]][left_point[1]] == obstacle
            right_hit = _inside(right_point, height, width) and grid[right_point[0]][right_point[1]] == obstacle
            mode = machine["transition"]["turn"]
            if mode in {"away_from_lateral_obstacle", "toward_lateral_obstacle"}:
                if left_hit == right_hit:
                    raise PortableStateError("lateral sensor ambiguity")
                toward = left if left_hit else right
                away = right if left_hit else left
                heading = away if mode == "away_from_lateral_obstacle" else toward
            elif mode == "always_left":
                heading = left
            elif mode == "always_right":
                heading = right
            else:
                raise PortableStateError("unsupported turn mode")
        current = current[0] + heading[0], current[1] + heading[1]
    raise PortableStateError("step budget exhausted")
