from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
Vector = tuple[int, int]


class StateMachineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Seed:
    boundary: Point
    inner: Point
    heading: Vector


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise StateMachineError("grid must be a non-empty rectangle")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _add(point: Point, vector: Vector) -> Point:
    return point[0] + vector[0], point[1] + vector[1]


def _sub(left: Point, right: Point) -> Vector:
    return left[0] - right[0], left[1] - right[1]


def _inside(point: Point, height: int, width: int) -> bool:
    return 0 <= point[0] < height and 0 <= point[1] < width


def _neighbours4(point: Point) -> tuple[Point, ...]:
    row, col = point
    return ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1))


def _components(points: set[Point]) -> list[set[Point]]:
    remaining = set(points)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        queue = deque([root])
        component = {root}
        while queue:
            current = queue.popleft()
            for neighbour in _neighbours4(current):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        result.append(component)
    return sorted(result, key=min)


def infer_seed(grid: Grid, trace_colour: int, mode: str) -> Seed:
    height, width = len(grid), len(grid[0])
    trace = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == trace_colour
    }
    components = _components(trace)
    if len(components) != 1 or len(components[0]) != 2:
        raise StateMachineError("v10 expects one adjacent two-cell seed")
    first, second = sorted(components[0])
    if second not in _neighbours4(first):
        raise StateMachineError("seed cells are not four-neighbours")

    def on_boundary(point: Point) -> bool:
        return point[0] in (0, height - 1) or point[1] in (0, width - 1)

    boundary_points = [point for point in (first, second) if on_boundary(point)]
    if len(boundary_points) != 1:
        raise StateMachineError("seed must have exactly one boundary endpoint")
    boundary = boundary_points[0]
    inner = second if boundary == first else first
    away = _sub(inner, boundary)
    heading = away if mode == "away_from_boundary" else (-away[0], -away[1])
    return Seed(boundary, inner, heading)


def execute_machine(machine: dict[str, Any], grid: Grid) -> Grid:
    if machine.get("schema") != "lexigen-sensor-state-machine-v1":
        raise StateMachineError("unsupported v10 state-machine schema")
    colours = machine["colours"]
    background = int(colours["background"])
    obstacle = int(colours["obstacle"])
    trace = int(colours["trace"])
    seed = infer_seed(grid, trace, str(machine["seed"]["direction_mode"]))
    current = _add(seed.inner, seed.heading)
    heading = seed.heading
    height, width = len(grid), len(grid[0])
    canvas = [list(row) for row in grid]
    visited_states: set[tuple[Point, Vector]] = set()
    max_steps = int(machine["execution"]["max_steps_factor"]) * height * width

    for _ in range(max_steps):
        if not _inside(current, height, width):
            return tuple(tuple(row) for row in canvas)
        state = current, heading
        if state in visited_states:
            raise StateMachineError("state machine entered a cycle")
        visited_states.add(state)
        if canvas[current[0]][current[1]] not in {background, trace}:
            raise StateMachineError("machine attempted to paint protected content")
        canvas[current[0]][current[1]] = trace

        forward = _add(current, heading)
        trigger_mode = str(machine["transition"]["trigger"])
        forward_value = grid[forward[0]][forward[1]] if _inside(forward, height, width) else background
        trigger = (
            forward_value == obstacle
            if trigger_mode == "forward_is_obstacle"
            else forward_value != background
        )
        if trigger:
            left: Vector = -heading[1], heading[0]
            right: Vector = heading[1], -heading[0]
            left_point, right_point = _add(current, left), _add(current, right)
            left_obstacle = _inside(left_point, height, width) and grid[left_point[0]][left_point[1]] == obstacle
            right_obstacle = _inside(right_point, height, width) and grid[right_point[0]][right_point[1]] == obstacle
            turn_mode = str(machine["transition"]["turn"])
            if turn_mode == "away_from_lateral_obstacle":
                if left_obstacle == right_obstacle:
                    raise StateMachineError("lateral sensor is ambiguous")
                heading = right if left_obstacle else left
            elif turn_mode == "toward_lateral_obstacle":
                if left_obstacle == right_obstacle:
                    raise StateMachineError("lateral sensor is ambiguous")
                heading = left if left_obstacle else right
            elif turn_mode == "always_left":
                heading = left
            elif turn_mode == "always_right":
                heading = right
            else:
                raise StateMachineError("unsupported turn mode")
        current = _add(current, heading)
    raise StateMachineError("state-machine step budget exhausted")
