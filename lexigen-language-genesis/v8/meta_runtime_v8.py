from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]
Edge = tuple[Point, Point]


class MetaRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Component:
    points: frozenset[Point]
    edges: tuple[Edge, ...]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise MetaRuntimeError("grid must be a non-empty rectangle")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def eval_expr(expr: dict[str, Any], env: dict[str, Any]) -> Any:
    op = expr.get("op")
    if op == "const":
        return expr["value"]
    if op == "var":
        name = str(expr["name"])
        if name not in env:
            raise MetaRuntimeError(f"unknown variable: {name}")
        return env[name]
    if op == "abs":
        return abs(int(eval_expr(expr["arg"], env)))
    if op == "sign":
        return _sign(int(eval_expr(expr["arg"], env)))
    if op in {"add", "sub", "mul"}:
        left = int(eval_expr(expr["left"], env))
        right = int(eval_expr(expr["right"], env))
        return {"add": left + right, "sub": left - right, "mul": left * right}[op]
    if op in {"eq", "lt", "gt"}:
        left = eval_expr(expr["left"], env)
        right = eval_expr(expr["right"], env)
        return {"eq": left == right, "lt": left < right, "gt": left > right}[op]
    if op in {"and", "or"}:
        values = [bool(eval_expr(arg, env)) for arg in expr["args"]]
        return all(values) if op == "and" else any(values)
    if op == "cardinality":
        target = str(expr["target"])
        value = env.get(target)
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise MetaRuntimeError(f"cardinality target is not finite: {target}")
        return len(value)
    if op == "fold_sum":
        collection_name = str(expr["collection"])
        collection = env.get(collection_name)
        if not isinstance(collection, (tuple, list)):
            raise MetaRuntimeError(f"fold collection is not finite: {collection_name}")
        total = 0
        for item in collection:
            child = dict(env)
            if collection_name == "edges":
                first, second = item
                child.update(
                    {
                        "dr": second[0] - first[0],
                        "dc": second[1] - first[1],
                        "edge_first": first,
                        "edge_second": second,
                    }
                )
            elif collection_name == "components":
                if not isinstance(item, Component):
                    raise MetaRuntimeError("component fold received invalid item")
                child.update({"points": item.points, "edges": item.edges, "component": item})
            total += int(eval_expr(expr["body"], child))
        return total
    raise MetaRuntimeError(f"unsupported expression operation: {op!r}")


def select_points(grid: Grid, colour: int) -> frozenset[Point]:
    return frozenset(
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == colour
    )


def build_edges(points: frozenset[Point], relation: dict[str, Any]) -> tuple[Edge, ...]:
    ordered = sorted(points)
    edges: list[Edge] = []
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            env = {"dr": second[0] - first[0], "dc": second[1] - first[1]}
            if bool(eval_expr(relation, env)):
                edges.append((first, second))
    return tuple(edges)


def connected_components(points: frozenset[Point], edges: tuple[Edge, ...]) -> tuple[Component, ...]:
    adjacency: dict[Point, set[Point]] = {point: set() for point in points}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    unseen = set(points)
    components: list[Component] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        members = {start}
        while queue:
            current = queue.popleft()
            for neighbour in sorted(adjacency[current]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    members.add(neighbour)
                    queue.append(neighbour)
        component_edges = tuple(
            edge for edge in edges if edge[0] in members and edge[1] in members
        )
        components.append(Component(frozenset(members), component_edges))
    return tuple(sorted(components, key=lambda item: min(item.points)))


def execute_extension(extension: dict[str, Any], grid: Grid) -> Grid:
    if extension.get("schema") != "lexigen-meta-grammar-extension-v1":
        raise MetaRuntimeError("unsupported extension schema")
    source_colour = int(extension["source"]["colour"])
    points = select_points(grid, source_colour)
    edges = build_edges(points, extension["relation"])
    components = connected_components(points, edges)
    if not components:
        return grid

    by_class: dict[int, list[Component]] = defaultdict(list)
    for component in components:
        env = {"points": component.points, "edges": component.edges, "component": component}
        class_value = int(eval_expr(extension["component_class"], env))
        by_class[class_value].append(component)
    if len(by_class) != 2:
        raise MetaRuntimeError("v8 extension requires exactly two observed classes")

    scores: dict[int, int] = {}
    for class_value, class_components in by_class.items():
        scores[class_value] = int(
            eval_expr(extension["group_score"], {"components": tuple(class_components)})
        )
    if len(set(scores.values())) != len(scores):
        raise MetaRuntimeError("class scores are tied")
    winner_mode = str(extension["winner"]["mode"])
    winner = (max if winner_mode == "max" else min)(scores, key=scores.get)

    values = [list(row) for row in grid]
    winner_colour = int(extension["render"]["winner_colour"])
    other_colour = int(extension["render"]["other_colour"])
    for class_value, class_components in by_class.items():
        colour = winner_colour if class_value == winner else other_colour
        for component in class_components:
            for row, col in component.points:
                values[row][col] = colour
    return tuple(tuple(row) for row in values)
