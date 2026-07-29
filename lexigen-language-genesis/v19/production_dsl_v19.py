from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
CoordSet = frozenset[tuple[int, int]]
Production = dict[str, Any]

SCHEMA = "lexigen-v19-invented-production-v1"
FORBIDDEN_NAMED_OPS = {
    "recolour_holey_components",
    "select_holey_components",
    "paint_enclosed_components",
    "move_singleton_towards",
    "edge_project",
    "recolour",
}


class ProductionRuntimeError(RuntimeError):
    pass


def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ProductionRuntimeError("invalid grid")
    return grid


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def mode(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def components(grid: Grid, colour: int) -> tuple[CoordSet, ...]:
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    result: list[CoordSet] = []
    for row in range(height):
        for col in range(width):
            start = (row, col)
            if grid[row][col] != colour or start in seen:
                continue
            stack = [start]
            seen.add(start)
            component: set[tuple[int, int]] = set()
            while stack:
                current = stack.pop()
                component.add(current)
                r, c = current
                for neighbour in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    nr, nc = neighbour
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and grid[nr][nc] == colour
                        and neighbour not in seen
                    ):
                        seen.add(neighbour)
                        stack.append(neighbour)
            result.append(frozenset(component))
    return tuple(result)


def _bounds(component: CoordSet) -> tuple[int, int, int, int]:
    if not component:
        raise ProductionRuntimeError("empty component")
    rows = [row for row, _ in component]
    cols = [col for _, col in component]
    return min(rows), max(rows), min(cols), max(cols)


def component_feature(component: CoordSet, name: str) -> int:
    r0, r1, c0, c1 = _bounds(component)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if name == "size":
        return len(component)
    if name == "bbox_height":
        return height
    if name == "bbox_width":
        return width
    if name == "bbox_area":
        return height * width
    if name != "holes":
        raise ProductionRuntimeError(f"unknown component feature: {name}")

    occupied = {(row - r0, col - c0) for row, col in component}
    complement = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if (row, col) not in occupied
    }
    seen: set[tuple[int, int]] = set()
    holes = 0
    for start in sorted(complement):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        touches_boundary = False
        while stack:
            row, col = stack.pop()
            touches_boundary |= row in (0, height - 1) or col in (0, width - 1)
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in complement and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if not touches_boundary:
            holes += 1
    return holes


def walk_ops(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if "op" in value:
            yield str(value["op"])
        for child in value.values():
            yield from walk_ops(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_ops(child)


def _eval(expr: Any, grid: Grid, env: dict[str, Any]) -> Any:
    if isinstance(expr, (int, bool)) or expr is None:
        return expr
    if isinstance(expr, list):
        return [_eval(item, grid, env) for item in expr]
    if not isinstance(expr, dict) or "op" not in expr:
        return expr
    op = str(expr["op"])
    if op == "input":
        return grid
    if op == "mode":
        return env.setdefault("__mode", mode(grid))
    if op == "item":
        return env["item"]
    if op == "components":
        colour = int(_eval(expr["colour"], grid, env))
        return components(grid, colour)
    if op == "feature":
        component = frozenset(_eval(expr["component"], grid, env))
        return component_feature(component, str(expr["name"]))
    if op in {"eq", "gt", "lt", "ge", "le"}:
        left = _eval(expr["left"], grid, env)
        right = _eval(expr["right"], grid, env)
        return {
            "eq": left == right,
            "gt": left > right,
            "lt": left < right,
            "ge": left >= right,
            "le": left <= right,
        }[op]
    if op == "filter":
        selected = []
        for item in _eval(expr["items"], grid, env):
            nested = dict(env)
            nested["item"] = item
            if bool(_eval(expr["predicate"], grid, nested)):
                selected.append(item)
        return tuple(selected)
    if op == "union_all":
        result: set[tuple[int, int]] = set()
        for item in _eval(expr["sets"], grid, env):
            result.update(item)
        return frozenset(result)
    if op == "paint":
        source = as_grid(_eval(expr["grid"], grid, env))
        coords = frozenset(_eval(expr["coords"], grid, env))
        colour = int(_eval(expr["colour"], grid, env))
        return tuple(
            tuple(colour if (row, col) in coords else value for col, value in enumerate(values))
            for row, values in enumerate(source)
        )
    raise ProductionRuntimeError(f"unknown opcode: {op}")


def execute(production: Production, value: Any) -> Grid:
    if production.get("schema") != SCHEMA:
        raise ProductionRuntimeError("unknown production schema")
    forbidden = sorted(set(walk_ops(production)) & FORBIDDEN_NAMED_OPS)
    if forbidden:
        raise ProductionRuntimeError(f"forbidden named opcodes: {forbidden}")
    grid = as_grid(value)
    output = _eval(production["body"], grid, {})
    return as_grid(output)


FEATURES = ("bbox_area", "bbox_height", "bbox_width", "holes", "size")
COMPARATORS = ("eq", "ge", "gt", "le", "lt")


def _production(source_colour: int, target_colour: int, feature: str, comparator: str, threshold: int) -> Production:
    item = {"op": "item"}
    predicate = {
        "op": comparator,
        "left": {"op": "feature", "name": feature, "component": item},
        "right": threshold,
    }
    return {
        "schema": SCHEMA,
        "parameters": {
            "source_colour": source_colour,
            "target_colour": target_colour,
            "feature": feature,
            "comparator": comparator,
            "threshold": threshold,
        },
        "body": {
            "op": "paint",
            "grid": {"op": "input"},
            "coords": {
                "op": "union_all",
                "sets": {
                    "op": "filter",
                    "items": {"op": "components", "colour": source_colour},
                    "predicate": predicate,
                },
            },
            "colour": target_colour,
        },
    }


def node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(node_count(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(node_count(child) for child in value)
    return 1


def invent(
    examples: list[tuple[Grid, Grid]],
    *,
    ablated_features: frozenset[str] = frozenset(),
) -> tuple[Production, dict[str, Any]]:
    if not examples:
        raise ValueError("at least one demonstration is required")
    colours = sorted({
        cell
        for source, target in examples
        for grid in (source, target)
        for row in grid
        for cell in row
    })
    source_colours = sorted({
        cell for source, _ in examples for row in source for cell in row
    })
    target_colours = sorted({
        cell for _, target in examples for row in target for cell in row
    })
    thresholds: dict[tuple[int, str], set[int]] = {}
    for source_colour in source_colours:
        for feature in FEATURES:
            values = {
                component_feature(component, feature)
                for source, _ in examples
                for component in components(source, source_colour)
            }
            thresholds[(source_colour, feature)] = {0, 1, *values}

    evaluated = 0
    runtime_invalid = 0
    survivors: list[tuple[int, str, Production]] = []
    for source_colour in source_colours:
        for target_colour in target_colours:
            if target_colour == source_colour:
                continue
            for feature in FEATURES:
                if feature in ablated_features:
                    continue
                for comparator in COMPARATORS:
                    for threshold in sorted(thresholds[(source_colour, feature)]):
                        production = _production(
                            source_colour,
                            target_colour,
                            feature,
                            comparator,
                            threshold,
                        )
                        evaluated += 1
                        try:
                            exact = all(
                                execute(production, source) == target
                                for source, target in examples
                            )
                        except ProductionRuntimeError:
                            runtime_invalid += 1
                            continue
                        if exact:
                            survivors.append((node_count(production), canonical(production), production))
    if not survivors:
        raise RuntimeError("production inventor found no exact production")
    _, _, selected = min(survivors)
    return selected, {
        "candidate_productions_evaluated": evaluated,
        "runtime_invalid_candidates": runtime_invalid,
        "exact_survivors": len(survivors),
        "invented_production_sha256": sha256_json(selected),
        "selected_node_count": node_count(selected),
        "selected_parameters": selected["parameters"],
        "ablated_features": sorted(ablated_features),
        "available_features": [feature for feature in FEATURES if feature not in ablated_features],
    }
