from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Coord = tuple[int, int]
CoordSet = frozenset[Coord]
Program = dict[str, Any]
SCHEMA = "lexigen-v19r2-executable-production-v1"
FORBIDDEN_NAMED_OPS = {
    "complete_marker_reflection", "reflect_marker_object", "symmetry_completion",
    "move_singleton_towards", "edge_project", "recolour",
    "decode_regular_linegrid", "overlay_equal_tiles",
    "canonical_rectangular_layers", "fill_internal_blank_axis",
    "extend_corner_marked_rays",
}

class RuntimeV19R2Error(RuntimeError):
    pass

def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise RuntimeV19R2Error("invalid grid")
    return grid

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def walk_ops(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if "op" in value:
            yield str(value["op"])
        for child in value.values():
            yield from walk_ops(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_ops(child)

def node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(node_count(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(node_count(child) for child in value)
    return 1

def _mode(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))

def _pair(value: Any) -> Coord:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeV19R2Error("expected coordinate pair")
    return int(value[0]), int(value[1])

def _bbox(coords: CoordSet) -> dict[str, int]:
    if not coords:
        raise RuntimeV19R2Error("bbox of empty coordinate set")
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return {
        "min_row": min(rows), "max_row": max(rows),
        "min_col": min(cols), "max_col": max(cols),
        "height": max(rows) - min(rows) + 1,
        "width": max(cols) - min(cols) + 1,
    }

def _bind(env: dict[str, Any], name: str, value: Any):
    existed = name in env
    previous = env.get(name)
    env[name] = value
    return existed, previous

def _restore(env: dict[str, Any], name: str, existed: bool, previous: Any) -> None:
    if existed:
        env[name] = previous
    else:
        env.pop(name, None)

def evaluate(expr: Any, grid: Grid, env: dict[str, Any]) -> Any:
    if isinstance(expr, (int, bool)) or expr is None:
        return expr
    if isinstance(expr, str):
        return expr
    if isinstance(expr, list):
        return [evaluate(item, grid, env) for item in expr]
    if not isinstance(expr, dict) or "op" not in expr:
        return expr
    op = str(expr["op"])
    if op == "var":
        name = str(expr["name"])
        if name not in env:
            raise RuntimeV19R2Error(f"unbound variable: {name}")
        return env[name]
    if op == "input":
        return grid
    if op == "height":
        return len(grid)
    if op == "width":
        return len(grid[0])
    if op == "mode":
        env.setdefault("__mode", _mode(grid))
        return env["__mode"]
    if op == "palette":
        return tuple(sorted({cell for row in grid for cell in row}))
    if op == "pair":
        return int(evaluate(expr["row"], grid, env)), int(evaluate(expr["col"], grid, env))
    if op == "first":
        return _pair(evaluate(expr["value"], grid, env))[0]
    if op == "second":
        return _pair(evaluate(expr["value"], grid, env))[1]
    if op == "coords_colour":
        colour = int(evaluate(expr["colour"], grid, env))
        return frozenset((r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour)
    if op == "bbox":
        return _bbox(frozenset(evaluate(expr["coords"], grid, env)))
    if op == "bbox_field":
        box = evaluate(expr["bbox"], grid, env)
        return int(box[str(expr["name"])])
    if op in {"add", "sub", "mul"}:
        left = int(evaluate(expr["left"], grid, env))
        right = int(evaluate(expr["right"], grid, env))
        if op == "add": return left + right
        if op == "sub": return left - right
        return left * right
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = evaluate(expr["left"], grid, env)
        right = evaluate(expr["right"], grid, env)
        return {"eq": left == right, "ne": left != right, "lt": left < right,
                "le": left <= right, "gt": left > right, "ge": left >= right}[op]
    if op == "and":
        return all(bool(evaluate(item, grid, env)) for item in expr["items"])
    if op == "or":
        return any(bool(evaluate(item, grid, env)) for item in expr["items"])
    if op == "not":
        return not bool(evaluate(expr["value"], grid, env))
    if op == "if":
        key = "then" if bool(evaluate(expr["condition"], grid, env)) else "else"
        return evaluate(expr[key], grid, env)
    if op in {"filter", "map"}:
        items = list(evaluate(expr["items"], grid, env))
        name = str(expr["var"])
        result = []
        for item in items:
            existed, previous = _bind(env, name, item)
            try:
                if op == "filter":
                    if bool(evaluate(expr["predicate"], grid, env)):
                        result.append(item)
                else:
                    result.append(evaluate(expr["body"], grid, env))
            finally:
                _restore(env, name, existed, previous)
        return frozenset(result) if op == "map" else tuple(result)
    if op == "unique":
        items = list(evaluate(expr["items"], grid, env))
        if len(items) != 1:
            raise RuntimeV19R2Error(f"unique expected one item, got {len(items)}")
        return items[0]
    if op == "set_union":
        result: set[Coord] = set()
        for item in expr["items"]:
            result.update(_pair(value) for value in evaluate(item, grid, env))
        return frozenset(result)
    if op == "set_intersection":
        sets = [set(_pair(v) for v in evaluate(item, grid, env)) for item in expr["items"]]
        return frozenset(set.intersection(*sets)) if sets else frozenset()
    if op == "set_difference":
        left = set(_pair(v) for v in evaluate(expr["left"], grid, env))
        right = set(_pair(v) for v in evaluate(expr["right"], grid, env))
        return frozenset(left - right)
    if op == "canvas":
        rows = int(evaluate(expr["rows"], grid, env))
        cols = int(evaluate(expr["cols"], grid, env))
        fill = int(evaluate(expr["fill"], grid, env))
        if not (1 <= rows <= 60 and 1 <= cols <= 60):
            raise RuntimeV19R2Error("canvas shape out of bounds")
        return tuple(tuple(fill for _ in range(cols)) for _ in range(rows))
    if op == "paint":
        base = as_grid(evaluate(expr["grid"], grid, env))
        coords = frozenset(_pair(v) for v in evaluate(expr["coords"], grid, env))
        colour = int(evaluate(expr["colour"], grid, env))
        return tuple(tuple(colour if (r, c) in coords else value for c, value in enumerate(row)) for r, row in enumerate(base))
    raise RuntimeV19R2Error(f"unknown opcode: {op}")
def execute(program: Program, value: Any) -> Grid:
    if program.get("schema") != SCHEMA:
        raise RuntimeV19R2Error("unknown program schema")
    forbidden = sorted(set(walk_ops(program)) & FORBIDDEN_NAMED_OPS)
    if forbidden:
        raise RuntimeV19R2Error(f"forbidden named operators: {forbidden}")
    grid = as_grid(value)
    env: dict[str, Any] = {}
    for binding in program.get("bindings", []):
        env[str(binding["name"])] = evaluate(binding["expr"], grid, env)
    return as_grid(evaluate(program["body"], grid, env))
