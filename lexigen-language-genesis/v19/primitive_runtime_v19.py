from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Expr = Any
Program = dict[str, Any]

PROGRAM_SCHEMA = "lexigen-v19-primitive-grid-v1"
PRODUCTION_SCHEMA = "lexigen-v19-invented-production-v1"
FORBIDDEN_OPS = {
    "move_singleton_towards", "edge_project", "recolour",
    "decode_regular_linegrid", "overlay_equal_tiles",
    "canonical_rectangular_layers", "fill_internal_blank_axis",
    "extend_corner_marked_rays", "render_concentric",
    "rect_objects", "rect_order",
}


class PrimitiveRuntimeError(RuntimeError):
    pass


def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0]:
        raise PrimitiveRuntimeError("empty grid")
    if any(len(row) != len(grid[0]) for row in grid):
        raise PrimitiveRuntimeError("ragged grid")
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


def _with_binding(env: dict[str, Any], name: str, value: Any):
    marker = object()
    previous = env.get(name, marker)
    env[name] = value
    return marker, previous


def _restore_binding(env: dict[str, Any], name: str, marker: Any, previous: Any) -> None:
    if previous is marker:
        env.pop(name, None)
    else:
        env[name] = previous


def evaluate(expr: Expr, grid: Grid, env: dict[str, Any]) -> Any:
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
            raise PrimitiveRuntimeError(f"unbound variable: {name}")
        return env[name]
    if op == "height":
        return len(grid)
    if op == "width":
        return len(grid[0])
    if op == "mode":
        if "__mode" not in env:
            env["__mode"] = _mode(grid)
        return env["__mode"]
    if op == "range":
        start = int(evaluate(expr.get("start", 0), grid, env))
        stop = int(evaluate(expr["stop"], grid, env))
        return list(range(start, stop))
    if op == "grid_coords":
        return [(row, col) for row in range(len(grid)) for col in range(len(grid[0]))]
    if op == "coord_row":
        return int(evaluate(expr["value"], grid, env)[0])
    if op == "coord_col":
        return int(evaluate(expr["value"], grid, env)[1])
    if op == "sample":
        row = int(evaluate(expr["row"], grid, env))
        col = int(evaluate(expr["col"], grid, env))
        default = int(evaluate(expr.get("default", 0), grid, env))
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return grid[row][col]
        return default
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = evaluate(expr["left"], grid, env)
        right = evaluate(expr["right"], grid, env)
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        return left >= right
    if op == "and":
        return all(bool(evaluate(item, grid, env)) for item in expr["items"])
    if op == "or":
        return any(bool(evaluate(item, grid, env)) for item in expr["items"])
    if op == "not":
        return not bool(evaluate(expr["value"], grid, env))
    if op == "if":
        branch = "then" if bool(evaluate(expr["condition"], grid, env)) else "else"
        return evaluate(expr[branch], grid, env)
    if op == "filter":
        items = list(evaluate(expr["items"], grid, env))
        name = str(expr["var"])
        result = []
        for item in items:
            marker, previous = _with_binding(env, name, item)
            try:
                if bool(evaluate(expr["predicate"], grid, env)):
                    result.append(item)
            finally:
                _restore_binding(env, name, marker, previous)
        return result
    if op == "fold":
        items = list(evaluate(expr["items"], grid, env))
        name = str(expr["var"])
        reducer = str(expr["reducer"])
        values = []
        for item in items:
            marker, previous = _with_binding(env, name, item)
            try:
                values.append(bool(evaluate(expr["body"], grid, env)))
            finally:
                _restore_binding(env, name, marker, previous)
        if reducer == "any":
            return any(values)
        if reducer == "all":
            return all(values)
        if reducer == "count":
            return sum(values)
        raise PrimitiveRuntimeError(f"unknown fold reducer: {reducer}")
    if op == "unique":
        items = list(evaluate(expr["items"], grid, env))
        if len(items) != 1:
            raise PrimitiveRuntimeError(f"unique reduction expected 1 item, got {len(items)}")
        return items[0]
    raise PrimitiveRuntimeError(f"unknown primitive opcode: {op}")


def execute(program: Program, value: Any) -> Grid:
    if program.get("schema") != PROGRAM_SCHEMA:
        raise PrimitiveRuntimeError("unknown program schema")
    forbidden = sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
    if forbidden:
        raise PrimitiveRuntimeError(f"forbidden scene opcodes: {forbidden}")
    grid = as_grid(value)
    env: dict[str, Any] = {}
    for binding in program.get("bindings", []):
        env[str(binding["name"])] = evaluate(binding["expr"], grid, env)
    rows = int(evaluate(program["shape"]["rows"], grid, env))
    cols = int(evaluate(program["shape"]["cols"], grid, env))
    if not (1 <= rows <= 60 and 1 <= cols <= 60):
        raise PrimitiveRuntimeError("output shape out of bounds")
    output = []
    for row in range(rows):
        line = []
        for col in range(cols):
            env["row"], env["col"] = row, col
            line.append(int(evaluate(program["cell"], grid, env)))
        output.append(tuple(line))
    return tuple(output)


def _substitute(value: Any, arguments: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if value.get("op") == "param":
            name = str(value["name"])
            if name not in arguments:
                raise PrimitiveRuntimeError(f"missing production argument: {name}")
            return arguments[name]
        return {key: _substitute(child, arguments) for key, child in value.items()}
    if isinstance(value, list):
        return [_substitute(child, arguments) for child in value]
    return value


def expand_production(production: dict[str, Any], arguments: dict[str, Any]) -> Program:
    if production.get("schema") != PRODUCTION_SCHEMA:
        raise PrimitiveRuntimeError("unknown production schema")
    required = [str(item["name"]) for item in production.get("parameters", [])]
    if sorted(required) != sorted(arguments):
        raise PrimitiveRuntimeError("production arguments do not match parameters")
    program = _substitute(production["body"], arguments)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise PrimitiveRuntimeError("production did not expand to a primitive program")
    return program


def execute_production(
    production: dict[str, Any], arguments: dict[str, Any], value: Any
) -> Grid:
    return execute(expand_production(production, arguments), value)
