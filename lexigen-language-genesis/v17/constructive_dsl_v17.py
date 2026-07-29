from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]
Expr = Any
Program = dict[str, Any]

SCHEMA = "lexigen-v17-constructive-grid-v1"
FORBIDDEN_OPS = {
    "move_singleton_towards",
    "edge_project",
    "recolour",
    "decode_regular_linegrid",
    "overlay_equal_tiles",
    "canonical_rectangular_layers",
    "fill_internal_blank_axis",
    "extend_corner_marked_rays",
    "render_concentric",
    "rect_objects",
    "rect_order",
}


class ConstructiveDSLRuntimeError(RuntimeError):
    pass

def as_grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0]:
        raise ConstructiveDSLRuntimeError("empty grid")
    if any(len(row) != len(grid[0]) for row in grid):
        raise ConstructiveDSLRuntimeError("ragged grid")
    return grid


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def palette_mode(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def unique_point(grid: Grid, colour: int) -> tuple[int, int]:
    points = [
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == colour
    ]
    if len(points) != 1:
        raise ConstructiveDSLRuntimeError(
            f"colour {colour} is not a unique point"
        )
    return points[0]


def _sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConstructiveDSLRuntimeError("expected pair")
    return int(value[0]), int(value[1])


def _sample(grid: Grid, row: int, col: int, default: int) -> int:
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        return grid[row][col]
    return int(default)


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
        return env[str(expr["name"])]
    if op == "height":
        return len(grid)
    if op == "width":
        return len(grid[0])
    if op == "mode":
        return env.setdefault("__mode", palette_mode(grid))
    if op == "unique_point":
        colour = int(evaluate(expr["colour"], grid, env))
        key = f"__unique_{colour}"
        return env.setdefault(key, unique_point(grid, colour))
    if op == "pair":
        return (
            int(evaluate(expr["row"], grid, env)),
            int(evaluate(expr["col"], grid, env)),
        )
    if op == "pair_add":
        left = _pair(evaluate(expr["left"], grid, env))
        right = _pair(evaluate(expr["right"], grid, env))
        return left[0] + right[0], left[1] + right[1]
    if op == "pair_sub":
        left = _pair(evaluate(expr["left"], grid, env))
        right = _pair(evaluate(expr["right"], grid, env))
        return left[0] - right[0], left[1] - right[1]
    if op == "pair_sign":
        value = _pair(evaluate(expr["value"], grid, env))
        return _sign(value[0]), _sign(value[1])
    if op in {"add", "sub", "min", "max"}:
        left = int(evaluate(expr["left"], grid, env))
        right = int(evaluate(expr["right"], grid, env))
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "min":
            return min(left, right)
        return max(left, right)
    if op == "clamp":
        value = int(evaluate(expr["value"], grid, env))
        low = int(evaluate(expr["low"], grid, env))
        high = int(evaluate(expr["high"], grid, env))
        return min(max(value, low), high)
    if op == "sample":
        row = int(evaluate(expr["row"], grid, env))
        col = int(evaluate(expr["col"], grid, env))
        default = int(evaluate(expr.get("default", 0), grid, env))
        return _sample(grid, row, col, default)
    if op in {"eq", "lt", "le", "gt", "ge"}:
        left = evaluate(expr["left"], grid, env)
        right = evaluate(expr["right"], grid, env)
        return {"eq": left == right, "lt": left < right,
                "le": left <= right, "gt": left > right,
                "ge": left >= right}[op]
    if op in {"and", "or"}:
        values = [bool(evaluate(item, grid, env)) for item in expr["items"]]
        return all(values) if op == "and" else any(values)
    if op == "not":
        return not bool(evaluate(expr["value"], grid, env))
    if op == "if":
        branch = "then" if bool(evaluate(expr["condition"], grid, env)) else "else"
        return evaluate(expr[branch], grid, env)
    raise ConstructiveDSLRuntimeError(f"unknown expression opcode: {op}")


def walk_ops(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if "op" in value:
            yield str(value["op"])
        for child in value.values():
            yield from walk_ops(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_ops(child)


def assert_program_allowed(program: Program) -> None:
    forbidden = sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
    if forbidden:
        raise ConstructiveDSLRuntimeError(
            f"program contains forbidden scene opcodes: {forbidden}"
        )
    if program.get("schema") != SCHEMA:
        raise ConstructiveDSLRuntimeError("unknown program schema")

def execute(program: Program, value: Any) -> Grid:
    assert_program_allowed(program)
    grid = as_grid(value)
    env: dict[str, Any] = {}
    rows = int(evaluate(program["shape"]["rows"], grid, env))
    cols = int(evaluate(program["shape"]["cols"], grid, env))
    if not (1 <= rows <= 60 and 1 <= cols <= 60):
        raise ConstructiveDSLRuntimeError("constructed shape is out of bounds")
    output = []
    for row in range(rows):
        values = []
        for col in range(cols):
            env["row"], env["col"] = row, col
            values.append(int(evaluate(program["cell"], grid, env)))
        output.append(tuple(values))
    return tuple(output)


def node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(node_count(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(node_count(child) for child in value)
    return 1


def _var(name: str) -> Expr:
    return {"op": "var", "name": name}


def _sample_at(row: Expr, col: Expr, default: Expr = 0) -> Expr:
    return {"op": "sample", "row": row, "col": col, "default": default}

def _binary(op: str, left: Expr, right: Expr) -> Expr:
    return {"op": op, "left": left, "right": right}


def _if(condition: Expr, then: Expr, otherwise: Expr) -> Expr:
    return {"op": "if", "condition": condition, "then": then, "else": otherwise}


def _base_program(rows: Expr, cols: Expr, cell: Expr) -> Program:
    return {
        "schema": SCHEMA,
        "shape": {"rows": rows, "cols": cols},
        "cell": cell,
    }


def local_value_candidates(colours: list[int]) -> Iterable[Program]:
    row, col = _var("row"), _var("col")
    source = _sample_at(row, col)
    for old in colours:
        for new in colours:
            if old == new:
                continue
            yield _base_program(
                {"op": "height"},
                {"op": "width"},
                _if(_binary("eq", source, old), new, source),
            )


def border_coordinate_candidates(colours: list[int]) -> Iterable[Program]:
    row, col = _var("row"), _var("col")
    height, width = {"op": "height"}, {"op": "width"}
    fills: list[Expr] = [{"op": "mode"}, *colours]
    for margin in (1, 2):
        rows = _binary("add", height, 2 * margin)
        cols = _binary("add", width, 2 * margin)
        row_outer = {"op": "or", "items": [
            _binary("lt", row, margin),
            _binary("ge", row, _binary("add", height, margin)),
        ]}
        col_outer = {"op": "or", "items": [
            _binary("lt", col, margin),
            _binary("ge", col, _binary("add", width, margin)),
        ]}
        corner = {"op": "and", "items": [row_outer, col_outer]}
        sample_row = {
            "op": "clamp",
            "value": _binary("sub", row, margin),
            "low": 0,
            "high": _binary("sub", height, 1),
        }
        sample_col = {
            "op": "clamp",
            "value": _binary("sub", col, margin),
            "low": 0,
            "high": _binary("sub", width, 1),
        }
        sampled = _sample_at(sample_row, sample_col)
        for fill in fills:
            yield _base_program(rows, cols, _if(corner, fill, sampled))


def directed_point_candidates(colours: list[int]) -> Iterable[Program]:
    row, col = _var("row"), _var("col")
    here = {"op": "pair", "row": row, "col": col}
    for source_colour in colours:
        for target_colour in colours:
            if source_colour == target_colour:
                continue
            source = {"op": "unique_point", "colour": source_colour}
            target = {"op": "unique_point", "colour": target_colour}
            delta = {
                "op": "pair_sign",
                "value": {"op": "pair_sub", "left": target, "right": source},
            }
            destination = {
                "op": "pair_add",
                "left": source,
                "right": delta,
            }
            unchanged = _sample_at(row, col)
            erased = _if(
                _binary("eq", here, source),
                {"op": "mode"},
                unchanged,
            )
            painted = _if(
                _binary("eq", here, destination),
                source_colour,
                erased,
            )
            yield _base_program(
                {"op": "height"},
                {"op": "width"},
                painted,
            )

def synthesize(examples: list[tuple[Grid, Grid]]) -> tuple[Program, dict[str, Any]]:
    if not examples:
        raise ValueError("at least one demonstration is required")
    colours = sorted({
        cell
        for source, target in examples
        for grid in (source, target)
        for row in grid
        for cell in row
    })
    search_families = (
        ("cell_local", local_value_candidates(colours)),
        ("coordinate_border", border_coordinate_candidates(colours)),
        ("directed_point", directed_point_candidates(colours)),
    )
    survivors: list[tuple[int, str, str, Program]] = []
    evaluated = 0
    runtime_rejections = 0
    for family, candidates in search_families:
        for program in candidates:
            evaluated += 1
            try:
                exact = all(execute(program, source) == target for source, target in examples)
            except ConstructiveDSLRuntimeError:
                runtime_rejections += 1
                continue
            if exact:
                survivors.append(
                    (node_count(program), canonical(program), family, program)
                )
    if not survivors:
        raise RuntimeError("constructive grammar found no exact program")
    size, _, family, program = min(survivors)
    report = {
        "candidate_programs_evaluated": evaluated,
        "runtime_rejections": runtime_rejections,
        "exact_survivors": len(survivors),
        "selected_search_family": family,
        "selected_node_count": size,
        "program_sha256": sha256_json(program),
        "grammar_schema": SCHEMA,
        "forbidden_opcode_hits": sorted(set(walk_ops(program)) & FORBIDDEN_OPS),
    }
    return program, report
