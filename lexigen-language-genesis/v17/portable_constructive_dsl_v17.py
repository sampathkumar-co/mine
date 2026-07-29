from __future__ import annotations

from collections import Counter
from typing import Any


class PortableConstructiveError(RuntimeError):
    pass


def _grid(value: Any):
    result = tuple(tuple(int(cell) for cell in row) for row in value)
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise PortableConstructiveError("invalid grid")
    return result


def _mode(grid):
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def _unique(grid, colour: int):
    points = [
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == colour
    ]
    if len(points) != 1:
        raise PortableConstructiveError("unique-point reduction failed")
    return points[0]


def _pair(value: Any):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PortableConstructiveError("expected pair")
    return int(value[0]), int(value[1])


def _eval(expr: Any, grid, state: dict[str, Any]):
    if isinstance(expr, (int, bool)) or expr is None:
        return expr
    if isinstance(expr, str):
        return expr
    if isinstance(expr, list):
        return [_eval(item, grid, state) for item in expr]
    if not isinstance(expr, dict) or "op" not in expr:
        return expr
    opcode = str(expr["op"])
    if opcode == "var":
        return state[str(expr["name"])]
    if opcode == "height":
        return len(grid)
    if opcode == "width":
        return len(grid[0])
    if opcode == "mode":
        if "mode" not in state:
            state["mode"] = _mode(grid)
        return state["mode"]
    if opcode == "unique_point":
        colour = int(_eval(expr["colour"], grid, state))
        key = f"point:{colour}"
        if key not in state:
            state[key] = _unique(grid, colour)
        return state[key]
    if opcode == "pair":
        return (
            int(_eval(expr["row"], grid, state)),
            int(_eval(expr["col"], grid, state)),
        )
    if opcode in {"pair_add", "pair_sub"}:
        left = _pair(_eval(expr["left"], grid, state))
        right = _pair(_eval(expr["right"], grid, state))
        factor = 1 if opcode == "pair_add" else -1
        return left[0] + factor * right[0], left[1] + factor * right[1]
    if opcode == "pair_sign":
        row, col = _pair(_eval(expr["value"], grid, state))
        sign = lambda value: 1 if value > 0 else -1 if value < 0 else 0
        return sign(row), sign(col)
    if opcode in {"add", "sub", "min", "max"}:
        left = int(_eval(expr["left"], grid, state))
        right = int(_eval(expr["right"], grid, state))
        if opcode == "add":
            return left + right
        if opcode == "sub":
            return left - right
        if opcode == "min":
            return min(left, right)
        return max(left, right)
    if opcode == "clamp":
        value = int(_eval(expr["value"], grid, state))
        low = int(_eval(expr["low"], grid, state))
        high = int(_eval(expr["high"], grid, state))
        return low if value < low else high if value > high else value
    if opcode == "sample":
        row = int(_eval(expr["row"], grid, state))
        col = int(_eval(expr["col"], grid, state))
        default = int(_eval(expr.get("default", 0), grid, state))
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return grid[row][col]
        return default
    if opcode in {"eq", "lt", "le", "gt", "ge"}:
        left = _eval(expr["left"], grid, state)
        right = _eval(expr["right"], grid, state)
        if opcode == "eq":
            return left == right
        if opcode == "lt":
            return left < right
        if opcode == "le":
            return left <= right
        if opcode == "gt":
            return left > right
        return left >= right
    if opcode == "and":
        return all(bool(_eval(item, grid, state)) for item in expr["items"])
    if opcode == "or":
        return any(bool(_eval(item, grid, state)) for item in expr["items"])
    if opcode == "not":
        return not bool(_eval(expr["value"], grid, state))
    if opcode == "if":
        key = "then" if bool(_eval(expr["condition"], grid, state)) else "else"
        return _eval(expr[key], grid, state)
    raise PortableConstructiveError(f"unknown opcode: {opcode}")


def execute_portable(program: dict[str, Any], value: Any):
    if program.get("schema") != "lexigen-v17-constructive-grid-v1":
        raise PortableConstructiveError("unknown schema")
    grid = _grid(value)
    state: dict[str, Any] = {}
    rows = int(_eval(program["shape"]["rows"], grid, state))
    cols = int(_eval(program["shape"]["cols"], grid, state))
    if not (1 <= rows <= 60 and 1 <= cols <= 60):
        raise PortableConstructiveError("shape out of bounds")
    output = []
    for row in range(rows):
        line = []
        for col in range(cols):
            state["row"] = row
            state["col"] = col
            line.append(int(_eval(program["cell"], grid, state)))
        output.append(tuple(line))
    return tuple(output)
