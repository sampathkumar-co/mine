from __future__ import annotations

from collections import Counter
from typing import Any



FORBIDDEN_OPS = {
    "move_singleton_towards", "edge_project", "recolour",
    "decode_regular_linegrid", "overlay_equal_tiles",
    "canonical_rectangular_layers", "fill_internal_blank_axis",
    "extend_corner_marked_rays", "render_concentric",
    "rect_objects", "rect_order",
}


def _ops(value: Any):
    if isinstance(value, dict):
        if "op" in value:
            yield str(value["op"])
        for child in value.values():
            yield from _ops(child)
    elif isinstance(value, list):
        for child in value:
            yield from _ops(child)

class PortableV19Error(RuntimeError):
    pass


def _grid(value: Any):
    result = tuple(tuple(int(cell) for cell in row) for row in value)
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise PortableV19Error("invalid grid")
    return result


def _mode(grid):
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def _eval(expr: Any, grid, state: dict[str, Any]):
    if isinstance(expr, (int, bool)) or expr is None:
        return expr
    if isinstance(expr, str):
        return expr
    if isinstance(expr, list):
        return [_eval(item, grid, state) for item in expr]
    if not isinstance(expr, dict) or "op" not in expr:
        return expr
    code = str(expr["op"])
    if code == "var":
        name = str(expr["name"])
        if name not in state:
            raise PortableV19Error(f"unbound variable: {name}")
        return state[name]
    if code == "height":
        return len(grid)
    if code == "width":
        return len(grid[0])
    if code == "mode":
        state.setdefault("__mode", _mode(grid))
        return state["__mode"]
    if code == "range":
        begin = int(_eval(expr.get("start", 0), grid, state))
        end = int(_eval(expr["stop"], grid, state))
        return list(range(begin, end))
    if code == "grid_coords":
        return [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))]
    if code == "coord_row":
        return int(_eval(expr["value"], grid, state)[0])
    if code == "coord_col":
        return int(_eval(expr["value"], grid, state)[1])
    if code == "sample":
        r = int(_eval(expr["row"], grid, state))
        c = int(_eval(expr["col"], grid, state))
        fallback = int(_eval(expr.get("default", 0), grid, state))
        return grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[0]) else fallback
    if code in {"eq", "ne", "lt", "le", "gt", "ge"}:
        a = _eval(expr["left"], grid, state)
        b = _eval(expr["right"], grid, state)
        return {
            "eq": a == b,
            "ne": a != b,
            "lt": a < b,
            "le": a <= b,
            "gt": a > b,
            "ge": a >= b,
        }[code]
    if code == "and":
        return all(bool(_eval(item, grid, state)) for item in expr["items"])
    if code == "or":
        return any(bool(_eval(item, grid, state)) for item in expr["items"])
    if code == "not":
        return not bool(_eval(expr["value"], grid, state))
    if code == "if":
        chosen = "then" if bool(_eval(expr["condition"], grid, state)) else "else"
        return _eval(expr[chosen], grid, state)
    if code in {"filter", "fold"}:
        items = list(_eval(expr["items"], grid, state))
        name = str(expr["var"])
        existed = name in state
        old = state.get(name)
        values = []
        try:
            for item in items:
                state[name] = item
                result = bool(_eval(expr.get("predicate", expr.get("body")), grid, state))
                if code == "filter":
                    if result:
                        values.append(item)
                else:
                    values.append(result)
        finally:
            if existed:
                state[name] = old
            else:
                state.pop(name, None)
        if code == "filter":
            return values
        reducer = str(expr["reducer"])
        if reducer == "any":
            return any(values)
        if reducer == "all":
            return all(values)
        if reducer == "count":
            return sum(values)
        raise PortableV19Error(f"unknown reducer: {reducer}")
    if code == "unique":
        values = list(_eval(expr["items"], grid, state))
        if len(values) != 1:
            raise PortableV19Error("unique reduction failed")
        return values[0]
    raise PortableV19Error(f"unknown opcode: {code}")


def execute_portable(program: dict[str, Any], value: Any):
    if program.get("schema") != "lexigen-v19-primitive-grid-v1":
        raise PortableV19Error("unknown schema")
    forbidden = sorted(set(_ops(program)) & FORBIDDEN_OPS)
    if forbidden:
        raise PortableV19Error(f"forbidden scene opcodes: {forbidden}")
    grid = _grid(value)
    state: dict[str, Any] = {}
    for binding in program.get("bindings", []):
        state[str(binding["name"])] = _eval(binding["expr"], grid, state)
    rows = int(_eval(program["shape"]["rows"], grid, state))
    cols = int(_eval(program["shape"]["cols"], grid, state))
    if not (1 <= rows <= 60 and 1 <= cols <= 60):
        raise PortableV19Error("shape out of bounds")
    output = []
    for row in range(rows):
        line = []
        for col in range(cols):
            state["row"], state["col"] = row, col
            line.append(int(_eval(program["cell"], grid, state)))
        output.append(tuple(line))
    return tuple(output)


def _expand(value: Any, arguments: dict[str, Any]):
    if isinstance(value, dict):
        if value.get("op") == "param":
            key = str(value["name"])
            if key not in arguments:
                raise PortableV19Error(f"missing argument: {key}")
            return arguments[key]
        return {name: _expand(child, arguments) for name, child in value.items()}
    if isinstance(value, list):
        return [_expand(child, arguments) for child in value]
    return value


def execute_production_portable(
    production: dict[str, Any], arguments: dict[str, Any], value: Any
):
    if production.get("schema") != "lexigen-v19-invented-production-v1":
        raise PortableV19Error("unknown production schema")
    expected = sorted(str(item["name"]) for item in production.get("parameters", []))
    if expected != sorted(arguments):
        raise PortableV19Error("argument mismatch")
    return execute_portable(_expand(production["body"], arguments), value)
