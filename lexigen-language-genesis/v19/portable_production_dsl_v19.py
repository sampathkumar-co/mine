from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

SCHEMA = "lexigen-v19-invented-production-v1"


class PortableProductionError(RuntimeError):
    pass


def _grid(value: Any):
    result = tuple(tuple(int(cell) for cell in row) for row in value)
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise PortableProductionError("invalid grid")
    return result


def _components(grid, colour: int):
    h, w = len(grid), len(grid[0])
    visited = set()
    groups = []
    for row in range(h):
        for col in range(w):
            if grid[row][col] != colour or (row, col) in visited:
                continue
            pending = [(row, col)]
            visited.add((row, col))
            group = set()
            while pending:
                r, c = pending.pop()
                group.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and grid[nr][nc] == colour
                        and (nr, nc) not in visited
                    ):
                        visited.add((nr, nc))
                        pending.append((nr, nc))
            groups.append(frozenset(group))
    return tuple(groups)


def _feature(component, name: str) -> int:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if name == "size":
        return len(component)
    if name == "bbox_height":
        return h
    if name == "bbox_width":
        return w
    if name == "bbox_area":
        return h * w
    if name != "holes":
        raise PortableProductionError("unknown feature")
    occupied = {(r - r0, c - c0) for r, c in component}
    empty = {(r, c) for r in range(h) for c in range(w)} - occupied
    visited = set()
    holes = 0
    for start in sorted(empty):
        if start in visited:
            continue
        pending = [start]
        visited.add(start)
        boundary = False
        while pending:
            r, c = pending.pop()
            if r in (0, h - 1) or c in (0, w - 1):
                boundary = True
            for neighbour in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbour in empty and neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        if not boundary:
            holes += 1
    return holes


def _eval(expr, grid, state):
    if isinstance(expr, (int, bool)) or expr is None:
        return expr
    if isinstance(expr, list):
        return [_eval(item, grid, state) for item in expr]
    if not isinstance(expr, dict) or "op" not in expr:
        return expr
    op = str(expr["op"])
    if op == "input":
        return grid
    if op == "item":
        return state["item"]
    if op == "components":
        return _components(grid, int(_eval(expr["colour"], grid, state)))
    if op == "feature":
        return _feature(frozenset(_eval(expr["component"], grid, state)), str(expr["name"]))
    if op in {"eq", "ge", "gt", "le", "lt"}:
        left = _eval(expr["left"], grid, state)
        right = _eval(expr["right"], grid, state)
        if op == "eq":
            return left == right
        if op == "ge":
            return left >= right
        if op == "gt":
            return left > right
        if op == "le":
            return left <= right
        return left < right
    if op == "filter":
        result = []
        for item in _eval(expr["items"], grid, state):
            nested = dict(state)
            nested["item"] = item
            if bool(_eval(expr["predicate"], grid, nested)):
                result.append(item)
        return tuple(result)
    if op == "union_all":
        result = set()
        for group in _eval(expr["sets"], grid, state):
            result.update(group)
        return frozenset(result)
    if op == "paint":
        source = _grid(_eval(expr["grid"], grid, state))
        coords = frozenset(_eval(expr["coords"], grid, state))
        colour = int(_eval(expr["colour"], grid, state))
        return tuple(
            tuple(colour if (r, c) in coords else value for c, value in enumerate(row))
            for r, row in enumerate(source)
        )
    raise PortableProductionError(f"unknown opcode: {op}")


def execute_portable(production: dict[str, Any], value: Any):
    if production.get("schema") != SCHEMA:
        raise PortableProductionError("unknown schema")
    grid = _grid(value)
    return _grid(_eval(production["body"], grid, {}))


def sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
