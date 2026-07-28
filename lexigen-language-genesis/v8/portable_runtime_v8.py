from __future__ import annotations

from typing import Any, Sequence

PortableGrid = tuple[tuple[int, ...], ...]
PortablePoint = tuple[int, int]


class PortableRuntimeError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> PortableGrid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise PortableRuntimeError("invalid grid")
    return grid


def _calc(node: dict[str, Any], variables: dict[str, Any]) -> Any:
    op = node["op"]
    if op == "const":
        return node["value"]
    if op == "var":
        return variables[node["name"]]
    if op == "abs":
        return abs(int(_calc(node["arg"], variables)))
    if op == "sign":
        value = int(_calc(node["arg"], variables))
        return 1 if value > 0 else -1 if value < 0 else 0
    if op == "add":
        return int(_calc(node["left"], variables)) + int(_calc(node["right"], variables))
    if op == "sub":
        return int(_calc(node["left"], variables)) - int(_calc(node["right"], variables))
    if op == "mul":
        return int(_calc(node["left"], variables)) * int(_calc(node["right"], variables))
    if op == "eq":
        return _calc(node["left"], variables) == _calc(node["right"], variables)
    if op == "lt":
        return _calc(node["left"], variables) < _calc(node["right"], variables)
    if op == "gt":
        return _calc(node["left"], variables) > _calc(node["right"], variables)
    if op == "and":
        return all(bool(_calc(child, variables)) for child in node["args"])
    if op == "or":
        return any(bool(_calc(child, variables)) for child in node["args"])
    if op == "cardinality":
        return len(variables[node["target"]])
    if op == "fold_sum":
        total = 0
        for item in variables[node["collection"]]:
            child = dict(variables)
            if node["collection"] == "edges":
                first, second = item
                child["dr"] = second[0] - first[0]
                child["dc"] = second[1] - first[1]
            else:
                child["points"] = item[0]
                child["edges"] = item[1]
            total += int(_calc(node["body"], child))
        return total
    raise PortableRuntimeError(f"unsupported portable operation: {op}")


def _components(points: set[PortablePoint], relation: dict[str, Any]) -> list[tuple[frozenset[PortablePoint], tuple[tuple[PortablePoint, PortablePoint], ...]]]:
    ordered = sorted(points)
    links: list[tuple[PortablePoint, PortablePoint]] = []
    neighbour_map = {point: set() for point in ordered}
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if bool(_calc(relation, {"dr": second[0] - first[0], "dc": second[1] - first[1]})):
                links.append((first, second))
                neighbour_map[first].add(second)
                neighbour_map[second].add(first)
    remaining = set(ordered)
    result = []
    while remaining:
        root = min(remaining)
        stack = [root]
        remaining.remove(root)
        members = {root}
        while stack:
            current = stack.pop()
            for neighbour in sorted(neighbour_map[current], reverse=True):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    members.add(neighbour)
                    stack.append(neighbour)
        internal = tuple(link for link in links if link[0] in members and link[1] in members)
        result.append((frozenset(members), internal))
    return sorted(result, key=lambda item: min(item[0]))


def execute_portable(extension: dict[str, Any], grid: PortableGrid) -> PortableGrid:
    if extension.get("schema") != "lexigen-meta-grammar-extension-v1":
        raise PortableRuntimeError("unsupported extension")
    source_colour = int(extension["source"]["colour"])
    points = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == source_colour
    }
    components = _components(points, extension["relation"])
    classes: dict[int, list[tuple[frozenset[PortablePoint], tuple[tuple[PortablePoint, PortablePoint], ...]]]] = {}
    for component in components:
        class_value = int(
            _calc(extension["component_class"], {"points": component[0], "edges": component[1]})
        )
        classes.setdefault(class_value, []).append(component)
    if len(classes) != 2:
        raise PortableRuntimeError("portable extension expected two classes")
    scores = {
        class_value: int(_calc(extension["group_score"], {"components": tuple(items)}))
        for class_value, items in classes.items()
    }
    if len(set(scores.values())) != 2:
        raise PortableRuntimeError("portable class score tie")
    winner = (max if extension["winner"]["mode"] == "max" else min)(scores, key=scores.get)
    canvas = [list(row) for row in grid]
    for class_value, items in classes.items():
        colour = int(
            extension["render"]["winner_colour"]
            if class_value == winner
            else extension["render"]["other_colour"]
        )
        for component_points, _ in items:
            for row, col in component_points:
                canvas[row][col] = colour
    return tuple(tuple(row) for row in canvas)
