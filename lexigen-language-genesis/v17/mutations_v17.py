from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable

AST = dict[str, Any]
TRANSFORMS = ("identity", "flip_h", "flip_v", "transpose", "rotate_180")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def ast_sha256(ast: AST) -> str:
    return hashlib.sha256(canonical(ast).encode("utf-8")).hexdigest()


def _paths(value: Any, prefix: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _paths(value[key], prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _paths(item, prefix + (index,))


def _replace(ast: AST, path: tuple[Any, ...], replacement: Any) -> AST:
    result = deepcopy(ast)
    cursor: Any = result
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = deepcopy(replacement)
    return result


def _integer_variants(value: int) -> tuple[int, ...]:
    candidates = {(value + 1) % 10, (value + 2) % 10}
    if value > 0:
        candidates.add(value - 1)
    return tuple(sorted(candidate for candidate in candidates if candidate != value))


def _correlated_mode_mutation(ast: AST) -> AST | None:
    text = canonical(ast)
    if '"components"' in text:
        source, target = "components", "colours"
    elif '"colours"' in text:
        source, target = "colours", "components"
    else:
        return None

    def visit(value: Any) -> Any:
        if value == source:
            return target
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(ast)


def generate_mutations(ast: AST, *, limit: int = 64) -> list[AST]:
    variants: dict[str, AST] = {}

    def add(candidate: AST | None) -> None:
        if candidate is None:
            return
        key = canonical(candidate)
        if key != canonical(ast):
            variants[key] = candidate

    add(_correlated_mode_mutation(ast))
    if isinstance(ast, dict) and isinstance(ast.get("grid"), dict):
        add(deepcopy(ast["grid"]))

    for path, value in _paths(ast):
        if not path:
            continue
        key = path[-1]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            for replacement in _integer_variants(value):
                add(_replace(ast, path, replacement))
        elif isinstance(value, str) and key == "name" and value in TRANSFORMS:
            for replacement in TRANSFORMS:
                if replacement != value:
                    add(_replace(ast, path, replacement))
        elif isinstance(value, str) and key == "mode" and value in {"colours", "components"}:
            add(_replace(ast, path, "components" if value == "colours" else "colours"))
        elif isinstance(value, list) and key == "order" and len(value) > 1:
            add(_replace(ast, path, list(reversed(value))))
            add(_replace(ast, path, value[1:] + value[:1]))
            swapped = list(value)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            add(_replace(ast, path, swapped))

    ordered = [variants[key] for key in sorted(variants)]
    return ordered[:limit]


def mutation_manifest(ast: AST, *, limit: int = 64) -> list[dict[str, Any]]:
    return [
        {"mutation_sha256": ast_sha256(candidate), "ast": candidate}
        for candidate in generate_mutations(ast, limit=limit)
    ]


OUTPUT_MUTATIONS = (
    "flip_first_cell",
    "flip_centre_cell",
    "flip_h",
    "flip_v",
    "crop_last_row",
    "crop_last_column",
    "pad_zero_border",
    "rotate_palette",
)


def apply_output_mutation(grid: Any, operator: str):
    values = [list(row) for row in grid]
    height, width = len(values), len(values[0])
    if operator == "flip_first_cell":
        values[0][0] = (values[0][0] + 1) % 10
    elif operator == "flip_centre_cell":
        row, column = height // 2, width // 2
        values[row][column] = (values[row][column] + 1) % 10
    elif operator == "flip_h":
        values = [list(reversed(row)) for row in values]
    elif operator == "flip_v":
        values = list(reversed(values))
    elif operator == "crop_last_row" and height > 1:
        values = values[:-1]
    elif operator == "crop_last_column" and width > 1:
        values = [row[:-1] for row in values]
    elif operator == "pad_zero_border":
        values = [[0] * (width + 2)] + [[0] + row + [0] for row in values] + [[0] * (width + 2)]
    elif operator == "rotate_palette":
        colours = sorted({cell for row in values for cell in row})
        mapping = {colour: colours[(index + 1) % len(colours)] for index, colour in enumerate(colours)}
        if len(colours) == 1:
            mapping[colours[0]] = (colours[0] + 1) % 10
        values = [[mapping[cell] for cell in row] for row in values]
    else:
        return tuple(tuple(row) for row in values)
    return tuple(tuple(row) for row in values)


def mutation_manifest(ast: AST, *, limit: int = 64) -> list[dict[str, Any]]:
    manifest = [
        {
            "kind": "ast",
            "mutation_sha256": ast_sha256(candidate),
            "ast": candidate,
        }
        for candidate in generate_mutations(ast, limit=limit)
    ]
    manifest.extend(
        {
            "kind": "output",
            "operator": operator,
            "mutation_sha256": hashlib.sha256(
                f"lexigen-v17-output:{operator}".encode("utf-8")
            ).hexdigest(),
        }
        for operator in OUTPUT_MUTATIONS
    )
    return manifest
