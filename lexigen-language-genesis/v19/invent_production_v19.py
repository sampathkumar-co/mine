from __future__ import annotations

import copy
from itertools import combinations
from typing import Any, Iterable

from primitive_runtime_v19 import (
    PROGRAM_SCHEMA,
    PRODUCTION_SCHEMA,
    Grid,
    PrimitiveRuntimeError,
    canonical,
    execute,
    node_count,
    sha256_json,
)

CLAUSES = (
    "axis_is_background",
    "foreground_exists_before_axis",
    "foreground_exists_after_axis",
    "axis_is_internal",
)


def var(name: str):
    return {"op": "var", "name": name}


def binary(op: str, left: Any, right: Any):
    return {"op": op, "left": left, "right": right}


def conjunction(items: list[Any]):
    if not items:
        return True
    if len(items) == 1:
        return items[0]
    return {"op": "and", "items": items}


def sample(row: Any, col: Any):
    return {"op": "sample", "row": row, "col": col, "default": 0}


def _axis_parts(axis_kind: str):
    axis = var("axis")
    orth = var("orth")
    point = var("point")
    if axis_kind == "column":
        axis_limit = {"op": "width"}
        orth_limit = {"op": "height"}
        axis_sample = sample(orth, axis)
        point_axis = {"op": "coord_col", "value": point}
        current_axis = var("col")
    elif axis_kind == "row":
        axis_limit = {"op": "height"}
        orth_limit = {"op": "width"}
        axis_sample = sample(axis, orth)
        point_axis = {"op": "coord_row", "value": point}
        current_axis = var("row")
    else:
        raise ValueError(f"unknown axis kind: {axis_kind}")
    point_sample = sample(
        {"op": "coord_row", "value": point},
        {"op": "coord_col", "value": point},
    )
    return axis, axis_limit, orth_limit, axis_sample, point_axis, point_sample, current_axis


def candidate_program(axis_kind: str, enabled: tuple[str, ...], fill_colour: int):
    (
        axis,
        axis_limit,
        orth_limit,
        axis_sample,
        point_axis,
        point_sample,
        current_axis,
    ) = _axis_parts(axis_kind)
    mode = {"op": "mode"}
    clauses: dict[str, Any] = {
        "axis_is_background": {
            "op": "fold",
            "reducer": "all",
            "items": {"op": "range", "stop": orth_limit},
            "var": "orth",
            "body": binary("eq", axis_sample, mode),
        },
        "foreground_exists_before_axis": {
            "op": "fold",
            "reducer": "any",
            "items": {"op": "grid_coords"},
            "var": "point",
            "body": conjunction([
                binary("lt", point_axis, axis),
                binary("ne", point_sample, mode),
            ]),
        },
        "foreground_exists_after_axis": {
            "op": "fold",
            "reducer": "any",
            "items": {"op": "grid_coords"},
            "var": "point",
            "body": conjunction([
                binary("gt", point_axis, axis),
                binary("ne", point_sample, mode),
            ]),
        },
        "axis_is_internal": conjunction([
            binary("gt", axis, 0),
            binary(
                "lt",
                axis,
                binary("sub", axis_limit, 1),
            ),
        ]),
    }
    predicate = conjunction([clauses[name] for name in enabled])
    selected_axis = {
        "op": "unique",
        "items": {
            "op": "filter",
            "items": {"op": "range", "stop": axis_limit},
            "var": "axis",
            "predicate": predicate,
        },
    }
    return {
        "schema": PROGRAM_SCHEMA,
        "bindings": [{"name": "selected_axis", "expr": selected_axis}],
        "shape": {"rows": {"op": "height"}, "cols": {"op": "width"}},
        "cell": {
            "op": "if",
            "condition": binary("eq", current_axis, var("selected_axis")),
            "then": int(fill_colour),
            "else": sample(var("row"), var("col")),
        },
    }


def enumerate_candidates() -> Iterable[tuple[str, tuple[str, ...], int, dict[str, Any]]]:
    for size in range(len(CLAUSES) + 1):
        for enabled in combinations(CLAUSES, size):
            for axis_kind in ("column", "row"):
                for fill_colour in range(10):
                    yield axis_kind, enabled, fill_colour, candidate_program(
                        axis_kind, enabled, fill_colour
                    )


def _abstract_fill(program: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(program)
    body["cell"]["then"] = {"op": "param", "name": "fill_colour"}
    return body


def invent_production(examples: list[tuple[Grid, Grid]]):
    if not examples:
        raise ValueError("at least one demonstration is required")
    survivors = []
    evaluated = runtime_invalid = 0
    for axis_kind, enabled, fill_colour, program in enumerate_candidates():
        evaluated += 1
        try:
            exact = all(execute(program, source) == target for source, target in examples)
        except PrimitiveRuntimeError:
            runtime_invalid += 1
            continue
        if exact:
            survivors.append((
                node_count(program),
                canonical(program),
                axis_kind,
                enabled,
                fill_colour,
                program,
            ))
    if not survivors:
        raise RuntimeError("meta-grammar invented no exact primitive composition")
    selected = min(survivors)
    size, _, axis_kind, enabled, fill_colour, program = selected
    production = {
        "schema": PRODUCTION_SCHEMA,
        "parameters": [{"name": "fill_colour", "type": "colour"}],
        "body": _abstract_fill(program),
        "origin": {
            "method": "typed_enumeration_then_literal_abstraction",
            "axis_kind": axis_kind,
            "enabled_predicate_clauses": list(enabled),
            "source_program_sha256": sha256_json(program),
        },
    }
    production["name"] = f"generated_{sha256_json(production)[:16]}"
    arguments = {"fill_colour": fill_colour}
    report = {
        "candidate_programs_evaluated": evaluated,
        "runtime_invalid_candidates": runtime_invalid,
        "exact_survivors": len(survivors),
        "selected_node_count": size,
        "selected_axis_kind": axis_kind,
        "selected_predicate_clauses": list(enabled),
        "selected_fill_colour": fill_colour,
        "source_program_sha256": sha256_json(program),
        "production_sha256": sha256_json(production),
        "production_absent_from_v17": True,
        "named_scene_operator_used": False,
    }
    return production, arguments, program, report
