from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
if str(V25) not in sys.path:
    sys.path.insert(0, str(V25))

from runtime_v25 import canonical, sha256_json

TYPE_ORDER = ("Color", "PointSet", "Grid")
POINT_UNARY = ("bbox_border", "dilate4", "erode4", "holes")


@dataclass(frozen=True, slots=True)
class Expression:
    type_name: str
    depth: int
    nodes: int
    ast_text: str

    @property
    def ast(self) -> dict[str, Any]:
        return json.loads(self.ast_text)

    @property
    def order_key(self) -> tuple[int, int, str]:
        return self.depth, self.nodes, self.ast_text


def make_expression(type_name: str, ast: dict[str, Any], depth: int, nodes: int) -> Expression:
    return Expression(
        type_name=type_name,
        depth=depth,
        nodes=nodes,
        ast_text=canonical(ast),
    )


def add_expression(
    layer: dict[str, Expression],
    expression: Expression,
    maximum_nodes: int,
) -> None:
    if expression.nodes > maximum_nodes:
        return
    existing = layer.get(expression.ast_text)
    if existing is None or expression.order_key < existing.order_key:
        layer[expression.ast_text] = expression


def sorted_layer(store: dict[str, list[dict[str, Expression]]], type_name: str, depth: int) -> list[Expression]:
    if depth < 0 or depth >= len(store[type_name]):
        return []
    return sorted(store[type_name][depth].values(), key=lambda item: item.order_key)


def upto(store: dict[str, list[dict[str, Expression]]], type_name: str, depth: int) -> list[Expression]:
    if depth < 0:
        return []
    values = [
        expression
        for layer in store[type_name][: depth + 1]
        for expression in layer.values()
    ]
    return sorted(values, key=lambda item: item.order_key)


def initial_store(maximum_depth: int) -> dict[str, list[dict[str, Expression]]]:
    store = {
        type_name: [dict() for _ in range(maximum_depth + 1)]
        for type_name in TYPE_ORDER
    }
    color_asts = (
        {"op": "param_color", "name": "c0"},
        {"op": "least_non_background"},
        {"op": "most_non_background"},
    )
    for ast in color_asts:
        expression = make_expression("Color", ast, 0, 1)
        store["Color"][0][expression.ast_text] = expression
    leaves = (
        ("PointSet", {"op": "non_background_points"}),
        ("Grid", {"op": "input_grid"}),
    )
    for type_name, ast in leaves:
        expression = make_expression(type_name, ast, 0, 1)
        store[type_name][0][expression.ast_text] = expression
    return store


def max_depth_partitions(
    store: dict[str, list[dict[str, Expression]]],
    child_types: tuple[str, ...],
    target_depth: int,
) -> Iterable[tuple[list[Expression], ...]]:
    for first_max in range(len(child_types)):
        pools: list[list[Expression]] = []
        for index, type_name in enumerate(child_types):
            depth = target_depth - 1
            if index < first_max:
                pool = upto(store, type_name, depth - 1)
            elif index == first_max:
                pool = sorted_layer(store, type_name, depth)
            else:
                pool = upto(store, type_name, depth)
            if not pool:
                break
            pools.append(pool)
        if len(pools) == len(child_types):
            yield tuple(pools)


def bounded_product(
    pools: tuple[list[Expression], ...],
    maximum_child_nodes: int,
) -> Iterable[tuple[Expression, ...]]:
    buckets: list[dict[int, list[Expression]]] = []
    for pool in pools:
        grouped: dict[int, list[Expression]] = {}
        for expression in pool:
            if expression.nodes <= maximum_child_nodes:
                grouped.setdefault(expression.nodes, []).append(expression)
        buckets.append(grouped)
    node_choices = [sorted(grouped) for grouped in buckets]
    for node_tuple in product(*node_choices):
        if sum(node_tuple) > maximum_child_nodes:
            continue
        expression_pools = [
            buckets[index][nodes]
            for index, nodes in enumerate(node_tuple)
        ]
        yield from product(*expression_pools)


def build_grammar(
    *,
    maximum_depth: int,
    maximum_nodes: int,
    maximum_structural_candidates: int,
) -> dict[str, Any]:
    store = initial_store(maximum_depth)
    colors = sorted_layer(store, "Color", 0)

    for depth in range(1, maximum_depth + 1):
        point_layer = store["PointSet"][depth]
        if depth == 1:
            for color in colors:
                add_expression(
                    point_layer,
                    make_expression(
                        "PointSet",
                        {"op": "points_of_color", "colour": color.ast},
                        depth,
                        1 + color.nodes,
                    ),
                    maximum_nodes,
                )
        for child in sorted_layer(store, "PointSet", depth - 1):
            for op in POINT_UNARY:
                add_expression(
                    point_layer,
                    make_expression(
                        "PointSet",
                        {"op": op, "points": child.ast},
                        depth,
                        1 + child.nodes,
                    ),
                    maximum_nodes,
                )

        grid_layer = store["Grid"][depth]
        if depth == 1:
            for color in colors:
                add_expression(
                    grid_layer,
                    make_expression(
                        "Grid",
                        {"op": "canvas", "colour": color.ast},
                        depth,
                        1 + color.nodes,
                    ),
                    maximum_nodes,
                )
        for pools in max_depth_partitions(store, ("Grid", "PointSet"), depth):
            for grid, points in bounded_product(pools, maximum_nodes - 1):
                add_expression(
                    grid_layer,
                    make_expression(
                        "Grid",
                        {"op": "crop_bbox", "grid": grid.ast, "points": points.ast},
                        depth,
                        1 + grid.nodes + points.nodes,
                    ),
                    maximum_nodes,
                )

        for pools in max_depth_partitions(store, ("Grid", "PointSet", "Color"), depth):
            for grid, points, color in bounded_product(pools, maximum_nodes - 1):
                add_expression(
                    grid_layer,
                    make_expression(
                        "Grid",
                        {
                            "op": "paint",
                            "grid": grid.ast,
                            "points": points.ast,
                            "colour": color.ast,
                        },
                        depth,
                        1 + grid.nodes + points.nodes + color.nodes,
                    ),
                    maximum_nodes,
                )

    candidates = [
        expression
        for depth in range(1, maximum_depth + 1)
        for expression in store["Grid"][depth].values()
    ]
    candidates.sort(key=lambda item: item.order_key)
    truncated = len(candidates) > maximum_structural_candidates
    candidates = candidates[:maximum_structural_candidates]

    document = {
        "schema": "lexigen-v30-source-induced-grammar-v1",
        "maximum_depth": maximum_depth,
        "maximum_nodes": maximum_nodes,
        "maximum_structural_candidates": maximum_structural_candidates,
        "structural_candidate_count_before_cap": len(candidates) if not truncated else None,
        "structural_candidate_count": len(candidates),
        "structural_cap_reached": truncated,
        "support_expression_counts": {
            type_name: sum(len(layer) for layer in store[type_name])
            for type_name in TYPE_ORDER
        },
        "candidate_sha256": sha256_json([item.ast_text for item in candidates]),
        "candidates": [
            {
                "depth": item.depth,
                "nodes": item.nodes,
                "ast_sha256": sha256_json(item.ast),
                "ast": item.ast,
            }
            for item in candidates
        ],
    }
    return document


def parameter_names(ast: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(ast, dict):
        if ast.get("op") == "param_color":
            result.add(str(ast["name"]))
        for value in ast.values():
            result.update(parameter_names(value))
    elif isinstance(ast, list):
        for value in ast:
            result.update(parameter_names(value))
    return result


def write_grammar(precommit_path: Path, output_path: Path) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    settings = precommit["enumeration"]
    document = build_grammar(
        maximum_depth=int(settings["maximum_depth"]),
        maximum_nodes=int(settings["maximum_nodes"]),
        maximum_structural_candidates=int(settings["maximum_structural_candidates"]),
    )
    document["precommit_sha256"] = sha256_json(precommit)
    output_path.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return document


def main() -> None:
    precommit_path = HERE / "V30_PRECOMMIT.json"
    output_path = HERE / "V30_GRAMMAR.json"
    document = write_grammar(precommit_path, output_path)
    print(json.dumps({
        "candidate_sha256": document["candidate_sha256"],
        "structural_candidate_count": document["structural_candidate_count"],
        "structural_cap_reached": document["structural_cap_reached"],
        "support_expression_counts": document["support_expression_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
