from __future__ import annotations

import heapq
import itertools
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

from runtime_v25 import (
    Grid,
    ObjectSet,
    PointSet,
    RuntimeV25Error,
    background,
    bbox_border,
    bbox_fill,
    canvas,
    canonical,
    col_span,
    colour_points,
    components4,
    connect_aligned,
    crop_bbox,
    derived_colour,
    dilate4,
    erode4,
    flip_grid_h,
    flip_grid_v,
    holes,
    non_background_points,
    object_feature,
    objects_to_points,
    outline4,
    paint,
    rotate_grid_180,
    row_span,
    select_objects,
    select_position,
    sha256_json,
    transform_points,
)

TYPE_ORDER = ("Color", "Grid", "ObjectSet", "PointSet")
OBJECT_FEATURES = ("bbox_area", "bbox_height", "bbox_width", "size")
EXTREMA = ("all", "maximum", "minimum")
POSITIONS = ("bottommost", "leftmost", "rightmost", "topmost")
POINT_UNARY = (
    "bbox_border",
    "bbox_fill",
    "bbox_reflect_bottom",
    "bbox_reflect_left",
    "bbox_reflect_right",
    "bbox_reflect_top",
    "col_span",
    "connect_aligned",
    "dilate4",
    "erode4",
    "grid_flip_h_points",
    "grid_flip_v_points",
    "grid_rotate180_points",
    "holes",
    "outline4",
    "row_span",
)
POINT_BINARY = ("difference", "intersection", "union")
GRID_UNARY = ("grid_flip_h", "grid_flip_v", "grid_rotate180")


@dataclass(frozen=True)
class Expression:
    type_name: str
    depth: int
    nodes: int
    ast_text: str
    ast: dict[str, Any]
    values: tuple[Any, ...]

    @property
    def order_key(self) -> tuple[int, str]:
        return self.nodes, self.ast_text


@dataclass(frozen=True)
class Candidate:
    type_name: str
    nodes: int
    ast_text: str
    ast: dict[str, Any]
    children: tuple[Expression, ...]
    constants: tuple[Any, ...]
    evaluator: Callable[[tuple[Expression, ...], tuple[Any, ...]], tuple[Any, ...]]

    @property
    def order_key(self) -> tuple[int, str]:
        return self.nodes, self.ast_text


class ProductStream(Iterator[Candidate]):
    def __init__(
        self,
        type_name: str,
        child_pools: Sequence[Sequence[Expression]],
        constant_choices: Sequence[tuple[Any, ...]],
        builder: Callable[[tuple[Expression, ...], tuple[Any, ...]], dict[str, Any]],
        evaluator: Callable[[tuple[Expression, ...], tuple[Any, ...]], tuple[Any, ...]],
    ) -> None:
        if any(not pool for pool in child_pools) or not constant_choices:
            self.heap: list[tuple[tuple[int, str], tuple[int, ...], Candidate]] = []
            self.seen: set[tuple[int, ...]] = set()
            self.child_pools = child_pools
            self.constant_choices = constant_choices
            self.type_name = type_name
            self.builder = builder
            self.evaluator = evaluator
            return
        self.child_pools = child_pools
        self.constant_choices = constant_choices
        self.type_name = type_name
        self.builder = builder
        self.evaluator = evaluator
        self.heap = []
        self.seen = set()
        self._push((0,) * (len(child_pools) + 1))

    def _candidate(self, indexes: tuple[int, ...]) -> Candidate:
        child_indexes = indexes[:-1]
        constant_index = indexes[-1]
        children = tuple(
            pool[index]
            for pool, index in zip(self.child_pools, child_indexes)
        )
        constants = self.constant_choices[constant_index]
        ast = self.builder(children, constants)
        ast_text = canonical(ast)
        return Candidate(
            type_name=self.type_name,
            nodes=1 + sum(child.nodes for child in children),
            ast_text=ast_text,
            ast=ast,
            children=children,
            constants=constants,
            evaluator=self.evaluator,
        )

    def _push(self, indexes: tuple[int, ...]) -> None:
        if indexes in self.seen:
            return
        for dimension, index in enumerate(indexes[:-1]):
            if index >= len(self.child_pools[dimension]):
                return
        if indexes[-1] >= len(self.constant_choices):
            return
        self.seen.add(indexes)
        candidate = self._candidate(indexes)
        heapq.heappush(self.heap, (candidate.order_key, indexes, candidate))

    def __next__(self) -> Candidate:
        if not self.heap:
            raise StopIteration
        _, indexes, candidate = heapq.heappop(self.heap)
        for dimension in range(len(indexes)):
            neighbour = list(indexes)
            neighbour[dimension] += 1
            self._push(tuple(neighbour))
        return candidate


def expressions_upto(store: dict[str, list[list[Expression]]], type_name: str, depth: int) -> list[Expression]:
    if depth < 0:
        return []
    result = [expr for layer in store[type_name][: depth + 1] for expr in layer]
    return sorted(result, key=lambda expr: expr.order_key)


def expression_layer(store: dict[str, list[list[Expression]]], type_name: str, depth: int) -> list[Expression]:
    if depth < 0 or depth >= len(store[type_name]):
        return []
    return sorted(store[type_name][depth], key=lambda expr: expr.order_key)


def depth_partition_pools(
    store: dict[str, list[list[Expression]]],
    child_types: tuple[str, ...],
    target_depth: int,
) -> list[tuple[list[Expression], ...]]:
    if not child_types or target_depth <= 0:
        return []
    partitions: list[tuple[list[Expression], ...]] = []
    for first_max in range(len(child_types)):
        pools: list[list[Expression]] = []
        valid = True
        for index, type_name in enumerate(child_types):
            if index < first_max:
                pool = expressions_upto(store, type_name, target_depth - 2)
            elif index == first_max:
                pool = expression_layer(store, type_name, target_depth - 1)
            else:
                pool = expressions_upto(store, type_name, target_depth - 1)
            if not pool:
                valid = False
                break
            pools.append(pool)
        if valid:
            partitions.append(tuple(pools))
    return partitions


def vector_unary(child: Expression, function: Callable[[Any, int], Any]) -> tuple[Any, ...]:
    return tuple(function(value, index) for index, value in enumerate(child.values))


def vector_binary(
    left: Expression,
    right: Expression,
    function: Callable[[Any, Any, int], Any],
) -> tuple[Any, ...]:
    return tuple(
        function(left_value, right_value, index)
        for index, (left_value, right_value) in enumerate(zip(left.values, right.values))
    )


def vector_ternary(
    first: Expression,
    second: Expression,
    third: Expression,
    function: Callable[[Any, Any, Any, int], Any],
) -> tuple[Any, ...]:
    return tuple(
        function(a, b, c, index)
        for index, (a, b, c) in enumerate(zip(first.values, second.values, third.values))
    )


def point_unary_value(op: str, points: PointSet, source: Grid) -> PointSet:
    if op == "bbox_fill":
        return bbox_fill(points)
    if op == "bbox_border":
        return bbox_border(points)
    if op == "row_span":
        return row_span(points)
    if op == "col_span":
        return col_span(points)
    if op == "connect_aligned":
        return connect_aligned(points)
    if op == "holes":
        return holes(points)
    if op == "outline4":
        return outline4(points)
    if op == "dilate4":
        return dilate4(points, len(source), len(source[0]))
    if op == "erode4":
        return erode4(points)
    mapping = {
        "grid_flip_h_points": "grid_flip_h",
        "grid_flip_v_points": "grid_flip_v",
        "grid_rotate180_points": "grid_rotate180",
    }
    if op in mapping or op.startswith("bbox_reflect_"):
        return transform_points(points, mapping.get(op, op), len(source), len(source[0]))
    raise RuntimeV25Error(f"unknown point unary: {op}")


def build_streams(
    store: dict[str, list[list[Expression]]],
    output_type: str,
    depth: int,
    sources: tuple[Grid, ...],
) -> list[ProductStream]:
    streams: list[ProductStream] = []

    def add(
        child_types: tuple[str, ...],
        constants: Sequence[tuple[Any, ...]],
        builder: Callable[[tuple[Expression, ...], tuple[Any, ...]], dict[str, Any]],
        evaluator: Callable[[tuple[Expression, ...], tuple[Any, ...]], tuple[Any, ...]],
    ) -> None:
        for pools in depth_partition_pools(store, child_types, depth):
            stream = ProductStream(output_type, pools, constants, builder, evaluator)
            if stream.heap:
                streams.append(stream)

    if output_type == "ObjectSet":
        add(
            ("PointSet",),
            ((),),
            lambda children, constants: {"op": "components4", "points": children[0].ast},
            lambda children, constants: vector_unary(children[0], lambda value, index: components4(frozenset(value))),
        )
        add(
            ("ObjectSet",),
            tuple((feature, extremum) for feature in OBJECT_FEATURES for extremum in EXTREMA),
            lambda children, constants: {
                "op": "select_objects",
                "objects": children[0].ast,
                "feature": constants[0],
                "extremum": constants[1],
            },
            lambda children, constants: vector_unary(
                children[0],
                lambda value, index: select_objects(tuple(value), constants[0], constants[1]),
            ),
        )
        add(
            ("ObjectSet",),
            tuple((direction,) for direction in POSITIONS),
            lambda children, constants: {
                "op": "select_position",
                "objects": children[0].ast,
                "direction": constants[0],
            },
            lambda children, constants: vector_unary(
                children[0],
                lambda value, index: select_position(tuple(value), constants[0]),
            ),
        )

    if output_type == "PointSet":
        add(
            ("Color",),
            ((),),
            lambda children, constants: {"op": "points_of_color", "colour": children[0].ast},
            lambda children, constants: tuple(
                colour_points(source, int(colour))
                for source, colour in zip(sources, children[0].values)
            ),
        )
        add(
            ("ObjectSet",),
            ((),),
            lambda children, constants: {"op": "objects_to_points", "objects": children[0].ast},
            lambda children, constants: vector_unary(
                children[0], lambda value, index: objects_to_points(tuple(value))
            ),
        )
        add(
            ("PointSet",),
            tuple((op,) for op in POINT_UNARY),
            lambda children, constants: {"op": constants[0], "points": children[0].ast},
            lambda children, constants: tuple(
                point_unary_value(constants[0], frozenset(value), sources[index])
                for index, value in enumerate(children[0].values)
            ),
        )
        add(
            ("PointSet", "PointSet"),
            tuple((op,) for op in POINT_BINARY),
            lambda children, constants: {
                "op": constants[0],
                "left": children[0].ast,
                "right": children[1].ast,
            },
            lambda children, constants: vector_binary(
                children[0],
                children[1],
                lambda left, right, index: (
                    frozenset(set(left) | set(right))
                    if constants[0] == "union"
                    else frozenset(set(left) & set(right))
                    if constants[0] == "intersection"
                    else frozenset(set(left) - set(right))
                ),
            ),
        )

    if output_type == "Grid":
        add(
            ("Color",),
            ((),),
            lambda children, constants: {"op": "canvas", "colour": children[0].ast},
            lambda children, constants: tuple(
                canvas(source, int(colour))
                for source, colour in zip(sources, children[0].values)
            ),
        )
        add(
            ("Grid",),
            tuple((op,) for op in GRID_UNARY),
            lambda children, constants: {"op": constants[0], "grid": children[0].ast},
            lambda children, constants: vector_unary(
                children[0],
                lambda value, index: (
                    flip_grid_h(value)
                    if constants[0] == "grid_flip_h"
                    else flip_grid_v(value)
                    if constants[0] == "grid_flip_v"
                    else rotate_grid_180(value)
                ),
            ),
        )
        add(
            ("Grid", "PointSet"),
            ((),),
            lambda children, constants: {
                "op": "crop_bbox",
                "grid": children[0].ast,
                "points": children[1].ast,
            },
            lambda children, constants: vector_binary(
                children[0],
                children[1],
                lambda grid_value, points_value, index: crop_bbox(
                    grid_value, frozenset(points_value)
                ),
            ),
        )
        add(
            ("Grid", "PointSet", "Color"),
            ((),),
            lambda children, constants: {
                "op": "paint",
                "grid": children[0].ast,
                "points": children[1].ast,
                "colour": children[2].ast,
            },
            lambda children, constants: vector_ternary(
                children[0],
                children[1],
                children[2],
                lambda grid_value, points_value, colour_value, index: paint(
                    grid_value,
                    frozenset(points_value),
                    int(colour_value),
                ),
            ),
        )

    return streams


def merge_streams(streams: Sequence[ProductStream]) -> Iterator[Candidate]:
    heap: list[tuple[tuple[int, str], int, Candidate, ProductStream]] = []
    for index, stream in enumerate(streams):
        try:
            candidate = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (candidate.order_key, index, candidate, stream))
    while heap:
        _, index, candidate, stream = heapq.heappop(heap)
        yield candidate
        try:
            following = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (following.order_key, index, following, stream))


def semantic_signature(type_name: str, values: tuple[Any, ...]) -> tuple[Any, ...]:
    if type_name == "Color":
        return tuple(int(value) for value in values)
    if type_name == "Grid":
        return tuple(tuple(tuple(int(cell) for cell in row) for row in value) for value in values)
    if type_name == "PointSet":
        return tuple(tuple(sorted(value)) for value in values)
    if type_name == "ObjectSet":
        return tuple(
            tuple(tuple(sorted(obj)) for obj in value)
            for value in values
        )
    raise RuntimeV25Error(f"unknown type: {type_name}")


def abstract_literal_colours(ast: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    colour_to_name: dict[int, str] = {}

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("op") == "literal_color":
            colour = int(value["value"])
            if colour not in colour_to_name:
                colour_to_name[colour] = f"c{len(colour_to_name)}"
            return {"op": "param_color", "name": colour_to_name[colour]}
        result: dict[str, Any] = {}
        if "op" in value:
            result["op"] = value["op"]
        for key in sorted(key for key in value if key != "op"):
            result[key] = visit(value[key])
        return result

    abstract = visit(ast)
    arguments = {
        name: colour
        for colour, name in sorted(colour_to_name.items(), key=lambda item: item[1])
    }
    return abstract, arguments


def make_expression(
    type_name: str,
    depth: int,
    ast: dict[str, Any],
    values: tuple[Any, ...],
    nodes: int = 1,
) -> Expression:
    return Expression(
        type_name=type_name,
        depth=depth,
        nodes=nodes,
        ast_text=canonical(ast),
        ast=ast,
        values=values,
    )


def initial_expressions(sources: tuple[Grid, ...]) -> dict[str, list[Expression]]:
    result = {type_name: [] for type_name in TYPE_ORDER}
    for colour in range(10):
        result["Color"].append(
            make_expression(
                "Color",
                0,
                {"op": "literal_color", "value": colour},
                tuple(colour for _ in sources),
            )
        )
    result["Color"].append(
        make_expression(
            "Color",
            0,
            {"op": "background"},
            tuple(background(source) for source in sources),
        )
    )
    for mode in ("least_non_background", "most_non_background"):
        try:
            values = tuple(derived_colour(source, mode) for source in sources)
        except RuntimeV25Error:
            continue
        result["Color"].append(make_expression("Color", 0, {"op": mode}, values))
    result["Grid"].append(
        make_expression("Grid", 0, {"op": "input_grid"}, tuple(sources))
    )
    result["PointSet"].append(
        make_expression(
            "PointSet",
            0,
            {"op": "non_background_points"},
            tuple(non_background_points(source) for source in sources),
        )
    )
    for type_name in result:
        result[type_name].sort(key=lambda expr: expr.order_key)
    return result


def enumerate_programs(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    maximum_depth: int,
    maximum_unique_per_type_per_depth: int,
    maximum_total_unique: int,
    maximum_raw_candidates: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one example is required")
    sources = tuple(source for source, _ in examples)
    targets = tuple(target for _, target in examples)
    nontrivial = any(source != target for source, target in examples)

    store: dict[str, list[list[Expression]]] = {
        type_name: [[] for _ in range(maximum_depth + 1)]
        for type_name in TYPE_ORDER
    }
    seen: dict[str, set[tuple[Any, ...]]] = {
        type_name: set() for type_name in TYPE_ORDER
    }
    total_unique = 0
    raw_candidates = 0
    runtime_invalid = 0
    semantic_duplicates = 0
    ast_duplicates = 0
    exact_expressions: list[Expression] = []
    ast_seen: set[str] = set()
    stats: list[dict[str, Any]] = []
    exhausted_reason: str | None = None

    leaves = initial_expressions(sources)
    for type_name in TYPE_ORDER:
        accepted = 0
        for expression in leaves[type_name]:
            signature = semantic_signature(type_name, expression.values)
            if signature in seen[type_name]:
                semantic_duplicates += 1
                continue
            seen[type_name].add(signature)
            store[type_name][0].append(expression)
            total_unique += 1
            accepted += 1
            if type_name == "Grid" and nontrivial and expression.values == targets:
                exact_expressions.append(expression)
        stats.append(
            {
                "depth": 0,
                "type": type_name,
                "unique_expressions": accepted,
                "raw_candidates": 0,
                "runtime_invalid": 0,
                "semantic_duplicates": 0,
            }
        )

    stop = False
    for depth in range(1, maximum_depth + 1):
        if stop:
            break
        for type_name in TYPE_ORDER:
            if type_name == "Color":
                stats.append(
                    {
                        "depth": depth,
                        "type": type_name,
                        "unique_expressions": 0,
                        "raw_candidates": 0,
                        "runtime_invalid": 0,
                        "semantic_duplicates": 0,
                    }
                )
                continue
            accepted = raw_before = invalid_before = duplicate_before = 0
            raw_before = raw_candidates
            invalid_before = runtime_invalid
            duplicate_before = semantic_duplicates
            streams = build_streams(store, type_name, depth, sources)
            for candidate in merge_streams(streams):
                if accepted >= maximum_unique_per_type_per_depth:
                    break
                if total_unique >= maximum_total_unique:
                    exhausted_reason = "maximum_total_unique_expressions"
                    stop = True
                    break
                if raw_candidates >= maximum_raw_candidates:
                    exhausted_reason = "maximum_raw_candidate_evaluations"
                    stop = True
                    break
                if candidate.ast_text in ast_seen:
                    ast_duplicates += 1
                    continue
                ast_seen.add(candidate.ast_text)
                raw_candidates += 1
                try:
                    values = candidate.evaluator(candidate.children, candidate.constants)
                    signature = semantic_signature(type_name, values)
                except (
                    RuntimeV25Error,
                    ValueError,
                    IndexError,
                    KeyError,
                    TypeError,
                    OverflowError,
                ):
                    runtime_invalid += 1
                    continue
                if signature in seen[type_name]:
                    semantic_duplicates += 1
                    continue
                expression = make_expression(
                    type_name,
                    depth,
                    candidate.ast,
                    values,
                    nodes=candidate.nodes,
                )
                seen[type_name].add(signature)
                store[type_name][depth].append(expression)
                total_unique += 1
                accepted += 1
                if type_name == "Grid" and nontrivial and values == targets:
                    exact_expressions.append(expression)
            store[type_name][depth].sort(key=lambda expr: expr.order_key)
            stats.append(
                {
                    "depth": depth,
                    "type": type_name,
                    "unique_expressions": accepted,
                    "raw_candidates": raw_candidates - raw_before,
                    "runtime_invalid": runtime_invalid - invalid_before,
                    "semantic_duplicates": semantic_duplicates - duplicate_before,
                }
            )
            if stop:
                break

    exact_structures: dict[str, dict[str, Any]] = {}
    for expression in sorted(exact_expressions, key=lambda expr: expr.order_key):
        abstract_ast, arguments = abstract_literal_colours(expression.ast)
        structure_hash = sha256_json(abstract_ast)
        entry = exact_structures.setdefault(
            structure_hash,
            {
                "structure_sha256": structure_hash,
                "structure": abstract_ast,
                "minimum_depth": expression.depth,
                "minimum_nodes": expression.nodes,
                "concrete_programs": [],
            },
        )
        entry["minimum_depth"] = min(entry["minimum_depth"], expression.depth)
        entry["minimum_nodes"] = min(entry["minimum_nodes"], expression.nodes)
        concrete = {
            "arguments": arguments,
            "concrete_ast_sha256": sha256_json(expression.ast),
            "depth": expression.depth,
            "nodes": expression.nodes,
        }
        if concrete not in entry["concrete_programs"]:
            entry["concrete_programs"].append(concrete)

    for entry in exact_structures.values():
        entry["concrete_programs"].sort(key=canonical)

    return {
        "schema": "lexigen-v25-semantic-enumeration-result-v1",
        "nontrivial_task": nontrivial,
        "maximum_depth": maximum_depth,
        "enumeration_complete": exhausted_reason is None,
        "exhausted_reason": exhausted_reason,
        "raw_candidate_evaluations": raw_candidates,
        "runtime_invalid_candidates": runtime_invalid,
        "semantic_duplicates": semantic_duplicates,
        "ast_duplicates": ast_duplicates,
        "total_unique_expressions": total_unique,
        "unique_by_type": {
            type_name: sum(len(layer) for layer in store[type_name])
            for type_name in TYPE_ORDER
        },
        "exact_concrete_programs": len(exact_expressions),
        "exact_abstract_structures": len(exact_structures),
        "exact_structures": [
            exact_structures[key] for key in sorted(exact_structures)
        ],
        "statistics": stats,
    }
