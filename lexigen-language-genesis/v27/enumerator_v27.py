from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
V26 = HERE.parent / "v26"
for folder in (HERE, V25, V26):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from enumerator_v25 import (
    Expression,
    RuntimeV25Error,
    build_streams,
    initial_expressions,
    make_expression,
    merge_streams,
    semantic_signature,
)
from enumerator_v26 import _exact_structures, grid_distance
from runtime_v25 import Grid, ObjectSet, PointSet, colour_points, non_background_points

TYPE_ORDER = ("Color", "ObjectSet", "PointSet", "Grid")


def _changed_points(source: Grid, target: Grid) -> PointSet:
    if len(source) != len(target) or len(source[0]) != len(target[0]):
        return frozenset()
    return frozenset(
        (row, col)
        for row, (source_row, target_row) in enumerate(zip(source, target))
        for col, (left, right) in enumerate(zip(source_row, target_row))
        if left != right
    )


def target_anchors(examples: Sequence[tuple[Grid, Grid]]) -> tuple[tuple[PointSet, ...], ...]:
    result: list[tuple[PointSet, ...]] = []
    for source, target in examples:
        anchors: set[PointSet] = {
            _changed_points(source, target),
            non_background_points(source),
            non_background_points(target),
        }
        for colour in sorted({cell for row in target for cell in row}):
            points = colour_points(target, colour)
            if points:
                anchors.add(points)
        anchors.discard(frozenset())
        result.append(tuple(sorted(anchors, key=lambda item: (len(item), tuple(sorted(item))))))
    return tuple(result)


def support_distance(
    type_name: str,
    values: tuple[Any, ...],
    anchors: tuple[tuple[PointSet, ...], ...],
) -> int:
    total = 0
    for value, choices in zip(values, anchors):
        points = value
        if type_name == "ObjectSet":
            points = frozenset(point for obj in value for point in obj)
        total += min(len(points.symmetric_difference(choice)) for choice in choices)
    return total


def enumerate_programs(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    maximum_depth: int,
    maximum_unique_considered_per_type_per_depth: int,
    maximum_raw_candidates: int,
    retained_total_caps: dict[str, int],
    beam_per_depth: dict[str, int],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one example is required")
    sources = tuple(source for source, _ in examples)
    targets = tuple(target for _, target in examples)
    anchors = target_anchors(examples)
    nontrivial = any(source != target for source, target in examples)

    store: dict[str, list[list[Expression]]] = {
        type_name: [[] for _ in range(maximum_depth + 1)]
        for type_name in TYPE_ORDER
    }
    seen: dict[str, set[tuple[Any, ...]]] = {
        type_name: set() for type_name in TYPE_ORDER
    }
    retained_by_type = {type_name: 0 for type_name in TYPE_ORDER}
    raw_candidates = runtime_invalid = semantic_duplicates = ast_duplicates = 0
    beam_dropped = {type_name: 0 for type_name in TYPE_ORDER}
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
            retained_by_type[type_name] += 1
            accepted += 1
            if type_name == "Grid" and nontrivial and expression.values == targets:
                exact_expressions.append(expression)
        stats.append({
            "depth": 0,
            "type": type_name,
            "retained_expressions": accepted,
            "unique_considered": accepted,
            "raw_candidates": 0,
            "runtime_invalid": 0,
            "semantic_duplicates": 0,
            "beam_dropped": 0,
        })

    stop = False
    for depth in range(1, maximum_depth + 1):
        if stop:
            break
        for type_name in TYPE_ORDER:
            if type_name == "Color":
                stats.append({
                    "depth": depth,
                    "type": type_name,
                    "retained_expressions": 0,
                    "unique_considered": 0,
                    "raw_candidates": 0,
                    "runtime_invalid": 0,
                    "semantic_duplicates": 0,
                    "beam_dropped": 0,
                })
                continue
            if retained_by_type[type_name] >= int(retained_total_caps[type_name]):
                stats.append({
                    "depth": depth,
                    "type": type_name,
                    "retained_expressions": 0,
                    "unique_considered": 0,
                    "raw_candidates": 0,
                    "runtime_invalid": 0,
                    "semantic_duplicates": 0,
                    "beam_dropped": 0,
                    "skipped_reason": "retained_type_cap",
                })
                continue

            raw_before = raw_candidates
            invalid_before = runtime_invalid
            duplicate_before = semantic_duplicates
            dropped_before = beam_dropped[type_name]
            unique_considered = 0
            pool: list[tuple[tuple[Any, ...], Expression]] = []
            streams = build_streams(store, type_name, depth, sources)
            for candidate in merge_streams(streams):
                if unique_considered >= maximum_unique_considered_per_type_per_depth:
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
                seen[type_name].add(signature)
                unique_considered += 1
                expression = make_expression(
                    type_name,
                    depth,
                    candidate.ast,
                    values,
                    nodes=candidate.nodes,
                )
                if type_name == "Grid":
                    if nontrivial and values == targets:
                        exact_expressions.append(expression)
                    distance = grid_distance(values, targets)
                    score = (distance[0], distance[1], expression.nodes, expression.ast_text)
                else:
                    distance = support_distance(type_name, values, anchors)
                    score = (distance, expression.nodes, expression.ast_text)
                pool.append((score, expression))

            pool.sort(key=lambda item: item[0])
            remaining = int(retained_total_caps[type_name]) - retained_by_type[type_name]
            keep = max(0, min(int(beam_per_depth[type_name]), remaining, len(pool)))
            retained_layer = [item[1] for item in pool[:keep]]
            retained_layer.sort(key=lambda expression: expression.order_key)
            store[type_name][depth].extend(retained_layer)
            retained_by_type[type_name] += len(retained_layer)
            beam_dropped[type_name] += max(0, len(pool) - keep)

            stats.append({
                "depth": depth,
                "type": type_name,
                "retained_expressions": len(retained_layer),
                "unique_considered": unique_considered,
                "raw_candidates": raw_candidates - raw_before,
                "runtime_invalid": runtime_invalid - invalid_before,
                "semantic_duplicates": semantic_duplicates - duplicate_before,
                "beam_dropped": beam_dropped[type_name] - dropped_before,
            })
            if stop:
                break

    exact_structures = _exact_structures(exact_expressions)
    return {
        "schema": "lexigen-v27-compact-guided-enumeration-result-v1",
        "nontrivial_task": nontrivial,
        "maximum_depth": maximum_depth,
        "enumeration_complete": exhausted_reason is None,
        "exhausted_reason": exhausted_reason,
        "raw_candidate_evaluations": raw_candidates,
        "runtime_invalid_candidates": runtime_invalid,
        "semantic_duplicates": semantic_duplicates,
        "ast_duplicates": ast_duplicates,
        "beam_dropped_by_type": beam_dropped,
        "total_retained_expressions": sum(retained_by_type.values()),
        "retained_by_type": retained_by_type,
        "exact_concrete_programs": len(exact_expressions),
        "exact_abstract_structures": len(exact_structures),
        "exact_structures": exact_structures,
        "statistics": stats,
    }
