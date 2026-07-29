from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
if str(V25) not in sys.path:
    sys.path.insert(0, str(V25))

from enumerator_v25 import (
    Expression,
    RuntimeV25Error,
    abstract_literal_colours,
    build_streams,
    canonical,
    initial_expressions,
    make_expression,
    merge_streams,
    semantic_signature,
    sha256_json,
)
from runtime_v25 import Grid

TYPE_ORDER = ("Color", "ObjectSet", "PointSet", "Grid")


def grid_distance(values: tuple[Any, ...], targets: tuple[Grid, ...]) -> tuple[int, int]:
    shape_mismatches = 0
    cell_mismatches = 0
    for value, target in zip(values, targets):
        if len(value) != len(target) or len(value[0]) != len(target[0]):
            shape_mismatches += 1
            cell_mismatches += abs(len(value) * len(value[0]) - len(target) * len(target[0]))
            continue
        cell_mismatches += sum(
            int(left != right)
            for value_row, target_row in zip(value, target)
            for left, right in zip(value_row, target_row)
        )
    return shape_mismatches, cell_mismatches


def _exact_structures(expressions: list[Expression]) -> list[dict[str, Any]]:
    structures: dict[str, dict[str, Any]] = {}
    for expression in sorted(expressions, key=lambda item: item.order_key):
        abstract_ast, arguments = abstract_literal_colours(expression.ast)
        structure_hash = sha256_json(abstract_ast)
        entry = structures.setdefault(structure_hash, {
            "structure_sha256": structure_hash,
            "structure": abstract_ast,
            "minimum_depth": expression.depth,
            "minimum_nodes": expression.nodes,
            "concrete_programs": [],
        })
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
    for entry in structures.values():
        entry["concrete_programs"].sort(key=canonical)
    return [structures[key] for key in sorted(structures)]


def enumerate_programs(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    maximum_depth: int,
    maximum_unique_per_type_per_depth: int,
    maximum_raw_candidates: int,
    retained_total_caps: dict[str, int],
    grid_beam_per_depth: int,
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
    retained_by_type = {type_name: 0 for type_name in TYPE_ORDER}
    raw_candidates = 0
    runtime_invalid = 0
    semantic_duplicates = 0
    ast_duplicates = 0
    beam_dropped = 0
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
            beam_before = beam_dropped
            unique_considered = 0
            retained_layer: list[Expression] = []
            grid_pool: list[tuple[tuple[int, int, int, str], Expression]] = []
            streams = build_streams(store, type_name, depth, sources)
            for candidate in merge_streams(streams):
                if unique_considered >= maximum_unique_per_type_per_depth:
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
                    grid_pool.append((
                        (distance[0], distance[1], expression.nodes, expression.ast_text),
                        expression,
                    ))
                else:
                    remaining = int(retained_total_caps[type_name]) - retained_by_type[type_name]
                    if remaining <= 0:
                        break
                    retained_layer.append(expression)
                    retained_by_type[type_name] += 1

            if type_name == "Grid":
                grid_pool.sort(key=lambda item: item[0])
                remaining = int(retained_total_caps["Grid"]) - retained_by_type["Grid"]
                keep = max(0, min(grid_beam_per_depth, remaining, len(grid_pool)))
                retained_layer = [item[1] for item in grid_pool[:keep]]
                retained_by_type["Grid"] += len(retained_layer)
                beam_dropped += max(0, len(grid_pool) - keep)

            retained_layer.sort(key=lambda expression: expression.order_key)
            store[type_name][depth].extend(retained_layer)
            stats.append({
                "depth": depth,
                "type": type_name,
                "retained_expressions": len(retained_layer),
                "unique_considered": unique_considered,
                "raw_candidates": raw_candidates - raw_before,
                "runtime_invalid": runtime_invalid - invalid_before,
                "semantic_duplicates": semantic_duplicates - duplicate_before,
                "beam_dropped": beam_dropped - beam_before,
            })
            if stop:
                break

    exact_structures = _exact_structures(exact_expressions)
    return {
        "schema": "lexigen-v26-guided-semantic-enumeration-result-v1",
        "nontrivial_task": nontrivial,
        "maximum_depth": maximum_depth,
        "enumeration_complete": exhausted_reason is None,
        "exhausted_reason": exhausted_reason,
        "raw_candidate_evaluations": raw_candidates,
        "runtime_invalid_candidates": runtime_invalid,
        "semantic_duplicates": semantic_duplicates,
        "ast_duplicates": ast_duplicates,
        "beam_dropped_grid_expressions": beam_dropped,
        "total_retained_expressions": sum(retained_by_type.values()),
        "retained_by_type": retained_by_type,
        "exact_concrete_programs": len(exact_expressions),
        "exact_abstract_structures": len(exact_structures),
        "exact_structures": exact_structures,
        "statistics": stats,
    }
