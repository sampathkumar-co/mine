from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V30 = HERE.parent / "v30"
V25 = HERE.parent / "v25"
for path in (V30, V25):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scan_one_v30 import assignments, generate_examples, load, sha256_file, write
from runtime_v25 import (
    RuntimeV25Error,
    as_grid,
    background,
    bbox_border,
    canvas,
    colour_points,
    crop_bbox,
    derived_colour,
    dilate4,
    erode4,
    holes,
    non_background_points,
    paint,
    sha256_json,
)


@dataclass(frozen=True, slots=True)
class PreparedNode:
    key: str
    op: str
    node: dict[str, Any]
    children: dict[str, "PreparedNode"]
    depends_on_parameter: bool


@dataclass(frozen=True, slots=True)
class ConcreteCandidate:
    structural_index: int
    candidate: dict[str, Any]
    prepared: PreparedNode
    parameters: dict[str, int]
    parameter_key: int | None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prepare_node(node: dict[str, Any]) -> PreparedNode:
    children = {
        key: prepare_node(value)
        for key, value in node.items()
        if isinstance(value, dict) and "op" in value
    }
    op = str(node["op"])
    depends = op == "param_color" or any(child.depends_on_parameter for child in children.values())
    return PreparedNode(
        key=canonical(node),
        op=op,
        node=node,
        children=children,
        depends_on_parameter=depends,
    )


def evaluate_prepared(
    prepared: PreparedNode,
    grid: tuple[tuple[int, ...], ...],
    parameters: dict[str, int],
    cache: dict[str, Any],
) -> Any:
    if prepared.key in cache:
        cached = cache[prepared.key]
        if isinstance(cached, BaseException):
            raise cached
        return cached
    node = prepared.node
    op = prepared.op
    try:
        if op == "param_color":
            result: Any = int(parameters[str(node["name"])])
        elif op == "background":
            result = background(grid)
        elif op in {"least_non_background", "most_non_background"}:
            result = derived_colour(grid, op)
        elif op == "input_grid":
            result = grid
        elif op == "canvas":
            result = canvas(grid, int(evaluate_prepared(prepared.children["colour"], grid, parameters, cache)))
        elif op == "points_of_color":
            result = colour_points(grid, int(evaluate_prepared(prepared.children["colour"], grid, parameters, cache)))
        elif op == "non_background_points":
            result = non_background_points(grid)
        elif op == "bbox_border":
            result = bbox_border(frozenset(evaluate_prepared(prepared.children["points"], grid, parameters, cache)))
        elif op == "dilate4":
            points = frozenset(evaluate_prepared(prepared.children["points"], grid, parameters, cache))
            result = dilate4(points, len(grid), len(grid[0]))
        elif op == "erode4":
            result = erode4(frozenset(evaluate_prepared(prepared.children["points"], grid, parameters, cache)))
        elif op == "holes":
            result = holes(frozenset(evaluate_prepared(prepared.children["points"], grid, parameters, cache)))
        elif op == "paint":
            result = paint(
                as_grid(evaluate_prepared(prepared.children["grid"], grid, parameters, cache)),
                frozenset(evaluate_prepared(prepared.children["points"], grid, parameters, cache)),
                int(evaluate_prepared(prepared.children["colour"], grid, parameters, cache)),
            )
        elif op == "crop_bbox":
            result = crop_bbox(
                as_grid(evaluate_prepared(prepared.children["grid"], grid, parameters, cache)),
                frozenset(evaluate_prepared(prepared.children["points"], grid, parameters, cache)),
            )
        else:
            raise RuntimeV25Error(f"unknown AST operation: {op}")
    except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        cache[prepared.key] = error
        raise
    cache[prepared.key] = result
    return result


def concrete_candidates(grammar: dict[str, Any], precommit: dict[str, Any]) -> tuple[list[ConcreteCandidate], bool]:
    colors = [int(value) for value in precommit["enumeration"]["literal_color_domain"]]
    maximum = int(precommit["enumeration"]["maximum_concrete_candidates_per_task"])
    result: list[ConcreteCandidate] = []
    cap_reached = False
    for structural_index, candidate in enumerate(grammar["candidates"]):
        prepared = prepare_node(candidate["ast"])
        for parameters in assignments(candidate["ast"], colors):
            if len(result) >= maximum:
                cap_reached = True
                return result, cap_reached
            normalized = {str(key): int(value) for key, value in parameters.items()}
            result.append(ConcreteCandidate(
                structural_index=structural_index,
                candidate=candidate,
                prepared=prepared,
                parameters=normalized,
                parameter_key=normalized.get("c0"),
            ))
    return result, cap_reached


def evaluate_candidates_memoized(
    examples: list[tuple[Any, Any]],
    grammar: dict[str, Any],
    precommit: dict[str, Any],
) -> dict[str, Any]:
    sources = tuple(as_grid(source) for source, _ in examples)
    targets = tuple(as_grid(target) for _, target in examples)
    task_nontrivial = any(source != target for source, target in zip(sources, targets))
    concrete, cap_reached = concrete_candidates(grammar, precommit)
    count = len(concrete)
    invalid = [False] * count
    identity_all = [True] * count
    exact_all = [True] * count
    groups: dict[int | None, list[int]] = {}
    for index, item in enumerate(concrete):
        groups.setdefault(item.parameter_key, []).append(index)

    for source, target in zip(sources, targets):
        for parameter_key in sorted(groups, key=lambda value: (-1 if value is None else value)):
            cache: dict[str, Any] = {}
            for index in groups[parameter_key]:
                if invalid[index]:
                    continue
                item = concrete[index]
                try:
                    output = as_grid(evaluate_prepared(item.prepared, source, item.parameters, cache))
                except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError):
                    invalid[index] = True
                    continue
                identity_all[index] = identity_all[index] and output == source
                exact_all[index] = exact_all[index] and output == target

    exact: list[dict[str, Any]] = []
    for index, item in enumerate(concrete):
        if invalid[index] or not task_nontrivial or not exact_all[index]:
            continue
        candidate = item.candidate
        exact.append({
            "structural_index": item.structural_index,
            "depth": int(candidate["depth"]),
            "nodes": int(candidate["nodes"]),
            "ast_sha256": candidate["ast_sha256"],
            "parameters": item.parameters,
            "concrete_program_sha256": sha256_json({"ast": candidate["ast"], "parameters": item.parameters}),
        })

    runtime_invalid = sum(1 for value in invalid if value)
    identity_candidates = sum(
        1 for index in range(count)
        if not invalid[index] and identity_all[index]
    )
    return {
        "schema": "lexigen-v30-task-enumeration-v1",
        "task_nontrivial": task_nontrivial,
        "concrete_candidates_tested": count,
        "runtime_invalid_candidates": runtime_invalid,
        "identity_candidates_rejected": identity_candidates,
        "exact_candidate_count": len(exact),
        "exact_candidates": exact,
        "selected_candidate": exact[0] if exact else None,
        "candidate_cap_reached": cap_reached,
    }


def validate_recovery_task(task_id: str, recovery: dict[str, Any]) -> None:
    equivalence_id = str(recovery["equivalence_gate"]["task_id"])
    allowed = {str(value) for value in recovery["recovery_task_ids"]}
    if task_id != equivalence_id and task_id not in allowed:
        raise RuntimeError("task identity is outside the frozen recovery set")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recovery_path = HERE / "V30_RECOVERY_PRECOMMIT.json"
    recovery = load(recovery_path)
    validate_recovery_task(args.task_id, recovery)

    precommit_path = V30 / "V30_PRECOMMIT.json"
    grammar_path = V30 / "V30_GRAMMAR.json"
    manifest_path = V30 / "V30_GRAMMAR_MANIFEST.json"
    precommit = load(precommit_path)
    grammar = load(grammar_path)
    manifest = load(manifest_path)
    if grammar["precommit_sha256"] != sha256_json(precommit):
        raise RuntimeError("grammar is not bound to the frozen precommit")
    if sha256_file(grammar_path) != recovery["grammar_file_sha256"]:
        raise RuntimeError("grammar file hash mismatch")
    if sha256_file(manifest_path) != recovery["grammar_manifest_sha256"]:
        raise RuntimeError("grammar manifest hash mismatch")
    if grammar["candidate_sha256"] != recovery["candidate_sequence_sha256"]:
        raise RuntimeError("candidate sequence hash mismatch")
    if manifest["candidate_sha256"] != grammar["candidate_sha256"]:
        raise RuntimeError("manifest candidate sequence mismatch")

    examples, generation = generate_examples(
        args.task_id,
        args.arcgen_root,
        int(precommit["examples_per_task"]),
        int(precommit["generator_attempts_per_task"]),
        int(precommit["per_generation_timeout_seconds"]),
    )
    demonstrations = [
        {"input": source, "output": target}
        for source, target in examples
    ]
    if len(examples) != int(precommit["examples_per_task"]):
        report = {
            "schema": "lexigen-v30-task-scan-v1",
            "task_id": args.task_id,
            "status": "generator_invalid",
            "accepted_examples": len(examples),
            "generation": generation,
            "demonstration_sha256": sha256_json(demonstrations),
            "precommit_sha256": sha256_file(precommit_path),
            "grammar_sha256": sha256_file(grammar_path),
            "grammar_manifest_sha256": sha256_file(manifest_path),
            "enumeration": None,
        }
    else:
        enumeration = evaluate_candidates_memoized(examples, grammar, precommit)
        report = {
            "schema": "lexigen-v30-task-scan-v1",
            "task_id": args.task_id,
            "status": "completed",
            "accepted_examples": len(examples),
            "generation": generation,
            "demonstration_sha256": sha256_json(demonstrations),
            "precommit_sha256": sha256_file(precommit_path),
            "grammar_sha256": sha256_file(grammar_path),
            "grammar_manifest_sha256": sha256_file(manifest_path),
            "enumeration": enumeration,
        }
    write(args.output, report)
    summary = {
        "task_id": args.task_id,
        "status": report["status"],
        "accepted_examples": report["accepted_examples"],
    }
    if report["enumeration"] is not None:
        summary.update({
            "candidates_tested": report["enumeration"]["concrete_candidates_tested"],
            "exact_candidates": report["enumeration"]["exact_candidate_count"],
            "candidate_cap_reached": report["enumeration"]["candidate_cap_reached"],
        })
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
