from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from portable_runtime_v7 import as_grid as portable_grid
from portable_runtime_v7 import execute_portable
from replay_gate4_v7 import load_examples
from semantic_ast_v7 import (
    Grid,
    canonical_json,
    execute_ast,
    extract_holes,
    extract_objects,
    regions_match,
    synthesize_ast,
)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def strongest_observable_ast(examples: list[tuple[Grid, Grid]]) -> dict[str, Any]:
    result = synthesize_ast(examples)
    if result.ast is None:
        raise AssertionError("no semantic AST fits current evidence")
    candidates = [
        candidate
        for candidate in result.exact_candidates
        if candidate["scene"]["hole_boundary"] == "all"
        and candidate["scene"]["exclude_frame_objects"] is True
        and candidate["scene"].get("object_colour_role") == "single_component"
        and candidate["match"]["feature"] == "normalised_points"
        and candidate["match"].get("symmetry") == "identity"
        and candidate["render"]["erase_source"] is True
    ]
    if candidates:
        return min(
            candidates,
            key=lambda ast: hashlib.sha256(canonical_json(ast).encode()).digest(),
        )
    selected = json.loads(json.dumps(result.ast))
    selected["scene"]["hole_boundary"] = "all"
    selected["scene"]["exclude_frame_objects"] = True
    selected["scene"]["object_colour_role"] = "single_component"
    selected["match"] = {"feature": "normalised_points", "symmetry": "identity"}
    selected["render"]["erase_source"] = True
    return selected


def assignment_count(ast: dict[str, Any], grid: Grid, limit: int = 2) -> int:
    scene = ast["scene"]
    holes = extract_holes(
        grid,
        int(scene["background_colour"]),
        int(scene["frame_colour"]),
        str(scene["hole_boundary"]),
    )
    objects = extract_objects(
        grid,
        int(scene["background_colour"]),
        int(scene["frame_colour"]),
        bool(scene["exclude_frame_objects"]),
        str(scene.get("object_colour_role", "any")),
    )
    options = [
        [
            index
            for index, source in enumerate(objects)
            if regions_match(source, hole, ast["match"])
        ]
        for hole in holes
    ]
    count = 0

    def search(position: int, used: frozenset[int]) -> None:
        nonlocal count
        if count >= limit:
            return
        if position == len(options):
            count += 1
            return
        for index in options[position]:
            if index not in used:
                search(position + 1, used | {index})

    search(0, frozenset())
    return count


def generate_case(task_module, seed: int) -> tuple[Grid, Grid]:
    random.seed(seed)
    pair = task_module.generate()
    source = tuple(tuple(int(cell) for cell in row) for row in pair["input"])
    target = tuple(tuple(int(cell) for cell in row) for row in pair["output"])
    return source, target


def ast_summary(ast: dict[str, Any]) -> dict[str, Any]:
    return {
        "background_colour": ast["scene"]["background_colour"],
        "frame_colour": ast["scene"]["frame_colour"],
        "hole_boundary": ast["scene"]["hole_boundary"],
        "object_colour_role": ast["scene"].get("object_colour_role"),
        "match_feature": ast["match"]["feature"],
        "match_symmetry": ast["match"].get("symmetry"),
        "erase_source": ast["render"]["erase_source"],
    }


def run(
    training: Path,
    arcgen_root: Path,
    output_dir: Path,
    discovery_start: int,
    discovery_count: int,
    holdout_start: int,
    holdout_count: int,
) -> dict[str, Any]:
    examples, payload = load_examples(training)
    sys.path.insert(0, str(arcgen_root))
    from tasks import task_228f6490  # type: ignore

    rounds: list[dict[str, Any]] = []
    ambiguous_discovery: list[int] = []
    for round_index in range(16):
        result = synthesize_ast(examples)
        if result.ast is None:
            raise AssertionError("v7 grammar became inconsistent with accumulated evidence")
        discriminator = strongest_observable_ast(examples)
        failure = None
        for seed in range(discovery_start, discovery_start + discovery_count):
            source, target = generate_case(task_228f6490, seed)
            if execute_ast(result.ast, source) == target:
                continue
            if assignment_count(discriminator, source) != 1:
                ambiguous_discovery.append(seed)
                continue
            failure = seed, source, target
            break
        rounds.append(
            {
                "round": round_index,
                "ast": ast_summary(result.ast),
                "ast_sha256": hashlib.sha256(
                    canonical_json(result.ast).encode()
                ).hexdigest(),
                "exact_candidate_count": result.exact_candidate_count,
                "counterexample_seed": None if failure is None else failure[0],
            }
        )
        if failure is None:
            break
        examples.append((failure[1], failure[2]))
    else:
        raise AssertionError("CEGIS round budget exhausted")

    final = synthesize_ast(examples)
    if final.ast is None:
        raise AssertionError("final v7 AST is missing")
    discriminator = strongest_observable_ast(examples)
    identifiable = 0
    ambiguous = 0
    exact = 0
    portable_exact = 0
    failures: list[int] = []
    for seed in range(holdout_start, holdout_start + holdout_count):
        source, target = generate_case(task_228f6490, seed)
        if assignment_count(discriminator, source) != 1:
            ambiguous += 1
            continue
        identifiable += 1
        primary = execute_ast(final.ast, source)
        portable = execute_portable(final.ast, portable_grid(source))
        exact += int(primary == target)
        portable_exact += int(portable == target and portable == primary)
        if primary != target:
            failures.append(seed)

    if failures:
        raise AssertionError(f"identifiable holdout failures: {failures[:20]}")
    if exact != identifiable or portable_exact != identifiable:
        raise AssertionError("portable runtime disagreed on identifiable holdout")

    task_source = arcgen_root / "tasks" / "task_228f6490.py"
    report = {
        "version": "v7",
        "benchmark": "ARC-GEN 228f6490 counterexample-guided semantic synthesis",
        "status": "post-failure external-family validation; no blind breakthrough claim",
        "arcgen_commit": git_commit(arcgen_root),
        "task_source_sha256": hashlib.sha256(task_source.read_bytes()).hexdigest(),
        "source_task_id": payload.get("selected_task_id"),
        "source_sealed_outputs_sha256": payload.get("sealed_outputs_sha256"),
        "sealed_outputs_accessed": False,
        "initial_demonstration_count": len(payload["train"]),
        "counterexample_count": len(examples) - len(payload["train"]),
        "rounds": rounds,
        "ambiguous_discovery_count": len(set(ambiguous_discovery)),
        "ambiguous_discovery_seeds": sorted(set(ambiguous_discovery)),
        "final_ast": final.ast,
        "final_ast_sha256": hashlib.sha256(
            canonical_json(final.ast).encode()
        ).hexdigest(),
        "holdout_start": holdout_start,
        "holdout_count": holdout_count,
        "holdout_identifiable": identifiable,
        "holdout_ambiguous": ambiguous,
        "holdout_exact": exact,
        "portable_holdout_exact": portable_exact,
        "identifiable_accuracy": exact / identifiable if identifiable else 0.0,
        "claim_boundary": (
            "v7 autonomously refines a relational AST using public post-failure generator counterexamples. "
            "The semantic meta-grammar and identifiability model remain human supplied."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "lexigen-v7-cegis-semantic-artifact-v1",
        "name": "cegis_semantics_" + report["final_ast_sha256"][:12],
        "semantic_ast": final.ast,
        "identifiability_ast": discriminator,
        "rounds": rounds,
        "provenance": {
            "arcgen_commit": report["arcgen_commit"],
            "task_source_sha256": report["task_source_sha256"],
            "sealed_outputs_accessed": False,
            "human_supplied_meta_grammar": True,
        },
    }
    artifact_path = output_dir / "v7-cegis-artifact.json"
    report_path = output_dir / "v7-cegis-report.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "artifact": artifact["name"],
        "rounds": len(rounds),
        "counterexamples": [
            row["counterexample_seed"]
            for row in rounds
            if row["counterexample_seed"] is not None
        ],
        "final_ast": ast_summary(final.ast),
        "holdout_identifiable": identifiable,
        "holdout_ambiguous": ambiguous,
        "holdout_accuracy": report["identifiable_accuracy"],
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7"))
    parser.add_argument("--discovery-start", type=int, default=10_000)
    parser.add_argument("--discovery-count", type=int, default=10_000)
    parser.add_argument("--holdout-start", type=int, default=20_000)
    parser.add_argument("--holdout-count", type=int, default=10_000)
    args = parser.parse_args()
    run(
        args.training,
        args.arcgen_root,
        args.output_dir,
        args.discovery_start,
        args.discovery_count,
        args.holdout_start,
        args.holdout_count,
    )


if __name__ == "__main__":
    main()
