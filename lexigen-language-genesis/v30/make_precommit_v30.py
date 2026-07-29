from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTRY = ROOT / "v19r5" / "V19R5_REGISTRY.json"
SOURCES = ROOT / "v29" / "V29_SOURCE_STRUCTURES.json"
OUTPUT = HERE / "V30_PRECOMMIT.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_task_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key.endswith("task_ids") and isinstance(child, list):
                result.update(item for item in child if isinstance(item, str))
            result.update(collect_task_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(collect_task_ids(child))
    return result


def source_operators(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        op = value.get("op")
        if isinstance(op, str):
            result.add(op)
        for child in value.values():
            result.update(source_operators(child))
    elif isinstance(value, list):
        for child in value:
            result.update(source_operators(child))
    return result


def main() -> None:
    registry = load(REGISTRY)
    sources = load(SOURCES)
    used: set[str] = set()
    for path in sorted(ROOT.rglob("*PRECOMMIT.json")):
        if path == OUTPUT:
            continue
        used.update(collect_task_ids(load(path)))
    used.update(collect_task_ids(sources))

    available = [
        task_id for task_id in registry["validation_task_ids"]
        if task_id not in used
    ]
    ranked = sorted(
        available,
        key=lambda task_id: hashlib.sha256(
            f"lexigen-v30-fresh-validation:{task_id}".encode()
        ).hexdigest(),
    )
    validation_task_ids = ranked[:20]
    assert len(validation_task_ids) == 20

    operators = sorted(source_operators(sources))
    signatures = {
        "bbox_border": ["PointSet", "PointSet"],
        "canvas": ["Color", "Grid"],
        "crop_bbox": ["Grid", "PointSet", "Grid"],
        "dilate4": ["PointSet", "PointSet"],
        "erode4": ["PointSet", "PointSet"],
        "holes": ["PointSet", "PointSet"],
        "input_grid": ["Grid"],
        "least_non_background": ["Color"],
        "most_non_background": ["Color"],
        "non_background_points": ["PointSet"],
        "paint": ["Grid", "PointSet", "Color", "Grid"],
        "param_color": ["Color"],
        "points_of_color": ["Color", "PointSet"],
    }
    assert set(operators) == set(signatures)

    precommit = {
        "schema": "lexigen-v30-source-induced-motif-grammar-precommit-v1",
        "arcgen_commit": registry["arcgen_commit"],
        "source_structures_sha256": sha256_file(SOURCES),
        "source_structure_count": len(sources["structures"]),
        "source_operator_inventory": operators,
        "typed_productions": {
            op: signatures[op] for op in operators
        },
        "grammar_rule": (
            "Enumerate every well-typed AST using only productions observed in "
            "the three frozen v28 exact programs; permit novel recombination, "
            "but add no unseen operator or hand-authored task rule."
        ),
        "enumeration": {
            "maximum_depth": 5,
            "maximum_nodes": 9,
            "maximum_structural_candidates": 50000,
            "maximum_concrete_candidates_per_task": 500000,
            "candidate_order": [
                "maximum_depth",
                "total_nodes",
                "canonical_ast",
                "parameter_assignment",
            ],
            "root_type": "Grid",
            "reject_input_identity": True,
            "literal_color_domain": list(range(10)),
            "no_semantic_pruning_before_exact_check": True,
        },
        "examples_per_task": 6,
        "generator_attempts_per_task": 16,
        "per_generation_timeout_seconds": 5,
        "validation_task_ids": validation_task_ids,
        "validation_identity_selection": (
            "Rank unused identities from the frozen v19r5 validation registry by "
            "sha256(lexigen-v30-fresh-validation:<task_id>); take first 20."
        ),
        "replacement_tasks_allowed": False,
        "post_validation_edits_allowed": False,
        "success_rule": (
            "Retain every exact candidate on six demonstrations. If any exist, "
            "freeze the minimum candidate by the preregistered order before fresh testing."
        ),
        "fresh_validation_rule": {
            "cases": 1000,
            "seed_namespace": "lexigen-v30-fresh",
            "independent_runtime_required": True,
            "verifier_cosynthesis_required_for_breakthrough_claim": True,
        },
        "claim_boundary": {
            "one_exact_heldout_task": "public heldout synthesis event",
            "two_exact_heldout_tasks": "repeated public heldout transfer",
            "world_level_breakthrough": (
                "forbidden without fresh-case success, independent runtime agreement, "
                "and verifier co-synthesis"
            ),
        },
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps(precommit, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({
        "precommit_sha256": sha256_file(OUTPUT),
        "fresh_validation_task_count": len(validation_task_ids),
        "first_ids": validation_task_ids[:5],
        "source_operators": operators,
        "previously_used_task_count": len(used),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
