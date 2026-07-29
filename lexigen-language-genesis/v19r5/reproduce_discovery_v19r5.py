from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

from scan_discovery_v19r5 import (
    ACTIONS,
    BASE_MODES,
    HERE,
    SET_MODES,
    _prepare,
    _render,
    canonical,
    factorized_production,
    file_sha256,
    generate_examples,
    load,
    sha256_json,
    structure_descriptor,
    write,
)


def exact_task_structures(examples):
    prepared_by_marker = {
        marker: [_prepare(source, target, marker) for source, target in examples]
        for marker in range(10)
    }
    task_structures: dict[str, dict[str, Any]] = {}
    exact_complete = 0
    for marker in range(10):
        prepared = prepared_by_marker[marker]
        if any(item is None for item in prepared):
            continue
        valid = [item for item in prepared if item is not None]
        for background in range(10):
            for set_mode in SET_MODES:
                for base_mode in BASE_MODES:
                    allowed_by_side: list[list[int]] = []
                    for side in range(4):
                        side_items = [item for item in valid if item["side"] == side]
                        allowed = [
                            action_index
                            for action_index in range(len(ACTIONS))
                            if all(
                                _render(
                                    item,
                                    action_index,
                                    set_mode,
                                    base_mode,
                                    background,
                                ) == item["target"]
                                for item in side_items
                            )
                        ]
                        allowed_by_side.append(allowed)
                    if any(not allowed for allowed in allowed_by_side):
                        continue
                    for indexes in itertools.product(*allowed_by_side):
                        actions = tuple(ACTIONS[index] for index in indexes)
                        exact_complete += 1
                        structure = structure_descriptor(actions, set_mode, base_mode)
                        structure_key = canonical(structure)
                        entry = task_structures.setdefault(
                            structure_key,
                            {"structure": structure, "arguments": []},
                        )
                        entry["arguments"].append({
                            "marker_colour": marker,
                            "output_background": background,
                        })
    return exact_complete, task_structures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "V19R5_DISCOVERY_REPRODUCTION.json",
    )
    args = parser.parse_args()
    registry_path = HERE / "V19R5_REGISTRY.json"
    registry = load(registry_path)
    precommit = load(HERE / "V19R5_PRECOMMIT.json")
    if file_sha256(registry_path) != precommit["registry_sha256"]:
        raise RuntimeError("registry identity changed")
    discovery_ids = list(registry["discovery_task_ids"])
    validation_ids = set(registry["validation_task_ids"])
    if len(discovery_ids) != precommit["discovery_task_count"]:
        raise RuntimeError("discovery denominator changed")

    structure_index: dict[str, dict[str, Any]] = {}
    task_reports = []
    imported_ids = []
    total_complete_programs = 0
    generator_invalid = 0

    for task_index, task_id in enumerate(discovery_ids):
        if task_id in validation_ids:
            raise RuntimeError("validation identity entered reproduction")
        examples, attempts, rejections, rejection_types = generate_examples(
            task_id,
            args.arcgen_root,
            int(precommit["discovery_examples_per_task"]),
            int(precommit["maximum_generator_attempts_per_task"]),
        )
        imported_ids.append(task_id)
        expected_examples = int(precommit["discovery_examples_per_task"])
        if len(examples) != expected_examples:
            generator_invalid += 1
            task_reports.append({
                "task_id": task_id,
                "status": "generator_invalid",
                "accepted_examples": len(examples),
                "generator_attempts": attempts,
                "generator_rejections": rejections,
                "rejection_types": rejection_types,
                "exact_complete_programs": 0,
                "exact_structures": 0,
            })
            continue

        exact_complete, task_structures = exact_task_structures(examples)
        total_complete_programs += exact_complete
        exact_structure_hashes = []
        for structure_key in sorted(task_structures):
            task_entry = task_structures[structure_key]
            production = factorized_production(task_entry["structure"])
            production_hash = sha256_json(production)
            exact_structure_hashes.append(production_hash)
            global_entry = structure_index.setdefault(
                production_hash,
                {"production": production, "discovery_tasks": {}},
            )
            global_entry["discovery_tasks"][task_id] = sorted(
                task_entry["arguments"], key=canonical
            )

        task_reports.append({
            "task_id": task_id,
            "status": "completed",
            "accepted_examples": len(examples),
            "generator_attempts": attempts,
            "generator_rejections": rejections,
            "rejection_types": rejection_types,
            "exact_complete_programs": exact_complete,
            "exact_structures": len(task_structures),
            "exact_structure_hashes": exact_structure_hashes,
        })
        if (task_index + 1) % 50 == 0:
            print(json.dumps({
                "tasks_processed": task_index + 1,
                "total_tasks": len(discovery_ids),
                "structures_seen": len(structure_index),
                "exact_complete_programs": total_complete_programs,
            }, sort_keys=True), flush=True)

    if set(imported_ids) & validation_ids:
        raise RuntimeError("validation generator was imported")

    qualifying = []
    library_dir = HERE / "reproduction-library"
    if library_dir.exists():
        shutil.rmtree(library_dir)
    library_dir.mkdir(parents=True)
    for production_hash in sorted(structure_index):
        entry = structure_index[production_hash]
        task_ids = sorted(entry["discovery_tasks"])
        if len(task_ids) < 2:
            continue
        production_path = library_dir / f"production-{production_hash}.json"
        write(production_path, entry["production"])
        qualifying.append({
            "production_sha256": production_hash,
            "production_file_sha256": file_sha256(production_path),
            "structure": entry["production"]["origin"]["structure"],
            "discovery_task_count": len(task_ids),
            "discovery_task_ids": task_ids,
            "arguments_by_task": {
                task_id: entry["discovery_tasks"][task_id]
                for task_id in task_ids
            },
        })

    library = {
        "schema": "lexigen-v19r5-factorized-library-v1",
        "registry_sha256": precommit["registry_sha256"],
        "meta_grammar_source_sha256": precommit["frozen_meta_grammar"]["source_sha256"],
        "discovery_task_count": len(discovery_ids),
        "generator_invalid_tasks": generator_invalid,
        "qualifying_production_count": len(qualifying),
        "qualification_rule": "at least two distinct discovery task identities",
        "validation_outputs_opened": False,
        "productions": qualifying,
    }
    library_path = HERE / "V19R5_LIBRARY_REPRODUCTION.json"
    write(library_path, library)
    report = {
        "schema": "lexigen-v19r5-discovery-reproduction-v1",
        "registry_sha256": precommit["registry_sha256"],
        "discovery_tasks": len(discovery_ids),
        "discovery_generators_imported": len(imported_ids),
        "validation_generators_imported": 0,
        "generator_invalid_tasks": generator_invalid,
        "total_exact_complete_programs": total_complete_programs,
        "distinct_structures_seen": len(structure_index),
        "qualifying_productions": len(qualifying),
        "library_sha256": file_sha256(library_path),
        "validation_outputs_opened": False,
        "transfer_demonstrated": False,
        "world_level_breakthrough": False,
        "task_reports": task_reports,
    }
    write(args.output, report)
    print("SUMMARY", json.dumps({
        "discovery_tasks": report["discovery_tasks"],
        "generator_invalid_tasks": generator_invalid,
        "total_exact_complete_programs": total_complete_programs,
        "distinct_structures_seen": len(structure_index),
        "qualifying_productions": len(qualifying),
        "library_sha256": report["library_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
