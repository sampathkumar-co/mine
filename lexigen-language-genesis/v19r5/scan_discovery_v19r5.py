from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V19R3 = HERE.parent / "v19r3"
V19R2 = HERE.parent / "v19r2"
for folder in (HERE, V19R3, V19R2):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from enumerate_v19r3 import (
    ACTIONS, BASE_MODES, SET_MODES, TOTAL_CANDIDATES,
    _prepare, _render, build_program, param,
)
from runtime_v19r2 import as_grid, canonical, sha256_json

PRODUCTION_SCHEMA = "lexigen-v19r5-factorized-production-v1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(task_id: str, attempt: int) -> int:
    text = f"lexigen-v19r5-case-v1:discovery:{task_id}:{attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def generate_examples(task_id: str, arcgen_root: Path, count: int, max_attempts: int):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    module = importlib.import_module(f"tasks.task_{task_id}")
    examples = []
    attempts = rejections = 0
    rejection_types: dict[str, int] = {}
    while len(examples) < count and attempts < max_attempts:
        seed = seed_for(task_id, attempts)
        attempts += 1
        random.seed(seed)
        try:
            pair = module.generate()
            examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
        except (ValueError, IndexError, TypeError, RuntimeError) as error:
            rejections += 1
            name = type(error).__name__
            rejection_types[name] = rejection_types.get(name, 0) + 1
    return examples, attempts, rejections, rejection_types


def structure_descriptor(actions, set_mode: str, base_mode: str):
    return {
        "actions": list(actions),
        "set_mode": set_mode,
        "base_mode": base_mode,
    }


def factorized_production(structure: dict[str, Any]):
    actions = tuple(structure["actions"])
    body = build_program(
        param("marker_colour"),
        param("output_background"),
        actions,
        str(structure["set_mode"]),
        str(structure["base_mode"]),
    )
    descriptor_hash = sha256_json(structure)
    return {
        "schema": PRODUCTION_SCHEMA,
        "name": f"generated_{descriptor_hash[:16]}",
        "parameters": [
            {"name": "marker_colour", "type": "colour"},
            {"name": "output_background", "type": "colour"},
        ],
        "body": body,
        "origin": {
            "method": "registry_complete_program_factorization",
            "structure": structure,
            "structure_sha256": descriptor_hash,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V19R5_DISCOVERY_REPORT.json")
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
            raise RuntimeError("validation identity entered discovery scan")
        examples, attempts, rejections, rejection_types = generate_examples(
            task_id,
            args.arcgen_root,
            int(precommit["discovery_examples_per_task"]),
            int(precommit["maximum_generator_attempts_per_task"]),
        )
        imported_ids.append(task_id)
        if len(examples) != int(precommit["discovery_examples_per_task"]):
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
            for background in range(10):
                for left in ACTIONS:
                    for right in ACTIONS:
                        for top in ACTIONS:
                            for bottom in ACTIONS:
                                actions = (left, right, top, bottom)
                                indexes = tuple(ACTIONS.index(name) for name in actions)
                                for set_mode in SET_MODES:
                                    for base_mode in BASE_MODES:
                                        exact = True
                                        for item in prepared:
                                            assert item is not None
                                            predicted = _render(
                                                item,
                                                indexes[item["side"]],
                                                set_mode,
                                                base_mode,
                                                background,
                                            )
                                            if predicted != item["target"]:
                                                exact = False
                                                break
                                        if not exact:
                                            continue
                                        exact_complete += 1
                                        structure = structure_descriptor(actions, set_mode, base_mode)
                                        structure_key = canonical(structure)
                                        entry = task_structures.setdefault(structure_key, {
                                            "structure": structure,
                                            "arguments": [],
                                        })
                                        entry["arguments"].append({
                                            "marker_colour": marker,
                                            "output_background": background,
                                        })

        total_complete_programs += exact_complete
        exact_structure_hashes = []
        for structure_key in sorted(task_structures):
            task_entry = task_structures[structure_key]
            structure = task_entry["structure"]
            production = factorized_production(structure)
            production_hash = sha256_json(production)
            exact_structure_hashes.append(production_hash)
            global_entry = structure_index.setdefault(production_hash, {
                "production": production,
                "discovery_tasks": {},
            })
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
        if (task_index + 1) % 25 == 0:
            print(json.dumps({
                "tasks_processed": task_index + 1,
                "total_tasks": len(discovery_ids),
                "structures_seen": len(structure_index),
                "exact_complete_programs": total_complete_programs,
            }, sort_keys=True), flush=True)

    if set(imported_ids) & validation_ids:
        raise RuntimeError("validation generator was imported")

    qualifying = []
    library_dir = HERE / "library"
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
    write(HERE / "V19R5_LIBRARY.json", library)
    report = {
        "schema": "lexigen-v19r5-discovery-scan-report-v1",
        "registry_sha256": precommit["registry_sha256"],
        "discovery_tasks": len(discovery_ids),
        "discovery_generators_imported": len(imported_ids),
        "validation_generators_imported": 0,
        "generator_invalid_tasks": generator_invalid,
        "total_exact_complete_programs": total_complete_programs,
        "distinct_structures_seen": len(structure_index),
        "qualifying_productions": len(qualifying),
        "library_sha256": file_sha256(HERE / "V19R5_LIBRARY.json"),
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
