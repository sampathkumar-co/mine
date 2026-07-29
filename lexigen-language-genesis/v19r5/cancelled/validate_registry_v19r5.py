from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V19R2 = HERE.parent / "v19r2"
V17 = HERE.parent / "v17"
for folder in (HERE, V19R2, V17):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from constructive_dsl_v17 import synthesize as synthesize_v17
from portable_runtime_v19r2 import execute_portable
from runtime_v19r2 import as_grid, canonical, execute, sha256_json

DISCOVERY_LIBRARY_COMMIT = "e7d2817783c1b13bf141bde75b07ea5353af397b"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(task_id: str, attempt: int) -> int:
    text = f"lexigen-v19r5-case-v1:validation:{task_id}:{attempt}"
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


def substitute(value: Any, arguments: dict[str, int]) -> Any:
    if isinstance(value, dict):
        if value.get("op") == "param":
            name = str(value["name"])
            if name not in arguments:
                raise RuntimeError(f"missing production argument: {name}")
            return arguments[name]
        return {key: substitute(child, arguments) for key, child in value.items()}
    if isinstance(value, list):
        return [substitute(child, arguments) for child in value]
    return value


def expand(production: dict[str, Any], arguments: dict[str, int]):
    expected = sorted(str(item["name"]) for item in production["parameters"])
    if expected != sorted(arguments):
        raise RuntimeError("argument mismatch")
    return substitute(production["body"], arguments)


def visible_colours(examples) -> list[int]:
    return sorted({cell for source, target in examples for grid in (source, target) for row in grid for cell in row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V19R5_VALIDATION_REPORT.json")
    args = parser.parse_args()

    precommit = load(HERE / "V19R5_PRECOMMIT.json")
    registry = load(HERE / "V19R5_REGISTRY.json")
    library_path = HERE / "V19R5_LIBRARY.json"
    library = load(library_path)
    discovery_ids = set(registry["discovery_task_ids"])
    validation_ids = list(registry["validation_task_ids"])
    if len(validation_ids) != precommit["validation_task_count"]:
        raise RuntimeError("validation denominator changed")
    if set(validation_ids) & discovery_ids:
        raise RuntimeError("registry split overlap")
    if file_sha256(library_path) != load(HERE / "V19R5_DISCOVERY_REPORT.json")["library_sha256"]:
        raise RuntimeError("frozen library identity changed")
    if library["validation_outputs_opened"]:
        raise RuntimeError("library claims validation was opened before freeze")

    productions = []
    for item in library["productions"]:
        path = HERE / "library" / f"production-{item['production_sha256']}.json"
        if file_sha256(path) != item["production_file_sha256"]:
            raise RuntimeError("production file identity changed")
        production = load(path)
        if sha256_json(production) != item["production_sha256"]:
            raise RuntimeError("production semantic hash changed")
        productions.append((item, production))

    candidates_dir = HERE / "validation_candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True)
    task_reports = []
    transfer_matches = []
    generator_invalid = 0
    validation_imported = []

    for task_index, task_id in enumerate(validation_ids):
        examples, attempts, rejections, rejection_types = generate_examples(
            task_id,
            args.arcgen_root,
            int(precommit["validation_examples_per_task"]),
            int(precommit["maximum_generator_attempts_per_task"]),
        )
        validation_imported.append(task_id)
        if len(examples) != int(precommit["validation_examples_per_task"]):
            generator_invalid += 1
            task_reports.append({
                "task_id": task_id,
                "status": "generator_invalid",
                "accepted_examples": len(examples),
                "generator_attempts": attempts,
                "generator_rejections": rejections,
                "rejection_types": rejection_types,
                "exact_structure_matches": 0,
                "strong_transfer_candidates": 0,
            })
            continue

        v17_status = "program"
        v17_report: dict[str, Any]
        try:
            _, result = synthesize_v17(examples)
            v17_report = result
        except RuntimeError as error:
            v17_status = "no_program"
            v17_report = {"failure": str(error)}

        palette = visible_colours(examples)
        pairs = list(itertools.product(palette, repeat=2))
        if len(pairs) > 100:
            raise RuntimeError(f"argument budget exceeded for task {task_id}")
        task_matches = []
        for library_item, production in productions:
            exact_arguments = []
            runtime_invalid = 0
            for marker, background in pairs:
                arguments = {
                    "marker_colour": int(marker),
                    "output_background": int(background),
                }
                concrete = expand(production, arguments)
                if task_id in canonical(production) or task_id in canonical(concrete):
                    raise RuntimeError("task identity leaked into production")
                try:
                    primary = [execute(concrete, source) for source, _ in examples]
                    portable = [execute_portable(concrete, source) for source, _ in examples]
                except Exception:
                    runtime_invalid += 1
                    continue
                targets = [target for _, target in examples]
                if primary == portable == targets:
                    exact_arguments.append({
                        "arguments": arguments,
                        "arguments_sha256": sha256_json(arguments),
                        "concrete_program_sha256": sha256_json(concrete),
                    })
            if not exact_arguments:
                continue
            match = {
                "task_id": task_id,
                "production_sha256": library_item["production_sha256"],
                "structure": library_item["structure"],
                "discovery_task_count": library_item["discovery_task_count"],
                "exact_argument_pairs": len(exact_arguments),
                "arguments": exact_arguments,
                "argument_pairs_evaluated": len(pairs),
                "runtime_invalid_argument_pairs": runtime_invalid,
                "v17_baseline_status": v17_status,
                "strong_transfer_candidate": len(exact_arguments) == 1 and v17_status == "no_program",
            }
            task_matches.append(match)
            transfer_matches.append(match)
            if match["strong_transfer_candidate"]:
                stem = f"{task_id}-{library_item['production_sha256']}"
                write(candidates_dir / f"candidate-{stem}.json", match)
                write(candidates_dir / f"production-{stem}.json", production)
                write(candidates_dir / f"concrete-{stem}.json", expand(production, exact_arguments[0]["arguments"]))

        task_reports.append({
            "task_id": task_id,
            "status": "completed",
            "accepted_examples": len(examples),
            "generator_attempts": attempts,
            "generator_rejections": rejections,
            "rejection_types": rejection_types,
            "palette": palette,
            "argument_pairs_per_structure": len(pairs),
            "library_structures_evaluated": len(productions),
            "v17_baseline_status": v17_status,
            "v17_baseline_report": v17_report,
            "exact_structure_matches": len(task_matches),
            "strong_transfer_candidates": sum(item["strong_transfer_candidate"] for item in task_matches),
            "matches": task_matches,
        })
        if (task_index + 1) % 25 == 0:
            print(json.dumps({
                "validation_tasks_processed": task_index + 1,
                "total_validation_tasks": len(validation_ids),
                "transfer_matches": len(transfer_matches),
                "strong_transfer_candidates": sum(item["strong_transfer_candidate"] for item in transfer_matches),
            }, sort_keys=True), flush=True)

    if validation_imported != validation_ids:
        raise RuntimeError("validation import order changed")
    strong = [item for item in transfer_matches if item["strong_transfer_candidate"]]
    report = {
        "schema": "lexigen-v19r5-heldout-registry-validation-v1",
        "discovery_library_commit": DISCOVERY_LIBRARY_COMMIT,
        "registry_sha256": precommit["registry_sha256"],
        "library_sha256": file_sha256(library_path),
        "frozen_production_count": len(productions),
        "validation_tasks": len(validation_ids),
        "validation_generators_imported": len(validation_imported),
        "generator_invalid_tasks": generator_invalid,
        "tasks_with_exact_structure_match": len({item["task_id"] for item in transfer_matches}),
        "total_structure_matches": len(transfer_matches),
        "strong_transfer_candidates": len(strong),
        "strong_candidate_task_ids": sorted({item["task_id"] for item in strong}),
        "structure_edits_after_library_freeze": 0,
        "task_specific_code_added": False,
        "public_heldout_transfer_demonstrated": len(strong) > 0,
        "sealed_external_success": False,
        "world_level_breakthrough": False,
        "transfer_matches": transfer_matches,
        "task_reports": task_reports,
    }
    write(args.output, report)
    print("SUMMARY", json.dumps({key: report[key] for key in (
        "validation_tasks", "generator_invalid_tasks",
        "tasks_with_exact_structure_match", "total_structure_matches",
        "strong_transfer_candidates", "public_heldout_transfer_demonstrated",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
