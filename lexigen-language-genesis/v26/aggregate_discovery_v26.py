from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-output", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = HERE / "V26_PRECOMMIT.json"
    precommit = load(precommit_path)
    expected_ids = list(precommit["discovery_task_ids"])
    found: dict[str, tuple[Path, dict[str, Any]]] = {}

    for path in sorted(args.reports_root.rglob("*.json")):
        report = load(path)
        if report.get("schema") != "lexigen-v26-task-scan-v1":
            continue
        task_id = str(report["task_id"])
        if task_id in found:
            raise RuntimeError(f"duplicate task report: {task_id}")
        found[task_id] = (path, report)

    if set(found) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(found))
        extra = sorted(set(found) - set(expected_ids))
        raise RuntimeError(f"report denominator changed: missing={missing}, extra={extra}")

    structure_index: dict[str, dict[str, Any]] = {}
    task_summaries: list[dict[str, Any]] = []
    totals = {
        "accepted_examples": 0,
        "generation_failures": 0,
        "generation_timeouts": 0,
        "raw_candidate_evaluations": 0,
        "runtime_invalid_candidates": 0,
        "semantic_duplicates": 0,
        "total_retained_expressions": 0,
        "exact_concrete_programs": 0,
    }
    generator_invalid = 0
    completed = 0

    for task_id in expected_ids:
        path, report = found[task_id]
        if report.get("split") != "discovery":
            raise RuntimeError(f"wrong split for {task_id}")
        generation = report["generation"]
        totals["accepted_examples"] += int(report["accepted_examples"])
        totals["generation_failures"] += int(generation["failures"])
        totals["generation_timeouts"] += int(generation["timeouts"])
        summary: dict[str, Any] = {
            "task_id": task_id,
            "status": report["status"],
            "report_sha256": sha256_file(path),
            "accepted_examples": report["accepted_examples"],
        }

        if report["status"] != "completed":
            generator_invalid += 1
            task_summaries.append(summary)
            continue

        completed += 1
        enumeration = report["enumeration"]
        for key in (
            "raw_candidate_evaluations",
            "runtime_invalid_candidates",
            "semantic_duplicates",
            "total_retained_expressions",
            "exact_concrete_programs",
        ):
            totals[key] += int(enumeration[key])
        summary.update({
            "enumeration_complete": enumeration["enumeration_complete"],
            "exhausted_reason": enumeration["exhausted_reason"],
            "raw_candidate_evaluations": enumeration["raw_candidate_evaluations"],
            "total_retained_expressions": enumeration["total_retained_expressions"],
            "exact_concrete_programs": enumeration["exact_concrete_programs"],
            "exact_abstract_structures": enumeration["exact_abstract_structures"],
        })
        task_summaries.append(summary)

        for item in enumeration["exact_structures"]:
            structure_hash = str(item["structure_sha256"])
            if structure_hash != hashlib.sha256(
                canonical(item["structure"]).encode("utf-8")
            ).hexdigest():
                raise RuntimeError(f"structure hash mismatch: {task_id}")
            entry = structure_index.setdefault(structure_hash, {
                "structure": item["structure"],
                "tasks": {},
                "minimum_depth": item["minimum_depth"],
                "minimum_nodes": item["minimum_nodes"],
            })
            if canonical(entry["structure"]) != canonical(item["structure"]):
                raise RuntimeError(f"structure collision: {structure_hash}")
            entry["minimum_depth"] = min(
                int(entry["minimum_depth"]), int(item["minimum_depth"])
            )
            entry["minimum_nodes"] = min(
                int(entry["minimum_nodes"]), int(item["minimum_nodes"])
            )
            entry["tasks"][task_id] = item["concrete_programs"]

    if args.library_dir.exists():
        shutil.rmtree(args.library_dir)
    args.library_dir.mkdir(parents=True, exist_ok=True)
    qualifying: list[dict[str, Any]] = []

    for structure_hash in sorted(structure_index):
        entry = structure_index[structure_hash]
        task_ids = sorted(entry["tasks"])
        if len(task_ids) < 2:
            continue
        structure_path = args.library_dir / f"structure-{structure_hash}.json"
        document = {
            "schema": "lexigen-v26-factorized-structure-v1",
            "structure_sha256": structure_hash,
            "structure": entry["structure"],
            "minimum_depth": entry["minimum_depth"],
            "minimum_nodes": entry["minimum_nodes"],
            "discovery_task_ids": task_ids,
            "concrete_programs_by_task": {
                task_id: entry["tasks"][task_id] for task_id in task_ids
            },
        }
        write(structure_path, document)
        qualifying.append({
            "structure_sha256": structure_hash,
            "structure_file_sha256": sha256_file(structure_path),
            "minimum_depth": entry["minimum_depth"],
            "minimum_nodes": entry["minimum_nodes"],
            "discovery_task_count": len(task_ids),
            "discovery_task_ids": task_ids,
        })

    library = {
        "schema": "lexigen-v26-factorized-library-v1",
        "precommit_sha256": sha256_file(precommit_path),
        "discovery_task_count": len(expected_ids),
        "qualification_rule": precommit["discovery_rule"],
        "qualifying_structure_count": len(qualifying),
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "structures": qualifying,
    }
    write(args.library_output, library)

    report = {
        "schema": "lexigen-v26-discovery-report-v1",
        "precommit_sha256": sha256_file(precommit_path),
        "discovery_task_count": len(expected_ids),
        "completed_task_count": completed,
        "generator_invalid_task_count": generator_invalid,
        "totals": totals,
        "distinct_exact_structure_count": len(structure_index),
        "qualifying_structure_count": len(qualifying),
        "library_sha256": sha256_file(args.library_output),
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "heldout_transfer_demonstrated": False,
        "world_level_breakthrough": False,
        "task_summaries": task_summaries,
    }
    write(args.output, report)
    print(json.dumps({
        "completed_tasks": completed,
        "generator_invalid_tasks": generator_invalid,
        "raw_candidate_evaluations": totals["raw_candidate_evaluations"],
        "exact_concrete_programs": totals["exact_concrete_programs"],
        "distinct_exact_structures": len(structure_index),
        "qualifying_structures": len(qualifying),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
