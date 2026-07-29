from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V24_DISCOVERY_REPORT.json")
    parser.add_argument("--library-output", type=Path, default=HERE / "V24_LIBRARY.json")
    parser.add_argument("--library-dir", type=Path, default=HERE / "library")
    args = parser.parse_args()
    precommit_path = HERE / "V24_PRECOMMIT.json"
    precommit = load(precommit_path)
    expected_ids = list(precommit["discovery_task_ids"])
    report_paths = sorted(args.reports_root.rglob("*.json"))
    reports = [load(path) for path in report_paths]
    by_task: dict[str, dict[str, Any]] = {}
    for report in reports:
        if report.get("schema") != "lexigen-v24-task-scan-v1":
            continue
        if report.get("split") != "discovery":
            raise RuntimeError("validation report entered discovery aggregation")
        task_id = str(report["task_id"])
        if task_id in by_task:
            raise RuntimeError(f"duplicate task report: {task_id}")
        by_task[task_id] = report
    if sorted(by_task) != sorted(expected_ids):
        missing = sorted(set(expected_ids) - set(by_task))
        extra = sorted(set(by_task) - set(expected_ids))
        raise RuntimeError(f"report identity mismatch: missing={missing}, extra={extra}")

    structure_index: dict[str, dict[str, Any]] = {}
    completed = invalid_tasks = 0
    total_candidates = total_runtime_invalid = total_exact = 0
    tasks_with_exact = 0
    for task_id in expected_ids:
        report = by_task[task_id]
        if report["status"] == "generator_invalid":
            invalid_tasks += 1
            continue
        completed += 1
        total_candidates += int(report["candidate_programs_tested"])
        total_runtime_invalid += int(report["runtime_invalid_candidates"])
        total_exact += int(report["exact_complete_programs"])
        tasks_with_exact += int(report["exact_complete_programs"] > 0)
        for item in report["exact_structures"]:
            structure_hash = str(item["structure_sha256"])
            entry = structure_index.setdefault(
                structure_hash,
                {
                    "structure_sha256": structure_hash,
                    "structure": item["structure"],
                    "discovery_tasks": {},
                },
            )
            if entry["structure"] != item["structure"]:
                raise RuntimeError("structure hash collision")
            entry["discovery_tasks"][task_id] = item["arguments"]

    qualifying = []
    args.library_dir.mkdir(parents=True, exist_ok=True)
    for old in args.library_dir.glob("*.json"):
        old.unlink()
    for structure_hash in sorted(structure_index):
        entry = structure_index[structure_hash]
        task_ids = sorted(entry["discovery_tasks"])
        if len(task_ids) < 2:
            continue
        structure_file = args.library_dir / f"structure-{structure_hash}.json"
        structure_document = {
            "schema": "lexigen-v24-induced-structure-v1",
            "structure_sha256": structure_hash,
            "structure": entry["structure"],
        }
        write(structure_file, structure_document)
        qualifying.append({
            "structure_sha256": structure_hash,
            "structure_file_sha256": sha(structure_file),
            "structure": entry["structure"],
            "discovery_task_count": len(task_ids),
            "discovery_task_ids": task_ids,
            "arguments_by_task": {
                task_id: entry["discovery_tasks"][task_id]
                for task_id in task_ids
            },
        })

    library = {
        "schema": "lexigen-v24-induced-library-v1",
        "precommit_sha256": sha(precommit_path),
        "discovery_task_count": len(expected_ids),
        "generator_invalid_tasks": invalid_tasks,
        "qualification_rule": "exact on at least two distinct frozen discovery identities",
        "qualifying_structure_count": len(qualifying),
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "structures": qualifying,
    }
    write(args.library_output, library)
    discovery_report = {
        "schema": "lexigen-v24-discovery-report-v1",
        "precommit_sha256": sha(precommit_path),
        "discovery_tasks": len(expected_ids),
        "completed_tasks": completed,
        "generator_invalid_tasks": invalid_tasks,
        "candidate_programs_tested": total_candidates,
        "runtime_invalid_candidates": total_runtime_invalid,
        "exact_complete_programs": total_exact,
        "tasks_with_any_exact_program": tasks_with_exact,
        "distinct_exact_structures": len(structure_index),
        "qualifying_structures": len(qualifying),
        "library_sha256": sha(args.library_output),
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "transfer_demonstrated": False,
        "world_level_breakthrough": False,
        "task_reports": [by_task[task_id] for task_id in expected_ids],
    }
    write(args.output, discovery_report)
    print(json.dumps({
        "completed_tasks": completed,
        "generator_invalid_tasks": invalid_tasks,
        "tasks_with_any_exact_program": tasks_with_exact,
        "distinct_exact_structures": len(structure_index),
        "qualifying_structures": len(qualifying),
        "library_sha256": discovery_report["library_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
