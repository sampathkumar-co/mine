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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = HERE / "V29_PRECOMMIT.json"
    library_path = HERE / "V29_TEMPLATE_LIBRARY.json"
    precommit = load(precommit_path)
    library = load(library_path)
    expected_ids = list(precommit["validation_task_ids"])
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(args.reports_root.rglob("*.json")):
        report = load(path)
        if report.get("schema") != "lexigen-v29-validation-task-report-v1":
            continue
        task_id = str(report["task_id"])
        if task_id in found:
            raise RuntimeError(f"duplicate validation report: {task_id}")
        found[task_id] = (path, report)
    if set(found) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(found))
        extra = sorted(set(found) - set(expected_ids))
        raise RuntimeError(f"validation denominator changed: missing={missing}, extra={extra}")

    totals = {
        "accepted_examples": 0,
        "generation_attempts": 0,
        "generation_failures": 0,
        "generation_timeouts": 0,
        "candidate_instantiations_tested": 0,
        "exact_instantiations": 0,
    }
    status_counts = {
        "generator_invalid": 0,
        "no_program": 0,
        "ambiguous": 0,
        "unique_exact": 0,
    }
    task_summaries: list[dict[str, Any]] = []
    unique_successes: list[dict[str, Any]] = []

    for task_id in expected_ids:
        path, report = found[task_id]
        if report["precommit_sha256"] != sha256_file(precommit_path):
            raise RuntimeError(f"precommit binding failed: {task_id}")
        if report["template_library_sha256"] != sha256_file(library_path):
            raise RuntimeError(f"template-library binding failed: {task_id}")
        if report.get("replacement_used") or report.get("human_survivor_selection_used"):
            raise RuntimeError(f"protocol violation: {task_id}")
        status = str(report["status"])
        if status not in status_counts:
            raise RuntimeError(f"unknown task status: {task_id}={status}")
        status_counts[status] += 1
        generation = report["generation"]
        totals["accepted_examples"] += int(report["accepted_examples"])
        totals["generation_attempts"] += int(generation["attempts"])
        totals["generation_failures"] += int(generation["failures"])
        totals["generation_timeouts"] += int(generation["timeouts"])
        totals["candidate_instantiations_tested"] += int(report["candidate_instantiations_tested"])
        totals["exact_instantiations"] += int(report["exact_instantiation_count"])
        summary = {
            "task_id": task_id,
            "status": status,
            "report_sha256": sha256_file(path),
            "accepted_examples": report["accepted_examples"],
            "candidate_instantiations_tested": report["candidate_instantiations_tested"],
            "exact_instantiation_count": report["exact_instantiation_count"],
        }
        if status != "generator_invalid" and int(report["candidate_instantiations_tested"]) != 160:
            raise RuntimeError(f"candidate denominator changed: {task_id}")
        if status == "unique_exact":
            exact = report["exact_instantiations"]
            if len(exact) != 1:
                raise RuntimeError(f"unique status mismatch: {task_id}")
            success = {
                "task_id": task_id,
                "demonstration_sha256": report["demonstration_sha256"],
                "template_sha256": exact[0]["template_sha256"],
                "operator_arguments": exact[0]["operator_arguments"],
                "colour_arguments": exact[0]["colour_arguments"],
                "concrete_program_sha256": exact[0]["concrete_program_sha256"],
                "report_sha256": sha256_file(path),
            }
            unique_successes.append(success)
            summary["unique_instantiation"] = success
        task_summaries.append(summary)

    result = {
        "schema": "lexigen-v29-validation-report-v1",
        "precommit_sha256": sha256_file(precommit_path),
        "template_library_sha256": sha256_file(library_path),
        "validation_task_count": len(expected_ids),
        "status_counts": status_counts,
        "totals": totals,
        "unique_success_count": len(unique_successes),
        "unique_successes": unique_successes,
        "task_summaries": task_summaries,
        "validation_generators_imported": len(expected_ids),
        "validation_outputs_opened": True,
        "heldout_template_match_demonstrated": bool(unique_successes),
        "fresh_validation_completed": False,
        "independent_runtime_completed": False,
        "verifier_cosynthesis_completed": False,
        "world_level_breakthrough": False,
        "claim_boundary": precommit["claim_boundary"],
    }
    write(args.output, result)
    print(json.dumps({
        "validation_tasks": len(expected_ids),
        "status_counts": status_counts,
        "candidates_tested": totals["candidate_instantiations_tested"],
        "unique_successes": len(unique_successes),
        "heldout_template_match_demonstrated": bool(unique_successes),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
