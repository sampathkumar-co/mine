from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V30_PRECOMMIT.json"
MANIFEST = HERE / "V30_GRAMMAR_MANIFEST.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit = load(PRECOMMIT)
    manifest = load(MANIFEST)
    expected_ids = list(precommit["validation_task_ids"])
    reports: dict[str, dict[str, Any]] = {}
    for task_id in expected_ids:
        path = args.reports_root / f"task-{task_id}.json"
        if not path.exists():
            raise RuntimeError(f"missing report for frozen task {task_id}")
        report = load(path)
        if report.get("schema") != "lexigen-v30-task-scan-v1":
            raise RuntimeError(f"invalid report schema for {task_id}")
        if report.get("task_id") != task_id:
            raise RuntimeError(f"task identity mismatch for {task_id}")
        if report.get("precommit_sha256") != sha256_file(PRECOMMIT):
            raise RuntimeError(f"precommit mismatch for {task_id}")
        if report.get("grammar_manifest_sha256") != sha256_file(MANIFEST):
            raise RuntimeError(f"grammar manifest mismatch for {task_id}")
        reports[task_id] = report

    totals = {
        "accepted_examples": 0,
        "generation_attempts": 0,
        "generation_timeouts": 0,
        "generation_failures": 0,
        "concrete_candidates_tested": 0,
        "runtime_invalid_candidates": 0,
        "identity_candidates_rejected": 0,
        "exact_candidates": 0,
    }
    status_counts = {
        "generator_invalid": 0,
        "no_program": 0,
        "exact_program_found": 0,
    }
    successes: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task_id in expected_ids:
        report = reports[task_id]
        totals["accepted_examples"] += int(report["accepted_examples"])
        generation = report["generation"]
        totals["generation_attempts"] += int(generation["attempts"])
        totals["generation_timeouts"] += int(generation["timeouts"])
        totals["generation_failures"] += int(generation["failures"])
        summary = {
            "task_id": task_id,
            "status": report["status"],
            "report_sha256": sha256_file(args.reports_root / f"task-{task_id}.json"),
            "accepted_examples": int(report["accepted_examples"]),
        }
        if report["status"] == "generator_invalid":
            status_counts["generator_invalid"] += 1
            summary["exact_candidate_count"] = 0
            task_summaries.append(summary)
            continue

        enumeration = report["enumeration"]
        exact_count = int(enumeration["exact_candidate_count"])
        totals["concrete_candidates_tested"] += int(enumeration["concrete_candidates_tested"])
        totals["runtime_invalid_candidates"] += int(enumeration["runtime_invalid_candidates"])
        totals["identity_candidates_rejected"] += int(enumeration["identity_candidates_rejected"])
        totals["exact_candidates"] += exact_count
        summary["exact_candidate_count"] = exact_count
        summary["candidate_cap_reached"] = bool(enumeration["candidate_cap_reached"])
        if exact_count:
            status_counts["exact_program_found"] += 1
            selected = enumeration["selected_candidate"]
            successes.append({
                "task_id": task_id,
                "exact_candidate_count": exact_count,
                "selected_candidate": selected,
            })
        else:
            status_counts["no_program"] += 1
        task_summaries.append(summary)

    report = {
        "schema": "lexigen-v30-validation-report-v1",
        "precommit_sha256": sha256_file(PRECOMMIT),
        "grammar_manifest_sha256": sha256_file(MANIFEST),
        "validation_task_count": len(expected_ids),
        "completed_report_count": len(reports),
        "status_counts": status_counts,
        "totals": totals,
        "heldout_synthesis_event_count": len(successes),
        "heldout_synthesis_demonstrated": len(successes) >= 1,
        "repeated_heldout_transfer_demonstrated": len(successes) >= 2,
        "selected_candidates_frozen": bool(successes),
        "fresh_validation_completed": False,
        "independent_runtime_completed": False,
        "verifier_cosynthesis_completed": False,
        "world_level_breakthrough": False,
        "successes": successes,
        "task_summaries": task_summaries,
        "claim_boundary": precommit["claim_boundary"],
    }
    write(args.output, report)
    print(json.dumps({
        "validation_tasks": len(expected_ids),
        "candidates_tested": totals["concrete_candidates_tested"],
        "exact_tasks": len(successes),
        "status_counts": status_counts,
        "heldout_synthesis_demonstrated": report["heldout_synthesis_demonstrated"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
