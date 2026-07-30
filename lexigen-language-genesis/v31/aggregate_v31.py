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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = HERE / "V31_PRECOMMIT.json"
    precommit = load(precommit_path)
    precommit_sha = sha256_file(precommit_path)
    expected_ids = list(precommit["fresh_identity_selection"]["task_ids"])
    expected_set = set(expected_ids)
    paths = sorted(args.reports_root.glob("task-*.json"))
    reports: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        report = load(path)
        task_id = str(report.get("task_id", ""))
        if task_id in reports:
            raise RuntimeError(f"duplicate report for {task_id}")
        if report.get("schema") != "lexigen-v31-task-report-v1":
            raise RuntimeError(f"invalid report schema: {path}")
        if report.get("precommit_sha256") != precommit_sha:
            raise RuntimeError(f"precommit mismatch: {path}")
        reports[task_id] = (path, report)
    if set(reports) != expected_set:
        missing = sorted(expected_set - set(reports))
        extra = sorted(set(reports) - expected_set)
        raise RuntimeError(f"report denominator mismatch missing={missing} extra={extra}")

    status_counts = {
        "generator_invalid": 0,
        "no_program": 0,
        "ambiguous": 0,
        "fresh_fail": 0,
        "fresh_pass": 0,
    }
    totals = {
        "accepted_examples": 0,
        "generation_attempts": 0,
        "generation_timeouts": 0,
        "generation_failures": 0,
        "candidate_evaluations": 0,
        "exact_demonstration_candidates": 0,
        "fresh_cases_requested": 0,
        "fresh_cases_generated": 0,
        "fresh_cases_passed": 0,
        "fresh_generation_timeouts": 0,
        "fresh_generation_errors": 0,
        "fresh_runtime_disagreements": 0,
        "fresh_target_mismatches": 0,
        "fresh_verifier_rejections": 0,
    }
    summaries: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for task_id in expected_ids:
        path, report = reports[task_id]
        status = str(report["status"])
        if status not in status_counts:
            raise RuntimeError(f"unknown status {status}: {path}")
        status_counts[status] += 1
        generation = report["generation"]
        totals["accepted_examples"] += int(report["accepted_examples"])
        totals["generation_attempts"] += int(generation["attempts"])
        totals["generation_timeouts"] += int(generation["timeouts"])
        totals["generation_failures"] += int(generation["failures"])
        totals["candidate_evaluations"] += int(report["candidate_count"])
        totals["exact_demonstration_candidates"] += len(report["exact_colors"])
        fresh = report["fresh_gate"]
        if fresh is not None:
            fresh_totals = fresh["totals"]
            totals["fresh_cases_requested"] += int(fresh_totals["requested_cases"])
            totals["fresh_cases_generated"] += int(fresh_totals["generated_cases"])
            totals["fresh_cases_passed"] += int(fresh_totals["passed_cases"])
            totals["fresh_generation_timeouts"] += int(fresh_totals["generation_timeouts"])
            totals["fresh_generation_errors"] += int(fresh_totals["generation_errors"])
            totals["fresh_runtime_disagreements"] += int(fresh_totals["runtime_disagreements"])
            totals["fresh_target_mismatches"] += int(fresh_totals["target_mismatches"])
            totals["fresh_verifier_rejections"] += int(fresh_totals["verifier_rejections"])
        summary = {
            "task_id": task_id,
            "status": status,
            "accepted_examples": report["accepted_examples"],
            "exact_colors": report["exact_colors"],
            "selected_color": report["selected_color"],
            "report_sha256": sha256_file(path),
        }
        if fresh is not None:
            summary["fresh_gate_passed"] = bool(fresh["passed"])
            summary["fresh_report_sha256"] = hashlib.sha256(
                canonical_fresh(fresh).encode("utf-8")
            ).hexdigest()
        summaries.append(summary)
        if status == "fresh_pass":
            successes.append({
                "task_id": task_id,
                "color": int(report["selected_color"]),
                "demonstration_sha256": report["demonstration_sha256"],
                "report_sha256": sha256_file(path),
                "fresh_case_count": int(fresh["case_count"]),
                "fresh_cases_passed": int(fresh["totals"]["passed_cases"]),
            })

    repeated = len(successes) >= 1
    multi_identity = len(successes) >= 2
    output = {
        "schema": "lexigen-v31-validated-motif-recurrence-report-v1",
        "precommit_sha256": precommit_sha,
        "source_task_id": precommit["source"]["task_id"],
        "source_fresh_cases_passed": precommit["source"]["fresh_cases_passed"],
        "validation_task_count": len(expected_ids),
        "completed_report_count": len(reports),
        "status_counts": status_counts,
        "totals": totals,
        "fresh_pass_count": len(successes),
        "successes": successes,
        "repeated_public_task_level_transfer_demonstrated": repeated,
        "multi_identity_motif_recurrence_demonstrated": multi_identity,
        "outside_human_reproduction_completed": False,
        "world_level_breakthrough": False,
        "task_summaries": summaries,
    }
    write(args.output, output)
    print(json.dumps({
        "validation_tasks": len(expected_ids),
        "status_counts": status_counts,
        "fresh_pass_count": len(successes),
        "repeated_transfer": repeated,
        "world_level_breakthrough": False,
    }, sort_keys=True))


def canonical_fresh(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
