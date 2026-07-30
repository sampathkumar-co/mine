from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V32_PRECOMMIT.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_report_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("task-*.json")
        if path.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit = load(PRECOMMIT)
    expected = list(precommit["fresh_identity_selection"]["task_ids"])
    expected_set = set(expected)
    paths = task_report_paths(args.reports_root)
    reports: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        report = load(path)
        if report.get("schema") != "lexigen-v32-task-report-v1":
            continue
        task_id = str(report["task_id"])
        if task_id in reports:
            raise RuntimeError(f"duplicate report for {task_id}")
        reports[task_id] = (path, report)

    actual_set = set(reports)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise RuntimeError(f"report denominator mismatch: missing={missing}, extra={extra}")
    if len(reports) != 64:
        raise RuntimeError("v32 requires exactly 64 reports")

    expected_precommit_sha = sha256_file(PRECOMMIT)
    statuses = Counter()
    totals = Counter()
    summaries: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for task_id in expected:
        path, report = reports[task_id]
        if report["precommit_sha256"] != expected_precommit_sha:
            raise RuntimeError(f"precommit hash mismatch for {task_id}")
        status = str(report["status"])
        if status not in {"generator_invalid", "no_program", "fresh_fail", "fresh_pass"}:
            raise RuntimeError(f"invalid status for {task_id}: {status}")
        statuses[status] += 1
        totals["accepted_examples"] += int(report["accepted_examples"])
        generation = report["generation"]
        totals["generation_attempts"] += int(generation["attempts"])
        totals["generation_timeouts"] += int(generation["timeouts"])
        totals["generation_failures"] += int(generation["failures"])

        enumeration = report["enumeration"]
        exact_count = 0
        selected = None
        if enumeration is not None:
            totals["concrete_candidates_tested"] += int(enumeration["concrete_candidates_tested"])
            totals["runtime_invalid_candidates"] += int(enumeration["runtime_invalid_candidates"])
            totals["identity_candidates_rejected"] += int(enumeration["identity_candidates_rejected"])
            exact_count = int(enumeration["exact_candidate_count"])
            totals["exact_demonstration_candidates"] += exact_count
            selected = enumeration["selected_candidate"]

        fresh = report["fresh_gate"]
        if fresh is not None:
            fresh_totals = fresh["totals"]
            for key, value in fresh_totals.items():
                totals[f"fresh_{key}"] += int(value)
        summary = {
            "task_id": task_id,
            "status": status,
            "accepted_examples": int(report["accepted_examples"]),
            "exact_candidate_count": exact_count,
            "selected_candidate": selected,
            "fresh_passed": bool(fresh and fresh["passed"]),
            "report_sha256": sha256_file(path),
        }
        summaries.append(summary)
        if status in {"fresh_pass", "fresh_fail"}:
            successes.append({
                "task_id": task_id,
                "status": status,
                "exact_candidate_count": exact_count,
                "selected_candidate": selected,
                "fresh_totals": fresh["totals"] if fresh else None,
            })

    fresh_pass_count = int(statuses["fresh_pass"])
    report = {
        "schema": "lexigen-v32-full-grammar-transfer-report-v1",
        "precommit_sha256": expected_precommit_sha,
        "validation_task_count": 64,
        "completed_report_count": len(reports),
        "status_counts": {
            key: int(statuses[key])
            for key in ("generator_invalid", "no_program", "fresh_fail", "fresh_pass")
        },
        "totals": {key: int(value) for key, value in sorted(totals.items())},
        "fresh_pass_count": fresh_pass_count,
        "second_fresh_validated_identity_demonstrated": fresh_pass_count >= 1,
        "multi_identity_public_transfer_demonstrated": fresh_pass_count >= 2,
        "outside_human_reproduction_completed": False,
        "world_level_breakthrough": False,
        "successes": successes,
        "task_summaries": summaries,
    }
    write(args.output, report)
    print(json.dumps({
        "validation_tasks": 64,
        "fresh_pass_count": fresh_pass_count,
        "status_counts": report["status_counts"],
        "second_fresh_validated_identity": report["second_fresh_validated_identity_demonstrated"],
        "world_level_breakthrough": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
