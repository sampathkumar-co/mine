from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precommit = load(Path(__file__).resolve().parent / "V21_PRECOMMIT.json")
    reports = [load(path) for path in sorted(args.results.rglob("*.json"))]
    by_task = {report["task_id"]: report for report in reports}
    expected = set(precommit["selected_discovery_task_ids"])
    if set(by_task) != expected:
        raise RuntimeError(f"task denominator mismatch: {sorted(expected-set(by_task))}")
    index = {}
    for report in reports:
        for entry in report.get("productions", []):
            digest = entry["production_sha256"]
            item = index.setdefault(digest, {
                "production": entry["production"],
                "tasks": {},
            })
            item["tasks"][report["task_id"]] = entry["arguments"]
    minimum = int(precommit["minimum_distinct_discovery_tasks_per_production"])
    survivors = []
    for digest, item in sorted(index.items()):
        if len(item["tasks"]) >= minimum:
            survivors.append({
                "production_sha256": digest,
                "production": item["production"],
                "distinct_task_count": len(item["tasks"]),
                "tasks": item["tasks"],
            })
    result = {
        "schema": "lexigen-v21-factorized-discovery-report-v1",
        "precommit_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent / "V21_PRECOMMIT.json").read_bytes()
        ).hexdigest(),
        "task_count": len(reports),
        "generator_invalid_count": sum(r["status"] != "completed" for r in reports),
        "tasks_with_any_exact_program": sum(bool(r.get("productions")) for r in reports),
        "total_exact_complete_programs": sum(r.get("exact_complete_programs", 0) for r in reports),
        "unique_production_structures": len(index),
        "minimum_distinct_tasks": minimum,
        "surviving_factorized_productions": survivors,
        "survivor_count": len(survivors),
        "claim_boundary": (
            "Public development discovery only. Validation identities and outputs remain unopened; "
            "this is not sealed external evidence or a world-level breakthrough."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "task_count": result["task_count"],
        "tasks_with_any_exact_program": result["tasks_with_any_exact_program"],
        "survivor_count": result["survivor_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
