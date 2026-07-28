from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def target_filename(name: str) -> str:
    return f"{name.replace('(', '_').replace(')', '').replace(',', '_')}.json"


def normalize_result(result: dict[str, object]) -> bool:
    changed = False
    method = str(result.get("method") or "")
    greedy_best = result.get("greedy_best_blocks")
    goal = int(result.get("goal_blocks") or 0)
    local_runs = result.get("local_runs") or []

    local_valid = any(
        isinstance(row, dict) and bool(row.get("valid"))
        for row in local_runs
    )
    greedy_succeeded = isinstance(greedy_best, int) and greedy_best <= goal

    if method == "generic_greedy" and local_valid and not greedy_succeeded:
        result["method"] = "stochastic_fixed_budget"
        changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = summary.get("results")
    if not isinstance(results, list):
        raise TypeError("summary results must be a list")

    changed_targets: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each result must be an object")
        if normalize_result(result):
            changed_targets.append(str(result["target"]["name"]))
        target_path = args.summary.parent / target_filename(str(result["target"]["name"]))
        target_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    root = Path(__file__).resolve().parent
    summary.setdefault("code_hashes", {})["normalize_evidence.py"] = hashlib.sha256(
        (root / "normalize_evidence.py").read_bytes()
    ).hexdigest()
    summary["evidence_normalization"] = {
        "rule": "correct_local_search_method_attribution_only",
        "changed_targets": changed_targets,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["evidence_normalization"], indent=2))


if __name__ == "__main__":
    main()
