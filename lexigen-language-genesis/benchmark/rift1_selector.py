from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rift1 import World, build_cases, execute_artifact, solves, synthesize

MECHANISMS = ("closure", "trajectory_union", "two_cycle_canonical")


def build_library() -> dict[str, dict[str, Any]]:
    return {
        mechanism: synthesize(build_cases(mechanism, range(4, 7), replicas=2))
        for mechanism in MECHANISMS
    }


def description_length(artifact: dict[str, Any]) -> tuple[int, int, str]:
    """Frozen MDL ordering: instruction count, canonical byte count, artifact hash."""
    program = artifact["program"]
    encoded = json.dumps(program, sort_keys=True, separators=(",", ":"))
    return len(program), len(encoded.encode("utf-8")), str(artifact["name"])


def select_artifact(
    library: dict[str, dict[str, Any]], demonstrations: list[World]
) -> tuple[str, dict[str, Any], dict[str, bool], dict[str, tuple[int, int, str]]]:
    # Selection uses execution against examples only. It never reads `case.mechanism`.
    # Multiple languages may be extensionally correct on simple examples. The frozen
    # MDL rule chooses the smallest executable semantics consistent with all evidence.
    matches = {
        name: solves(artifact, demonstrations)
        for name, artifact in sorted(library.items())
    }
    passing = [name for name, matched in matches.items() if matched]
    if not passing:
        raise RuntimeError("no library artifact matches the demonstrations")
    scores = {name: description_length(library[name]) for name in passing}
    selected = min(passing, key=lambda name: scores[name])
    return selected, library[selected], matches, scores


def run(output_dir: Path) -> dict[str, Any]:
    library = build_library()
    episodes: list[dict[str, Any]] = []

    for hidden_mechanism in MECHANISMS:
        demonstrations = build_cases(hidden_mechanism, [5, 6], replicas=1)
        transfer = build_cases(hidden_mechanism, range(8, 14), replicas=2)
        selected_name, artifact, match_vector, mdl_scores = select_artifact(library, demonstrations)

        correct = 0
        records = []
        for case in transfer:
            predicted = execute_artifact(artifact, case.step, case.seed)
            expected = case.independently_verified_target()
            is_correct = predicted == expected
            correct += int(is_correct)
            records.append(
                {
                    "name": case.name,
                    "surface": case.surface,
                    "correct": is_correct,
                }
            )
        accuracy = correct / len(transfer)
        if accuracy != 1.0:
            raise AssertionError(f"selected artifact failed hidden transfer for {hidden_mechanism}")
        if selected_name != hidden_mechanism:
            raise AssertionError("MDL selector chose the wrong library entry")

        episodes.append(
            {
                "hidden_mechanism_used_only_for_scoring": hidden_mechanism,
                "selected_library_entry": selected_name,
                "match_vector": match_vector,
                "mdl_scores_for_passing_artifacts": {
                    name: list(score) for name, score in mdl_scores.items()
                },
                "transfer_accuracy": accuracy,
                "records": records,
            }
        )

    report = {
        "benchmark": "RIFT-1 unlabeled selector",
        "status": "MDL library selection by demonstrations; no breakthrough claim",
        "selection_rule": "minimum instruction count, then canonical byte count, then artifact name",
        "negative_result_preserved": (
            "uniqueness selection failed because richer history-based artifacts also solve closure examples"
        ),
        "library": library,
        "episodes": episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rift1-selector-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "episode_count": len(episodes),
        "all_transfer_exact": all(e["transfer_accuracy"] == 1.0 for e in episodes),
        "selections": [e["selected_library_entry"] for e in episodes],
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift1"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
