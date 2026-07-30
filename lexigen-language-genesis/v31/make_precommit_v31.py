from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTRY = ROOT / "v19r5" / "V19R5_REGISTRY.json"
OUTPUT = HERE / "V31_PRECOMMIT.json"
TASK_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(task_id: str) -> str:
    return hashlib.sha256(
        f"lexigen-v31-validated-motif-recurrence:{task_id}".encode()
    ).hexdigest()


def collect_used_ids() -> set[str]:
    used: set[str] = set()
    for path in ROOT.rglob("*.json"):
        if "v19r5" in path.parts or "v31" in path.parts:
            continue
        try:
            used.update(TASK_PATTERN.findall(path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return used


def main() -> None:
    registry = load(REGISTRY)
    used = collect_used_ids()
    available = sorted(
        (task_id for task_id in registry["validation_task_ids"] if task_id not in used),
        key=lambda task_id: (rank(task_id), task_id),
    )
    selected = available[:64]
    if len(selected) != 64:
        raise RuntimeError("fewer than 64 fresh validation identities remain")

    source_report = ROOT / "v30-956-fresh" / "V30_956_FRESH_REPORT.json"
    source_evidence = ROOT / "v30-956-fresh" / "V30_956_FRESH_EVIDENCE.json"
    source_precommit = ROOT / "v30-956-fresh" / "V30_956_FRESH_PRECOMMIT.json"
    precommit = {
        "schema": "lexigen-v31-validated-motif-recurrence-precommit-v1",
        "source": {
            "task_id": "9565186b",
            "commit": "5796d91",
            "fresh_precommit_sha256": sha256_file(source_precommit),
            "fresh_report_sha256": sha256_file(source_report),
            "fresh_evidence_sha256": sha256_file(source_evidence),
            "fresh_cases_passed": 1000,
            "concrete_program_sha256": "5bc7ccf73fa03eee67d7b5b894397ced698720af28d145bca38fe284f7186cb8",
        },
        "motif": {
            "name": "background_preserving_foreground_recolor",
            "ast": {
                "op": "paint",
                "grid": {"op": "input_grid"},
                "points": {"op": "non_background_points"},
                "colour": {"op": "param_color", "name": "c0"},
            },
            "candidate_colors": list(range(10)),
            "candidate_order": "ascending color value",
        },
        "fresh_identity_selection": {
            "registry": "v19r5 validation_task_ids",
            "registry_sha256": sha256_file(REGISTRY),
            "previously_used_identity_count": len(set(registry["validation_task_ids"]) & used),
            "remaining_identity_count_before_selection": len(available),
            "rank_rule": "sha256(lexigen-v31-validated-motif-recurrence:<task_id>), then task_id",
            "task_count": 64,
            "task_ids": selected,
        },
        "demonstration_gate": {
            "examples_per_task": 6,
            "generator_attempts_per_task": 16,
            "per_generation_timeout_seconds": 5,
            "seed_namespace": "lexigen-v31-demonstration",
            "candidate_count_per_task": 10,
            "identity_outputs_rejected": True,
            "success_rule": "exactly one color candidate must match all six demonstrations",
            "ambiguous_matches_rejected": True,
            "replacement_tasks_allowed": False,
        },
        "immediate_fresh_gate": {
            "runs_only_after_unique_demonstration_match": True,
            "case_count": 100,
            "case_indices": {"first": 0, "last": 99},
            "seed_namespace": "lexigen-v31-fresh",
            "generator_attempts_per_case": 1,
            "per_generation_timeout_seconds": 5,
            "replacement_cases_allowed": False,
            "primary_runtime": "frozen v25 eval_ast",
            "independent_runtime": "direct background-preserving foreground recolor",
            "relational_verifier_required": True,
            "all_cases_must_pass": True,
        },
        "claim_boundary": {
            "one_new_fresh_pass": "repeated public task-level transfer across source and one new identity",
            "two_or_more_new_fresh_passes": "multi-identity recurrence of the validated motif",
            "outside_human_reproduction": False,
            "world_level_breakthrough": False,
            "reason": "public generator family and human-chosen evaluation protocol remain",
        },
        "post_result_edits_allowed": False,
        "task_replacement_allowed": False,
    }
    OUTPUT.write_bytes(
        (json.dumps(precommit, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps({
        "precommit_sha256": sha256_file(OUTPUT),
        "selected_task_count": len(selected),
        "remaining_before_selection": len(available),
        "first_task_ids": selected[:5],
        "last_task_ids": selected[-5:],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
