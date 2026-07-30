from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTRY = ROOT / "v19r5" / "V19R5_REGISTRY.json"
V30 = ROOT / "v30"
OUTPUT = HERE / "V32_PRECOMMIT.json"
TASK_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(task_id: str) -> str:
    return hashlib.sha256(
        f"lexigen-v32-full-grammar-transfer:{task_id}".encode()
    ).hexdigest()


def collect_used_ids() -> set[str]:
    used: set[str] = set()
    for path in ROOT.rglob("*.json"):
        if "v19r5" in path.parts or "v32" in path.parts:
            continue
        try:
            used.update(TASK_PATTERN.findall(path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return used


def main() -> None:
    registry = load(REGISTRY)
    manifest = load(V30 / "V30_GRAMMAR_MANIFEST.json")
    used = collect_used_ids()
    available = sorted(
        (task_id for task_id in registry["validation_task_ids"] if task_id not in used),
        key=lambda task_id: (rank(task_id), task_id),
    )
    selected = available[:64]
    if len(selected) != 64:
        raise RuntimeError("fewer than 64 fresh validation identities remain")

    precommit = {
        "schema": "lexigen-v32-full-grammar-transfer-precommit-v1",
        "source": {
            "v31_evidence_commit": "e321665639065141306688da16a3a867d67404c8",
            "v30_grammar_implementation_commit": "5aa2c43",
            "v30_recovery_evidence_commit": "5f1b8ca",
            "v30_precommit_sha256": sha256_file(V30 / "V30_PRECOMMIT.json"),
            "v30_grammar_manifest_sha256": sha256_file(V30 / "V30_GRAMMAR_MANIFEST.json"),
            "v30_grammar_script_sha256": sha256_file(V30 / "grammar_v30.py"),
            "candidate_sequence_sha256": manifest["candidate_sha256"],
            "structural_candidate_count": manifest["structural_candidate_count"],
            "memoized_recovery_scanner_sha256": "c59d6624e60959facc795ab466e027435f4e17466f7fdff126b788fa4ef315e4",
            "memoized_recovery_equivalence_sha256": "41cbd42cc26682bb8487d26daecd7b33417e80d2276774fde84106c3b1f4d5bb",
            "v30_full_denominator_report_sha256": "37954acab4f93b031665517629ac91499dc4891324c818a020e8be344d17874c",
        },
        "fresh_identity_selection": {
            "registry": "v19r5 validation_task_ids",
            "registry_sha256": sha256_file(REGISTRY),
            "previously_used_identity_count": len(set(registry["validation_task_ids"]) & used),
            "remaining_identity_count_before_selection": len(available),
            "rank_rule": "sha256(lexigen-v32-full-grammar-transfer:<task_id>), then task_id",
            "task_count": 64,
            "task_ids": selected,
        },
        "demonstration_gate": {
            "examples_per_task": 6,
            "generator_attempts_per_task": 16,
            "per_generation_timeout_seconds": 5,
            "seed_namespace": "lexigen-v32-demonstration",
            "seed_rule": "uint32(first_16_hex(sha256(lexigen-v32-demonstration:<task_id>:<attempt_index>)))",
            "replacement_tasks_allowed": False,
        },
        "enumeration": {
            "grammar": "frozen v30 source-induced typed grammar",
            "maximum_concrete_candidates_per_task": 500000,
            "candidate_order": [
                "structural_index",
                "ascending parameter assignment",
            ],
            "identity_outputs_rejected": True,
            "all_exact_candidates_retained": True,
            "selected_candidate_rule": "minimum exact candidate under the frozen order",
            "memoized_evaluator": "exact shared-subtree evaluator proven byte-identical on v30 equivalence task",
            "candidate_or_semantic_pruning_after_generation": False,
        },
        "immediate_fresh_gate": {
            "runs_only_after_at_least_one_demonstration_exact_candidate": True,
            "selected_candidate_frozen_by_rule": True,
            "case_count": 100,
            "case_indices": {"first": 0, "last": 99},
            "seed_namespace": "lexigen-v32-fresh",
            "seed_rule": "uint32(first_16_hex(sha256(lexigen-v32-fresh:<task_id>:<case_index>)))",
            "generator_attempts_per_case": 1,
            "per_generation_timeout_seconds": 5,
            "replacement_cases_allowed": False,
            "primary_runtime": "frozen v25 eval_ast",
            "independent_runtime": "independent implementation of all 13 source-grammar operators",
            "both_runtimes_must_match_target": True,
            "all_cases_must_pass": True,
        },
        "claim_boundary": {
            "one_new_fresh_pass": "second fresh-validated public identity solved by the source-induced grammar architecture",
            "two_or_more_new_fresh_passes": "multi-identity public transfer of the source-induced grammar architecture",
            "outside_human_reproduction": False,
            "world_level_breakthrough": False,
            "reason": "public generator family and human-chosen protocol remain",
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
        "previously_used_identity_count": precommit["fresh_identity_selection"]["previously_used_identity_count"],
        "first_task_ids": selected[:5],
        "last_task_ids": selected[-5:],
        "structural_candidate_count": manifest["structural_candidate_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
