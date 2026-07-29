from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRECOMMIT_COMMIT = "11791a6"
IMPLEMENTATION_COMMIT = "43f17d362a5cb1e87c60f83c0fbbe36e87c78c74"
WORKFLOW_RUN = 30463044724
ARTIFACT_DIGEST = "sha256:2760fa3b2d1419f3cc95ff6cdbdb867b6ff212ab45caaabf56609e6428847e85"
RIFT_RUN = 30463043864

FILES = (
    "README.md",
    "V24_PRECOMMIT.json",
    "runtime_v24.py",
    "generate_case_v24.py",
    "scan_one_v24.py",
    "aggregate_discovery_v24.py",
    "test_v24.py",
    "V24_DISCOVERY_REPORT.json",
    "V24_LIBRARY.json",
    "V24_INTEGRITY_AUDIT.json",
    "freeze_v24_evidence.py",
)


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    precommit = load("V24_PRECOMMIT.json")
    report = load("V24_DISCOVERY_REPORT.json")
    library = load("V24_LIBRARY.json")
    audit = load("V24_INTEGRITY_AUDIT.json")

    require(precommit["discovery_task_count"] == 64, "discovery denominator changed")
    require(precommit["validation_task_count"] == 32, "validation denominator changed")
    require(precommit["maximum_complete_programs_per_task"] == 272008, "candidate denominator changed")
    require(precommit["per_generation_timeout_seconds"] == 5, "timeout rule changed")
    require(not precommit["replacement_tasks_allowed"], "replacement enabled")
    require(not precommit["human_survivor_selection_allowed"], "human selection enabled")

    require(report["discovery_tasks"] == report["completed_tasks"] == 64, "incomplete discovery")
    require(report["generator_invalid_tasks"] == 0, "generator-invalid tasks changed")
    require(report["candidate_programs_tested"] == 17408512, "total candidate denominator changed")
    require(report["exact_complete_programs"] == 0, "exact-program verdict changed")
    require(report["tasks_with_any_exact_program"] == 0, "task success count changed")
    require(report["distinct_exact_structures"] == 0, "structure count changed")
    require(report["qualifying_structures"] == 0, "library qualification changed")
    require(report["validation_generators_imported"] == 0, "validation generator imported")
    require(not report["validation_outputs_opened"], "validation output opened")
    require(all(item["generation"]["timeouts"] == 0 for item in report["task_reports"]), "generation timeout count changed")
    require(all(item["generation"]["failures"] == 0 for item in report["task_reports"]), "generation failure count changed")

    require(library["qualifying_structure_count"] == 0, "nonempty library")
    require(library["structures"] == [], "library structures changed")
    require(library["validation_generators_imported"] == 0, "validation entered library")
    require(not library["validation_outputs_opened"], "validation output flag changed")
    require(audit["audit_verdict"] == "valid_negative_induced_language_discovery", "audit verdict changed")

    evidence = {
        "schema": "lexigen-v24-frozen-evidence-v1",
        "version": 24,
        "precommit_commit": PRECOMMIT_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "workflow_run_id": WORKFLOW_RUN,
        "workflow_artifact_digest": ARTIFACT_DIGEST,
        "rift_workflow_run_id": RIFT_RUN,
        "frozen_discovery_tasks": 64,
        "frozen_validation_tasks": 32,
        "completed_discovery_tasks": 64,
        "generator_invalid_tasks": 0,
        "generator_timeouts": 0,
        "candidate_programs_per_task": 272008,
        "candidate_programs_tested": 17408512,
        "exact_complete_programs": 0,
        "tasks_with_any_exact_program": 0,
        "distinct_exact_structures": 0,
        "qualifying_structures": 0,
        "library_sha256": sha(HERE / "V24_LIBRARY.json"),
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "heldout_transfer_demonstrated": False,
        "files": {name: sha(HERE / name) for name in FILES},
        "claim_boundary": {
            "generic_typed_relational_grammar_tested": True,
            "reusable_structure_discovered": False,
            "unrestricted_semantic_language_invention": False,
            "sealed_external_success": False,
            "external_discovery": False,
            "world_level_breakthrough": False,
        },
    }
    output = HERE / "V24_EVIDENCE.json"
    output.write_bytes((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    digest = sha(output)
    markdown = (
        "# Lexigen v24 frozen evidence\n\n"
        "- Frozen discovery identities: **64**\n"
        "- Frozen held-out identities left unopened: **32**\n"
        "- Generator-invalid tasks/timeouts: **0 / 0**\n"
        "- Concrete candidates tested: **17,408,512**\n"
        "- Exact complete programs: **0**\n"
        "- Reusable qualifying structures: **0**\n\n"
        "v24 is a valid negative discovery result. The generic typed relational grammar did not produce an exact program on any precommitted discovery identity, so the held-out validation set was not opened. It is not a language-invention breakthrough or external discovery.\n\n"
        f"Evidence SHA-256: `{digest}`\n"
    )
    (HERE / "EVIDENCE.md").write_bytes(markdown.encode("utf-8"))
    print(json.dumps({"evidence_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
