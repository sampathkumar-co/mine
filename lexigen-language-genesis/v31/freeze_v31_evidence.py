from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V31_PRECOMMIT.json"
GITHUB_REPORT = HERE / "V31_REPORT.json"
LOCAL_REPORT = HERE / "V31_LOCAL_REPORT.json"
AUDIT = HERE / "V31_LOCAL_AUDIT.json"
EVIDENCE = HERE / "V31_EVIDENCE.json"
MARKDOWN = HERE / "EVIDENCE.md"

EXPECTED_PRECOMMIT = "73a064c3b90b1bab4f7a979fd205482208a41f1377aeb47bb549885ff186b203"
EXPECTED_GITHUB = "23d6ad3d55804efa5b33b90d4416d8bd382edb01a5f6180693f8e94c14b9a85f"
EXPECTED_LOCAL = "04ab66e291f660458d0bb389adf90c54fba98a6db7c9a9ec56e2023b6de37c1e"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def verdict_map(report: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        row["task_id"]: (
            row["status"],
            tuple(row["exact_colors"]),
            row["selected_color"],
        )
        for row in report["task_summaries"]
    }


def main() -> None:
    assert sha256_file(PRECOMMIT) == EXPECTED_PRECOMMIT
    assert sha256_file(GITHUB_REPORT) == EXPECTED_GITHUB
    assert sha256_file(LOCAL_REPORT) == EXPECTED_LOCAL
    precommit = load(PRECOMMIT)
    github = load(GITHUB_REPORT)
    local = load(LOCAL_REPORT)

    expected_status = {
        "ambiguous": 0,
        "fresh_fail": 0,
        "fresh_pass": 0,
        "generator_invalid": 0,
        "no_program": 64,
    }
    for report in (github, local):
        assert report["validation_task_count"] == 64
        assert report["completed_report_count"] == 64
        assert report["status_counts"] == expected_status
        assert report["fresh_pass_count"] == 0
        assert report["successes"] == []
        assert report["repeated_public_task_level_transfer_demonstrated"] is False
        assert report["multi_identity_motif_recurrence_demonstrated"] is False
        assert report["outside_human_reproduction_completed"] is False
        assert report["world_level_breakthrough"] is False

    assert verdict_map(github) == verdict_map(local)
    github_hashes = {
        row["task_id"]: row["report_sha256"]
        for row in github["task_summaries"]
    }
    local_hashes = {
        row["task_id"]: row["report_sha256"]
        for row in local["task_summaries"]
    }
    differing = sorted(
        task_id for task_id in github_hashes
        if github_hashes[task_id] != local_hashes[task_id]
    )
    assert differing == ["0d87d2a6"]

    github_totals = github["totals"]
    local_totals = local["totals"]
    assert github_totals["accepted_examples"] == 384
    assert local_totals["accepted_examples"] == 384
    assert github_totals["candidate_evaluations"] == 640
    assert local_totals["candidate_evaluations"] == 640
    assert github_totals["exact_demonstration_candidates"] == 0
    assert local_totals["exact_demonstration_candidates"] == 0
    assert github_totals["generation_attempts"] == 387
    assert github_totals["generation_timeouts"] == 3
    assert local_totals["generation_attempts"] == 388
    assert local_totals["generation_timeouts"] == 4

    audit = {
        "schema": "lexigen-v31-local-reproduction-audit-v1",
        "authoritative_environment": "GitHub ubuntu-24.04 aggregate",
        "github_report_sha256": EXPECTED_GITHUB,
        "local_report_sha256": EXPECTED_LOCAL,
        "verdicts_identical": True,
        "byte_identical": False,
        "differing_task_ids": differing,
        "environment_divergence": {
            "task_id": "0d87d2a6",
            "cause": "one additional permitted Windows generator timeout shifted an accepted demonstration seed",
            "github_generation_attempts_total": 387,
            "github_generation_timeouts_total": 3,
            "local_generation_attempts_total": 388,
            "local_generation_timeouts_total": 4,
            "github_status": "no_program",
            "local_status": "no_program",
            "scientific_verdict_agrees": True,
        },
    }
    write(AUDIT, audit)
    evidence = {
        "schema": "lexigen-v31-validated-motif-recurrence-evidence-v1",
        "precommit_sha256": EXPECTED_PRECOMMIT,
        "authoritative_report_sha256": EXPECTED_GITHUB,
        "local_report_sha256": EXPECTED_LOCAL,
        "local_audit_sha256": sha256_file(AUDIT),
        "workflow": {
            "run_id": 30560489974,
            "conclusion": "success",
            "rift_run_id": 30560489793,
            "rift_conclusion": "success",
            "artifact_id": 8766679109,
            "artifact_digest": "sha256:475c7f4cc9556f8910f403ffeadc6c28d6d2203d684b88985c88f5093befcb92",
        },
        "denominator": {
            "task_count": 64,
            "accepted_examples": 384,
            "candidate_evaluations": 640,
            "exact_demonstration_candidate_count": 0,
            "fresh_pass_count": 0,
            "no_program_task_count": 64,
            "ambiguous_task_count": 0,
            "generator_invalid_task_count": 0,
        },
        "source": {
            "task_id": "9565186b",
            "fresh_cases_passed": 1000,
            "fresh_evidence_sha256": "f87d666d5a1dda22fa7ef179096c90eeed7670834673aecf67ad3007c71e3759",
        },
        "claims": {
            "validated_source_identity": True,
            "motif_recurrence_in_fresh_64_identity_sample": False,
            "repeated_public_task_level_transfer": False,
            "multi_identity_motif_recurrence": False,
            "outside_human_reproduction": False,
            "world_level_breakthrough": False,
        },
        "environment_note": {
            "github_report_authoritative": True,
            "local_verdict_reproduction": True,
            "one_environment_sensitive_report": True,
            "environment_sensitive_task_id": "0d87d2a6",
        },
    }
    write(EVIDENCE, evidence)

    markdown = f"""# v31 validated motif recurrence evidence

- Fresh identities completed: **64/64**
- Candidate evaluations: **640**
- Unique demonstration matches: **0**
- Fresh-pass identities: **0**
- No-program identities: **64**
- Authoritative report SHA-256: `{EXPECTED_GITHUB}`
- Workflow: `30560489974` (success)
- RIFT regression: `30560489793` (success)

The independent Windows run reached the same 64 `no_program` verdicts. Its aggregate bytes differ because task `0d87d2a6` had one additional permitted generator timeout, shifting one accepted demonstration seed. GitHub's aggregate is authoritative.

The v30 motif remains validated on source identity `9565186b`, but it did not recur in this preregistered 64-identity sample. Repeated task-level transfer, outside-human reproduction and a world-level breakthrough are not demonstrated.
"""
    MARKDOWN.write_text(markdown, encoding="utf-8")

    print(json.dumps({
        "authoritative_report_sha256": EXPECTED_GITHUB,
        "local_report_sha256": EXPECTED_LOCAL,
        "audit_sha256": sha256_file(AUDIT),
        "evidence_sha256": sha256_file(EVIDENCE),
        "fresh_pass_count": 0,
        "repeated_public_task_level_transfer": False,
        "world_level_breakthrough": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
