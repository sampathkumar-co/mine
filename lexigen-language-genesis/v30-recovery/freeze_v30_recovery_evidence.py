from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V30_RECOVERY_PRECOMMIT.json"
EQUIVALENCE = HERE / "V30_RECOVERY_EQUIVALENCE.json"
AGGREGATE = HERE / "V30_RECOVERED_VALIDATION_REPORT.json"
AUDIT = HERE / "V30_LOCAL_RECOVERY_AUDIT.json"
EVIDENCE = HERE / "V30_RECOVERY_EVIDENCE.json"
MARKDOWN = HERE / "EVIDENCE.md"

EXPECTED_PRECOMMIT = "308921e41dcd473259ffa027d308bce62ee11b1f95730b2c8f9c99f05587428b"
EXPECTED_EQUIVALENCE = "41cbd42cc26682bb8487d26daecd7b33417e80d2276774fde84106c3b1f4d5bb"
EXPECTED_AGGREGATE = "37954acab4f93b031665517629ac91499dc4891324c818a020e8be344d17874c"

RECOVERY_IDS = (
    "1e81d6f9", "639f5a19", "3ee1011a", "e4075551", "6ca952ad",
    "d94c3b52", "6d0160f0", "05f2a901", "e39e9282",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    assert sha256_file(PRECOMMIT) == EXPECTED_PRECOMMIT
    assert sha256_file(EQUIVALENCE) == EXPECTED_EQUIVALENCE
    assert sha256_file(AGGREGATE) == EXPECTED_AGGREGATE
    precommit = load(PRECOMMIT)
    equivalence = load(EQUIVALENCE)
    aggregate = load(AGGREGATE)
    assert equivalence["byte_identical"] is True
    assert equivalence["recovery_authorized"] is True
    assert aggregate["completed_report_count"] == 20
    assert aggregate["validation_task_count"] == 20
    assert aggregate["status_counts"] == {
        "exact_program_found": 1,
        "generator_invalid": 0,
        "no_program": 19,
    }
    assert aggregate["heldout_synthesis_demonstrated"] is True
    assert aggregate["heldout_synthesis_event_count"] == 1
    assert aggregate["repeated_heldout_transfer_demonstrated"] is False
    assert aggregate["world_level_breakthrough"] is False

    totals = aggregate["totals"]
    assert totals == {
        "accepted_examples": 120,
        "concrete_candidates_tested": 2551920,
        "exact_candidates": 136,
        "generation_attempts": 120,
        "generation_failures": 0,
        "generation_timeouts": 0,
        "identity_candidates_rejected": 70324,
        "runtime_invalid_candidates": 1423517,
    }
    assert aggregate["successes"] == [{
        "exact_candidate_count": 136,
        "selected_candidate": {
            "ast_sha256": "165ddda73ec0467d33928df08b3a115dfd9c93f5f259cb90b13094c04bd7870a",
            "concrete_program_sha256": "5bc7ccf73fa03eee67d7b5b894397ced698720af28d145bca38fe284f7186cb8",
            "depth": 1,
            "nodes": 4,
            "parameters": {"c0": 5},
            "structural_index": 4,
        },
        "task_id": "9565186b",
    }]
    expected_reports = {
        item["task_id"]: item["report_sha256"]
        for item in aggregate["task_summaries"]
    }

    local_rows: list[dict[str, Any]] = []
    for task_id in RECOVERY_IDS:
        path = HERE / f"recovered-{task_id}.json"
        report = load(path)
        actual_hash = sha256_file(path)
        row = {
            "task_id": task_id,
            "local_report_sha256": actual_hash,
            "github_aggregate_report_sha256": expected_reports[task_id],
            "byte_identical_to_github": actual_hash == expected_reports[task_id],
            "status": report["status"],
            "accepted_examples": report["accepted_examples"],
            "exact_candidate_count": report["enumeration"]["exact_candidate_count"],
            "generation_attempts": report["generation"]["attempts"],
            "generation_timeouts": report["generation"]["timeouts"],
            "demonstration_sha256": report["demonstration_sha256"],
        }
        assert row["status"] == "completed"
        assert row["accepted_examples"] == 6
        assert row["exact_candidate_count"] == 0
        local_rows.append(row)

    matching = [row["task_id"] for row in local_rows if row["byte_identical_to_github"]]
    differing = [row["task_id"] for row in local_rows if not row["byte_identical_to_github"]]
    assert matching == [
        "1e81d6f9", "639f5a19", "3ee1011a", "e4075551",
        "6ca952ad", "d94c3b52", "6d0160f0", "05f2a901",
    ]
    assert differing == ["e39e9282"]

    local_e39 = next(row for row in local_rows if row["task_id"] == "e39e9282")
    assert local_e39["generation_attempts"] == 8
    assert local_e39["generation_timeouts"] == 2
    github_e39 = next(
        item for item in aggregate["task_summaries"]
        if item["task_id"] == "e39e9282"
    )
    assert github_e39["exact_candidate_count"] == 0

    audit = {
        "schema": "lexigen-v30-local-recovery-audit-v1",
        "authoritative_environment": "GitHub ubuntu-24.04 zero-timeout aggregate",
        "authoritative_aggregate_sha256": EXPECTED_AGGREGATE,
        "local_report_count": len(local_rows),
        "byte_identical_report_count": len(matching),
        "byte_identical_task_ids": matching,
        "environment_divergent_task_ids": differing,
        "environment_divergence": {
            "task_id": "e39e9282",
            "cause": "two permitted Windows generator-call timeouts shifted the six accepted demonstration seeds",
            "local_generation_attempts": 8,
            "local_generation_timeouts": 2,
            "github_generation_attempts": 6,
            "github_generation_timeouts": 0,
            "local_exact_candidate_count": 0,
            "github_exact_candidate_count": 0,
            "scientific_verdict_agrees": True,
            "byte_identical": False,
        },
        "reports": local_rows,
    }
    write(AUDIT, audit)

    evidence = {
        "schema": "lexigen-v30-recovered-full-denominator-evidence-v1",
        "recovery_precommit_sha256": EXPECTED_PRECOMMIT,
        "recovery_equivalence_sha256": EXPECTED_EQUIVALENCE,
        "aggregate_report_sha256": EXPECTED_AGGREGATE,
        "local_recovery_audit_sha256": sha256_file(AUDIT),
        "original_workflow_run_id": 30479425040,
        "recovery_workflow": {
            "run_id": 30556083038,
            "conclusion": "success",
            "rift_run_id": 30556083783,
            "rift_conclusion": "success",
            "artifact_id": 8764885792,
            "artifact_digest": "sha256:8f5c0862ba56a2fe2cdf630f473b09c4a9e81fd6260aeafb5df2339d79e33f7b",
        },
        "denominator": {
            "task_count": 20,
            "accepted_examples": 120,
            "concrete_candidates_tested": 2551920,
            "exact_candidate_count": 136,
            "exact_task_count": 1,
            "no_program_task_count": 19,
            "generator_invalid_task_count": 0,
        },
        "validated_success": {
            "task_id": "9565186b",
            "selected_concrete_program_sha256": "5bc7ccf73fa03eee67d7b5b894397ced698720af28d145bca38fe284f7186cb8",
            "fresh_report_sha256": "71bb437f15039c9a4319e4bd81fe87b2ca076c71e7339d430c41a2c5732d6651",
            "fresh_evidence_sha256": "f87d666d5a1dda22fa7ef179096c90eeed7670834673aecf67ad3007c71e3759",
            "fresh_workflow_run_id": 30554399664,
            "fresh_cases_passed": 1000,
        },
        "claims": {
            "public_heldout_synthesis_event": True,
            "fresh_case_validation_on_one_identity": True,
            "repeated_task_level_transfer": False,
            "outside_human_reproduction": False,
            "world_level_breakthrough": False,
        },
        "environment_note": {
            "github_aggregate_authoritative": True,
            "eight_local_reports_byte_identical": True,
            "e39e9282_environment_sensitive_generation": True,
            "e39e9282_scientific_verdict_agrees": True,
        },
    }
    write(EVIDENCE, evidence)
    markdown = f"""# v30 recovered full-denominator evidence

- Frozen identities completed: **20/20**
- Concrete candidates tested: **2,551,920**
- Exact-program identities: **1** (`9565186b`)
- No-program identities: **19**
- Generator-invalid identities: **0**
- Aggregate SHA-256: `{EXPECTED_AGGREGATE}`
- Recovery workflow: `30556083038` (success)
- Fresh validation: **1000/1000** cases passed for `9565186b`

Eight Windows recovery reports are byte-identical to GitHub. `e39e9282` had two permitted Windows generator timeouts, so later demonstration seeds were accepted; both environments still found zero programs. GitHub's zero-timeout report is authoritative.

This demonstrates one public held-out synthesis event with fresh-case validation on that identity. It does not demonstrate repeated task-level transfer, outside-human reproduction, or a world-level breakthrough.
"""
    MARKDOWN.write_text(markdown, encoding="utf-8")

    print(json.dumps({
        "aggregate_report_sha256": EXPECTED_AGGREGATE,
        "audit_sha256": sha256_file(AUDIT),
        "evidence_sha256": sha256_file(EVIDENCE),
        "exact_task_count": 1,
        "fresh_cases_passed": 1000,
        "repeated_task_level_transfer": False,
        "world_level_breakthrough": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
