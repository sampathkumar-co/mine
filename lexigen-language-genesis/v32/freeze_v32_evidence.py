from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V32_PRECOMMIT.json"
GITHUB_REPORT = HERE / "V32_REPORT.json"
LOCAL_REPORT = HERE / "V32_LOCAL_REPORT.json"
SUCCESS_REPORT = HERE / "V32_SUCCESS_9caf5b84.json"
AUDIT = HERE / "V32_LOCAL_AUDIT.json"
EVIDENCE = HERE / "V32_EVIDENCE.json"
MARKDOWN = HERE / "EVIDENCE.md"

EXPECTED_PRECOMMIT = "555194c0d7d35caab361e81a02bd79002fdb3f1837e4180b644ac1f31ffdbe2e"
EXPECTED_GITHUB = "2d2c81fd4db7e811736ca02543195e75e952140058a627b0acb095358884a31c"
EXPECTED_LOCAL = "ffeedae33f076b181a1729b7444fc6fa2baabaff7374e091fe83c125925d0f2a"
EXPECTED_SUCCESS = "ab8c1411484ac4cb7516d3a4ec9aa808f5fe7d1f3b2029be923dc78d29ec6f06"


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
            row["exact_candidate_count"],
            json.dumps(row["selected_candidate"], sort_keys=True),
            row["fresh_passed"],
        )
        for row in report["task_summaries"]
    }


def main() -> None:
    assert sha256_file(PRECOMMIT) == EXPECTED_PRECOMMIT
    assert sha256_file(GITHUB_REPORT) == EXPECTED_GITHUB
    assert sha256_file(LOCAL_REPORT) == EXPECTED_LOCAL
    assert sha256_file(SUCCESS_REPORT) == EXPECTED_SUCCESS
    github = load(GITHUB_REPORT)
    local = load(LOCAL_REPORT)
    success_report = load(SUCCESS_REPORT)

    expected_status = {
        "fresh_fail": 0,
        "fresh_pass": 1,
        "generator_invalid": 0,
        "no_program": 63,
    }
    for report in (github, local):
        assert report["validation_task_count"] == 64
        assert report["completed_report_count"] == 64
        assert report["status_counts"] == expected_status
        assert report["fresh_pass_count"] == 1
        assert report["second_fresh_validated_identity_demonstrated"] is True
        assert report["outside_human_reproduction_completed"] is False
        assert report["world_level_breakthrough"] is False
        assert len(report["successes"]) == 1

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
    assert differing == ["e681b708"]
    assert github_hashes["9caf5b84"] == EXPECTED_SUCCESS
    assert local_hashes["9caf5b84"] == EXPECTED_SUCCESS

    selected = {
        "ast_sha256": "d630e72d5fdbff4bbd3e3e8295444ac7efc9646f2f63ac002190e3af01fe5dfc",
        "concrete_program_sha256": "8f2cb9f03e35dd8ad8cb39d32b26caed4e267d3b99a194fd15bd69338e40e701",
        "depth": 2,
        "nodes": 8,
        "parameters": {"c0": 7},
        "structural_index": 240,
    }
    assert success_report["task_id"] == "9caf5b84"
    assert success_report["status"] == "fresh_pass"
    assert success_report["accepted_examples"] == 6
    assert success_report["enumeration"]["exact_candidate_count"] == 1
    assert success_report["selected_candidate"] == selected
    assert success_report["fresh_gate"]["passed"] is True
    assert success_report["fresh_gate"]["totals"] == {
        "generated_cases": 100,
        "generation_errors": 0,
        "generation_timeouts": 0,
        "independent_runtime_errors": 0,
        "passed_cases": 100,
        "primary_runtime_errors": 0,
        "requested_cases": 100,
        "runtime_disagreements": 0,
        "target_mismatches": 0,
    }

    github_totals = github["totals"]
    local_totals = local["totals"]
    for totals in (github_totals, local_totals):
        assert totals["accepted_examples"] == 384
        assert totals["concrete_candidates_tested"] == 8166144
        assert totals["exact_demonstration_candidates"] == 1
        assert totals["fresh_requested_cases"] == 100
        assert totals["fresh_generated_cases"] == 100
        assert totals["fresh_passed_cases"] == 100
        assert totals["fresh_runtime_disagreements"] == 0
        assert totals["fresh_target_mismatches"] == 0
    assert github_totals["generation_attempts"] == 387
    assert github_totals["generation_timeouts"] == 3
    assert local_totals["generation_attempts"] == 390
    assert local_totals["generation_timeouts"] == 6

    audit = {
        "schema": "lexigen-v32-local-reproduction-audit-v1",
        "authoritative_environment": "GitHub ubuntu-24.04 aggregate",
        "github_report_sha256": EXPECTED_GITHUB,
        "local_report_sha256": EXPECTED_LOCAL,
        "verdicts_identical": True,
        "successful_task_report_byte_identical": True,
        "successful_task_report_sha256": EXPECTED_SUCCESS,
        "aggregate_byte_identical": False,
        "differing_task_ids": differing,
        "environment_divergence": {
            "task_id": "e681b708",
            "cause": "three additional permitted Windows generator timeouts shifted accepted demonstration seeds",
            "github_generation_attempts_total": 387,
            "github_generation_timeouts_total": 3,
            "local_generation_attempts_total": 390,
            "local_generation_timeouts_total": 6,
            "github_status": "no_program",
            "local_status": "no_program",
            "scientific_verdict_agrees": True,
        },
    }
    write(AUDIT, audit)
    evidence = {
        "schema": "lexigen-v32-full-grammar-transfer-evidence-v1",
        "precommit_sha256": EXPECTED_PRECOMMIT,
        "authoritative_report_sha256": EXPECTED_GITHUB,
        "local_report_sha256": EXPECTED_LOCAL,
        "successful_task_report_sha256": EXPECTED_SUCCESS,
        "local_audit_sha256": sha256_file(AUDIT),
        "workflow": {
            "run_id": 30563686569,
            "conclusion": "success",
            "rift_run_id": 30563686030,
            "rift_conclusion": "success",
            "artifact_id": 8768073198,
            "artifact_digest": "sha256:6e72073dc4918bbd17ee142efdbe247bb7ce982fc963be287794a8425d19a8ec",
        },
        "denominator": {
            "task_count": 64,
            "accepted_examples": 384,
            "concrete_candidates_tested": 8166144,
            "exact_demonstration_candidate_count": 1,
            "fresh_pass_count": 1,
            "no_program_task_count": 63,
            "fresh_fail_task_count": 0,
            "generator_invalid_task_count": 0,
        },
        "successful_transfer": {
            "task_id": "9caf5b84",
            "selected_candidate": selected,
            "fresh_cases_requested": 100,
            "fresh_cases_generated": 100,
            "fresh_cases_passed": 100,
            "runtime_disagreements": 0,
            "target_mismatches": 0,
            "primary_runtime_errors": 0,
            "independent_runtime_errors": 0,
        },
        "prior_fresh_validated_identity": {
            "task_id": "9565186b",
            "fresh_cases_passed": 1000,
            "fresh_evidence_sha256": "f87d666d5a1dda22fa7ef179096c90eeed7670834673aecf67ad3007c71e3759",
        },
        "claims": {
            "second_fresh_validated_public_identity": True,
            "repeated_public_task_level_transfer": True,
            "multi_identity_same_motif_recurrence": False,
            "outside_human_reproduction": False,
            "world_level_breakthrough": False,
        },
        "environment_note": {
            "github_report_authoritative": True,
            "local_verdict_reproduction": True,
            "successful_task_byte_identical": True,
            "one_environment_sensitive_no_program_report": True,
            "environment_sensitive_task_id": "e681b708",
        },
    }
    write(EVIDENCE, evidence)
    markdown = f"""# v32 full-grammar transfer evidence

- Fresh identities completed: **64/64**
- Concrete programs tested: **8,166,144**
- Exact-program identities: **1** (`9caf5b84`)
- No-program identities: **63**
- Fresh validation: **100/100** cases passed
- Authoritative report SHA-256: `{EXPECTED_GITHUB}`
- Successful task report SHA-256: `{EXPECTED_SUCCESS}`
- Workflow: `30563686569` (success)
- RIFT regression: `30563686030` (success)

The selected depth-2, eight-node program preserves the most frequent foreground colour and recolours every other foreground cell to `7`. The exact successful report is byte-identical on GitHub and Windows.

This is the second fresh-validated public identity solved by the source-induced grammar architecture, after `9565186b`. It demonstrates repeated public task-level transfer. It does not establish outside-human reproduction or a world-level breakthrough.

One no-program identity, `e681b708`, accepted different demonstration seeds because Windows observed three additional permitted generator timeouts. Both environments still found zero programs; GitHub's report is authoritative.
"""
    MARKDOWN.write_text(markdown, encoding="utf-8")

    print(json.dumps({
        "authoritative_report_sha256": EXPECTED_GITHUB,
        "successful_task_report_sha256": EXPECTED_SUCCESS,
        "audit_sha256": sha256_file(AUDIT),
        "evidence_sha256": sha256_file(EVIDENCE),
        "fresh_pass_count": 1,
        "repeated_public_task_level_transfer": True,
        "world_level_breakthrough": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
