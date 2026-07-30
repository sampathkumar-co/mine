from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V30_956_FRESH_PRECOMMIT.json"
REPORT = HERE / "V30_956_FRESH_REPORT.json"
EVIDENCE = HERE / "V30_956_FRESH_EVIDENCE.json"
MARKDOWN = HERE / "EVIDENCE.md"

EXPECTED_PRECOMMIT = "e154ee74a55f7105a962f2ff9f6abf307c3c1ba62024b781d8e40552eb7f241e"
EXPECTED_REPORT = "71bb437f15039c9a4319e4bd81fe87b2ca076c71e7339d430c41a2c5732d6651"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def main() -> None:
    precommit = load(PRECOMMIT)
    report = load(REPORT)
    assert sha256_file(PRECOMMIT) == EXPECTED_PRECOMMIT
    assert sha256_file(REPORT) == EXPECTED_REPORT
    assert report["precommit_sha256"] == EXPECTED_PRECOMMIT
    assert report["source_task_id"] == "9565186b"
    assert report["case_count"] == 1000
    assert report["fresh_gate_passed"] is True
    assert report["fresh_case_validation_demonstrated"] is True
    assert report["repeated_task_level_transfer_demonstrated"] is False
    assert report["world_level_breakthrough"] is False
    totals = report["totals"]
    assert totals == {
        "requested_cases": 1000,
        "generated_cases": 1000,
        "generation_timeouts": 0,
        "generation_errors": 0,
        "primary_runtime_errors": 0,
        "independent_runtime_errors": 0,
        "runtime_disagreements": 0,
        "target_mismatches": 0,
        "verifier_rejections": 0,
        "passed_cases": 1000,
    }

    records = report["case_records"]
    assert len(records) == 1000
    assert [item["case_index"] for item in records] == list(range(1000))
    assert len({item["seed"] for item in records}) == 1000
    assert all(item["passed"] is True for item in records)
    assert all(item["runtime_agrees"] is True for item in records)
    assert all(item["target_matches"] is True for item in records)
    assert all(item["verifier_accepts"] is True for item in records)
    assert report["selected_candidate"] == precommit["selected_candidate"]

    evidence = {
        "schema": "lexigen-v30-9565186b-fresh-evidence-v1",
        "task_id": "9565186b",
        "precommit_sha256": EXPECTED_PRECOMMIT,
        "report_sha256": EXPECTED_REPORT,
        "case_record_sequence_sha256": sha256_json(records),
        "selected_candidate": report["selected_candidate"],
        "totals": totals,
        "fresh_gate_passed": True,
        "local_runtime_report_sha256": EXPECTED_REPORT,
        "github_reproduction": {
            "workflow_run_id": 30554399664,
            "artifact_id": 8764161444,
            "artifact_digest": "sha256:dc7c62c136d39a98e1bd613fca514edb369b697a440d88147a23162a226f9506",
            "report_sha256": EXPECTED_REPORT,
            "rift_run_id": 30554399915,
            "rift_conclusion": "success"
        },
        "claims": {
            "public_heldout_synthesis_event": True,
            "fresh_case_validation_on_one_identity": True,
            "repeated_task_level_transfer": False,
            "outside_human_reproduction": False,
            "world_level_breakthrough": False
        }
    }
    write(EVIDENCE, evidence)
    markdown = f"""# v30 task 9565186b fresh evidence

- Frozen candidate: `paint(input_grid, non_background_points(input_grid), 5)`
- Fixed fresh cases: **1000**
- Passed: **1000**
- Generation failures/timeouts: **0**
- Runtime disagreements: **0**
- Target mismatches: **0**
- Verifier rejections: **0**
- Report SHA-256: `{EXPECTED_REPORT}`
- Independent GitHub reproduction: run `30554399664`

This demonstrates fresh-case validation on one previously unused public identity. It does not demonstrate repeated task-level transfer, outside-human reproduction, or a world-level breakthrough.
"""
    MARKDOWN.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "evidence_sha256": sha256_file(EVIDENCE),
        "report_sha256": EXPECTED_REPORT,
        "fresh_gate_passed": True,
        "world_level_breakthrough": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
