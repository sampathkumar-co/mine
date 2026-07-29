from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V29_PRECOMMIT.json"
LIBRARY = HERE / "V29_TEMPLATE_LIBRARY.json"
LOCAL_REPORT = HERE / "V29_LOCAL_VALIDATION_REPORT.json"
REPORT = HERE / "V29_VALIDATION_REPORT.json"
EVIDENCE = HERE / "V29_EVIDENCE.json"
MARKDOWN = HERE / "EVIDENCE.md"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    precommit = load(PRECOMMIT)
    library = load(LIBRARY)
    local_report = load(LOCAL_REPORT)

    assert local_report["schema"] == "lexigen-v29-validation-report-v1"
    assert local_report["validation_task_count"] == 20
    assert local_report["totals"]["candidate_instantiations_tested"] == 3200
    assert local_report["status_counts"] == {
        "ambiguous": 0,
        "generator_invalid": 0,
        "no_program": 20,
        "unique_exact": 0,
    }
    assert local_report["unique_success_count"] == 0
    assert local_report["heldout_template_match_demonstrated"] is False
    assert local_report["world_level_breakthrough"] is False
    assert library["validation_outputs_opened"] is False
    assert len(precommit["validation_task_ids"]) == 20

    REPORT.write_bytes(LOCAL_REPORT.read_bytes())
    report_sha = sha256_file(REPORT)
    assert report_sha == "bf740077661d9d3c596041cbbbd8dc0df772ffdacd732f25791a7c915faaee29"

    evidence = {
        "schema": "lexigen-v29-frozen-evidence-v1",
        "precommit_sha256": sha256_file(PRECOMMIT),
        "template_library_sha256": sha256_file(LIBRARY),
        "validation_report_sha256": report_sha,
        "github_workflow_run_id": 30474816826,
        "github_artifact_id": 8733180910,
        "github_artifact_digest": "sha256:576fd41ca4f08f52c709d2386e536b592ddabd610d2750e093d5292bcf5aff61",
        "local_and_github_report_byte_identical": True,
        "validation_task_count": 20,
        "candidate_instantiations_tested": 3200,
        "unique_success_count": 0,
        "heldout_template_match_demonstrated": False,
        "fresh_validation_completed": True,
        "verifier_cosynthesis_completed": False,
        "world_level_breakthrough": False,
        "claim_boundary": (
            "Public ARC-GEN held-out negative-transfer evidence for one frozen "
            "operator-hole template; not sealed external evidence or unrestricted invention."
        ),
    }
    write_json(EVIDENCE, evidence)

    lines = [
        "# Lexigen v29 Evidence",
        "",
        "- Twenty precommitted held-out identities completed.",
        "- Exactly 3,200 frozen template instantiations were tested.",
        "- No unique or ambiguous exact instantiation appeared.",
        "- The local and GitHub aggregate reports are byte-identical.",
        f"- Validation report SHA-256: `{report_sha}`.",
        "- Held-out transfer was not demonstrated.",
        "- Verifier co-synthesis was not opened because no candidate transferred.",
        "- This is not a world-level breakthrough.",
        "",
    ]
    MARKDOWN.write_bytes(("\n".join(lines)).encode("utf-8"))

    print(json.dumps({
        "evidence_sha256": sha256_file(EVIDENCE),
        "validation_report_sha256": report_sha,
        "unique_success_count": 0,
        "heldout_template_match_demonstrated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
