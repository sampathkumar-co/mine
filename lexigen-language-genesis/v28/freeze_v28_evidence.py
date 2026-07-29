from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load(name: str) -> Any:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    precommit = load("V28_PRECOMMIT.json")
    report = load("V28_DISCOVERY_REPORT.json")
    library = load("V28_LIBRARY.json")
    first = load("V28_FIRST_COMPARISON_REPORT.json")
    summary = load("V28_FIRST_COMPARISON_SUMMARY.json")

    precommit_sha = sha256_file(HERE / "V28_PRECOMMIT.json")
    library_sha = sha256_file(HERE / "V28_LIBRARY.json")
    require(report["schema"] == "lexigen-v28-discovery-report-v1", "report schema")
    require(library["schema"] == "lexigen-v28-factorized-library-v1", "library schema")
    require(report["precommit_sha256"] == precommit_sha, "report precommit binding")
    require(library["precommit_sha256"] == precommit_sha, "library precommit binding")
    require(report["library_sha256"] == library_sha, "library file binding")
    require(report["discovery_task_count"] == precommit["discovery_task_count"] == 40, "denominator")
    require(
        report["completed_task_count"] + report["generator_invalid_task_count"] == 40,
        "task accounting",
    )
    require(report["validation_generators_imported"] == 0, "validation imported")
    require(not report["validation_outputs_opened"], "validation outputs opened")
    require(library["validation_generators_imported"] == 0, "library validation imported")
    require(not library["validation_outputs_opened"], "library validation opened")
    require(not report["heldout_transfer_demonstrated"], "premature transfer claim")
    require(not report["world_level_breakthrough"], "premature breakthrough claim")
    require(summary["comparison"]["precommitted_gate_passed"], "first efficiency gate")

    expected_ids = list(precommit["discovery_task_ids"])
    summaries = report["task_summaries"]
    require([item["task_id"] for item in summaries] == expected_ids, "task order or identity changed")
    first_summary = next(item for item in summaries if item["task_id"] == "c074846d")
    require(
        first_summary["report_sha256"] == sha256_file(HERE / "V28_FIRST_COMPARISON_REPORT.json"),
        "first report did not reproduce byte-for-byte",
    )
    require(first["demonstration_sha256"] == summary.get("demonstration_sha256", first["demonstration_sha256"]), "demonstration binding")

    files = {
        name: sha256_file(HERE / name)
        for name in (
            "V28_PRECOMMIT.json",
            "V28_FIRST_COMPARISON_REPORT.json",
            "V28_FIRST_COMPARISON_SUMMARY.json",
            "V28_DISCOVERY_REPORT.json",
            "V28_LIBRARY.json",
            "enumerator_v28.py",
            "scan_one_v28.py",
            "aggregate_discovery_v28.py",
            "test_v28.py",
        )
    }
    evidence = {
        "schema": "lexigen-v28-evidence-v1",
        "version": 28,
        "files": files,
        "discovery_task_count": report["discovery_task_count"],
        "completed_task_count": report["completed_task_count"],
        "generator_invalid_task_count": report["generator_invalid_task_count"],
        "totals": report["totals"],
        "distinct_exact_structure_count": report["distinct_exact_structure_count"],
        "qualifying_structure_count": report["qualifying_structure_count"],
        "first_comparison_gate_passed": True,
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "heldout_transfer_demonstrated": False,
        "world_level_breakthrough": False,
        "claim_boundary": precommit["claim_boundary"],
    }
    write(HERE / "V28_EVIDENCE.json", evidence)

    verdict = (
        "The discovery library is nonempty and may proceed to the frozen held-out gate."
        if report["qualifying_structure_count"]
        else "The discovery library is empty; held-out validation remains unopened."
    )
    markdown = f"""# Lexigen v28 Evidence

- Discovery identities: {report['discovery_task_count']}
- Completed identities: {report['completed_task_count']}
- Generator-invalid identities: {report['generator_invalid_task_count']}
- Raw candidate evaluations: {report['totals']['raw_candidate_evaluations']}
- Retained semantic expressions: {report['totals']['total_retained_expressions']}
- Exact concrete programs: {report['totals']['exact_concrete_programs']}
- Distinct exact structures: {report['distinct_exact_structure_count']}
- Structures qualifying on at least two identities: {report['qualifying_structure_count']}
- Validation generators imported: 0
- Validation outputs opened: false
- World-level breakthrough: false

{verdict}

v28 is an architecture-efficiency result within the frozen typed ARC-GEN language. It is not sealed external evidence, unrestricted semantic invention, or a world-level breakthrough.
"""
    (HERE / "EVIDENCE.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({
        "evidence_sha256": sha256_file(HERE / "V28_EVIDENCE.json"),
        "qualifying_structure_count": report["qualifying_structure_count"],
        "validation_outputs_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
