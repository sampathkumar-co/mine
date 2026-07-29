from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "v14-dual-runtime-accepted-report.json"
FILES = (
    "scene_runtime_v14.py",
    "portable_scene_runtime_v14.py",
    "scene_synthesizer_v14.py",
    "test_scene_v14.py",
    "validate_v14_reproducible.py",
    "../external/arcgen_gate_v14.py",
    "v14-dual-runtime-accepted-report.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["families"] == 9
    assert report["total_accepted_cases"] == 90_000
    assert report["primary_failures"] == 0
    assert report["portable_failures"] == 0
    assert report["runtime_disagreements"] == 0

    evidence = {
        "schema": "lexigen-language-genesis-v14-evidence-v1",
        "version": "v14",
        "base_engine_commit": "18885982a880eb903f15752be04f7fd7ab2e8c2f",
        "v13_visible_evidence_commit": "13cd271a2d4813001563842a51a1e72dd100aa1a",
        "arcgen_commit": "a15cbdb44c776610aeeb9f487a06af875d3d0878",
        "development_gates": [1, 2, 3, 4, 5, 6, 8, 9, 12],
        "development_task_ids": [
            "dc433765", "49d1d64f", "c8f0f002", "9f236235", "ea9794b1",
            "c920a713", "eb5a1d5d", "da2b0fe3", "6e19193c",
        ],
        "families": report["families"],
        "accepted_cases_per_family": report["accepted_cases_per_family"],
        "total_accepted_cases_per_runtime": report["total_accepted_cases"],
        "total_cross_runtime_executions": 2 * report["total_accepted_cases"],
        "generator_attempts": report["total_generator_attempts"],
        "generator_rejections": report["total_generator_rejections"],
        "primary_failures": report["primary_failures"],
        "portable_failures": report["portable_failures"],
        "runtime_disagreements": report["runtime_disagreements"],
        "adversarial_tests_passed": 8,
        "combined_regression_tests_passed": 38,
        "composition_only_v13_scan": {
            "visible_packages": 39,
            "exact_compositions_found": 0,
            "max_depth": 2,
            "status": "preserved negative development observation",
        },
        "claim_boundary": {
            "world_level_breakthrough": False,
            "autonomous_language_genesis": False,
            "reason": (
                "The generic v14 scene operations were human-authored after visible failures. "
                "Sealed v14-only wins would be external capability evidence, not autonomous semantic invention."
            ),
        },
        "files": {
            relative: sha256((HERE / relative).resolve())
            for relative in FILES
        },
        "families_report": report["reports"],
    }
    evidence_path = HERE / "V14_EVIDENCE.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# v14 evidence",
        "",
        f"- Base v13 engine: `{evidence['base_engine_commit']}`",
        f"- Visible v13 evidence: `{evidence['v13_visible_evidence_commit']}`",
        f"- ARC-GEN: `{evidence['arcgen_commit']}`",
        f"- Families: **{evidence['families']}**",
        f"- Accepted cases per runtime: **{evidence['total_accepted_cases_per_runtime']:,}**",
        f"- Cross-runtime executions: **{evidence['total_cross_runtime_executions']:,}**",
        f"- Generator rejections recorded: **{evidence['generator_rejections']}**",
        f"- Primary failures: **{evidence['primary_failures']}**",
        f"- Portable failures: **{evidence['portable_failures']}**",
        f"- Runtime disagreements: **{evidence['runtime_disagreements']}**",
        f"- Adversarial tests: **{evidence['adversarial_tests_passed']} passed**",
        f"- Combined regressions: **{evidence['combined_regression_tests_passed']} passed**",
        "",
        "## Claim boundary",
        "",
        evidence["claim_boundary"]["reason"],
        "",
        f"Evidence JSON SHA-256: `{sha256(evidence_path)}`",
    ]
    (HERE / "EVIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_sha256": sha256(evidence_path)}, indent=2))


if __name__ == "__main__":
    main()
