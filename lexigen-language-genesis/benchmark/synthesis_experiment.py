from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from artifact_runtime import execute_artifact
from prototype import diagnose_representation_failure
from rift0 import bounded_unroll_3, build_cases, evaluate
from synthesizer import synthesize


def run(output_dir: Path) -> dict[str, Any]:
    diagnostic_cases = build_cases(range(4, 7), replicas=2)
    transfer_cases = build_cases(range(7, 13), replicas=3)
    diagnosis = diagnose_representation_failure(diagnostic_cases)
    if not diagnosis["representation_failure_supported"]:
        raise AssertionError("diagnosis did not justify synthesis")

    result = synthesize(diagnostic_cases, diagnosis, max_length=6, max_instructions=256)
    artifact = result.artifact

    def synthesized_solver(step, seed):
        return execute_artifact(artifact, step, seed, max_instructions=512)

    report: dict[str, Any] = {
        "experiment": "RIFT-0 CEGIS bytecode composition",
        "status": "fixed-meta-language synthesis; no L4/L5 or breakthrough claim",
        "programs_tested": result.programs_tested,
        "counterexample_rounds": result.counterexample_rounds,
        "active_case_names": list(result.active_case_names),
        "artifact": artifact,
        "bounded_transfer": evaluate(bounded_unroll_3, transfer_cases),
        "synthesized_transfer": evaluate(synthesized_solver, transfer_cases),
    }

    if report["bounded_transfer"]["accuracy"] >= 0.80:
        raise AssertionError("fixed starting language did not expose a gap")
    if report["synthesized_transfer"]["accuracy"] != 1.0:
        raise AssertionError("synthesized artifact did not transfer exactly")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "synthesized-language-artifact.json"
    report_path = output_dir / "synthesis-report.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "artifact_name": artifact["name"],
        "program": artifact["program"],
        "programs_tested": result.programs_tested,
        "counterexample_rounds": result.counterexample_rounds,
        "transfer_accuracy": report["synthesized_transfer"]["accuracy"],
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift0"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
