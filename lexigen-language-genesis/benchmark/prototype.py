from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from artifact_runtime import execute_artifact
from rift0 import World, bounded_unroll_3, build_cases, evaluate


def trace_case(case: World, max_rounds: int = 64) -> list[frozenset[str]]:
    states = [case.seed]
    for _ in range(max_rounds):
        updated = case.step(states[-1])
        states.append(updated)
        if updated == states[-2]:
            return states
    raise RuntimeError(f"diagnostic case did not stabilize: {case.name}")


def diagnose_representation_failure(cases: list[World]) -> dict[str, Any]:
    failed = []
    traces = []
    for case in cases:
        target = case.independently_verified_target()
        baseline = bounded_unroll_3(case.step, case.seed)
        trace = trace_case(case)
        traces.append((case, target, trace))
        if baseline != target:
            failed.append(case.name)

    monotone = all(
        before.issubset(after)
        for _, _, trace in traces
        for before, after in zip(trace, trace[1:])
    )
    target_is_first_stable_state = all(
        trace[-1] == trace[-2] == target for _, target, trace in traces
    )
    required_rounds = sorted({len(trace) - 2 for _, _, trace in traces})
    variable_depth = len(required_rounds) > 1
    exceeds_starting_language = any(rounds > 3 for rounds in required_rounds)

    return {
        "failed_cases": failed,
        "monotone": monotone,
        "target_is_first_stable_state": target_is_first_stable_state,
        "required_rounds": required_rounds,
        "variable_depth": variable_depth,
        "exceeds_starting_language": exceeds_starting_language,
        "representation_failure_supported": bool(
            failed
            and monotone
            and target_is_first_stable_state
            and variable_depth
            and exceeds_starting_language
        ),
    }


def invent_artifact(diagnosis: dict[str, Any]) -> dict[str, Any]:
    if not diagnosis.get("representation_failure_supported"):
        raise RuntimeError("evidence does not justify a language extension")

    evidence = json.dumps(diagnosis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    suffix = hashlib.sha256(evidence).hexdigest()[:10]
    return {
        "schema": "lexigen-language-artifact-v1",
        "name": f"stabilize_{suffix}",
        "signature": "((State -> State), State) -> State",
        "activation_rule": {
            "requires": [
                "observed transition is monotone on diagnostic traces",
                "correct targets equal the first stable states",
                "required iteration depth varies and exceeds frozen unroll depth",
            ]
        },
        "termination_contract": "finite state space and monotone non-decreasing transition",
        "program": [
            {"op": "APPLY_STEP"},
            {"op": "RETURN_IF_STABLE"},
            {"op": "ASSERT_MONOTONE"},
            {"op": "ADVANCE"},
            {"op": "JUMP", "target": 0},
        ],
        "provenance": {
            "method": "counterexample-guided control-schema induction prototype",
            "diagnosis_sha256": hashlib.sha256(evidence).hexdigest(),
            "warning": (
                "This v0 prototype selects a human-authored bytecode schema after evidence checks. "
                "It validates the benchmark plumbing but is not autonomous semantic invention."
            ),
        },
    }


def run(output_dir: Path) -> dict[str, Any]:
    calibration_cases = build_cases(range(1, 4), replicas=2)
    diagnostic_cases = build_cases(range(4, 7), replicas=2)
    transfer_cases = build_cases(range(7, 13), replicas=3)

    diagnosis = diagnose_representation_failure(diagnostic_cases)
    artifact = invent_artifact(diagnosis)

    def artifact_solver(step, seed):
        return execute_artifact(artifact, step, seed)

    report: dict[str, Any] = {
        "experiment": "RIFT-0 counterexample-guided prototype",
        "status": "research scaffold; no novelty or breakthrough claim",
        "calibration_bounded": evaluate(bounded_unroll_3, calibration_cases),
        "diagnostic_bounded": evaluate(bounded_unroll_3, diagnostic_cases),
        "transfer_bounded": evaluate(bounded_unroll_3, transfer_cases),
        "diagnosis": diagnosis,
        "artifact_transfer": evaluate(artifact_solver, transfer_cases),
        "artifact_name": artifact["name"],
    }

    if report["calibration_bounded"]["accuracy"] < 0.95:
        raise AssertionError("starting language must solve calibration cases")
    if report["diagnostic_bounded"]["accuracy"] >= 0.80:
        raise AssertionError("diagnostic cases must expose the representation limit")
    if report["artifact_transfer"]["accuracy"] != 1.0:
        raise AssertionError("emitted artifact must transfer exactly in the synthetic scaffold")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "invented-language-artifact.json"
    report_path = output_dir / "prototype-report.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    summary = {
        "artifact": artifact["name"],
        "bounded_transfer_accuracy": report["transfer_bounded"]["accuracy"],
        "artifact_transfer_accuracy": report["artifact_transfer"]["accuracy"],
        "report_sha256": digest,
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
