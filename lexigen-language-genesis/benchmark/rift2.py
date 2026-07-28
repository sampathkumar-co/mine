from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from rift1 import State, World, build_cases

STOP_PREDICATES = ("stable", "repeat")
FINALIZERS = ("current", "trace_union", "canonical_pair")
MECHANISMS = ("closure", "trajectory_union", "two_cycle_canonical")


def canonical(state: State) -> tuple[str, ...]:
    return tuple(sorted(state))


def execute_trajectory_operator(
    instance: dict[str, Any],
    step: Callable[[State], State],
    seed: State,
    *,
    max_steps: int = 1_000,
) -> State:
    """Independent interpreter for the induced higher-order operator."""
    if instance.get("schema") != "lexigen-trajectory-operator-instance-v1":
        raise ValueError("unsupported operator instance")
    stop = instance.get("stop")
    finalize = instance.get("finalize")
    if stop not in STOP_PREDICATES or finalize not in FINALIZERS:
        raise ValueError("unsupported operator parameter")

    current = seed
    trace: list[State] = []
    seen: set[tuple[str, ...]] = set()

    for _ in range(max_steps):
        trace.append(current)
        seen.add(canonical(current))
        nxt = step(current)
        should_stop = (
            nxt == current if stop == "stable" else canonical(nxt) in seen
        )
        if should_stop:
            if finalize == "current":
                return current
            if finalize == "trace_union":
                return frozenset().union(*trace)
            return min((current, nxt), key=canonical)
        current = nxt
    raise RuntimeError("trajectory operator exhausted its step budget")


def candidate_instance(stop: str, finalize: str) -> dict[str, Any]:
    payload = {"stop": stop, "finalize": finalize}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "lexigen-trajectory-operator-instance-v1",
        "name": f"trajectory_instance_{hashlib.sha256(encoded).hexdigest()[:10]}",
        "stop": stop,
        "finalize": finalize,
    }


def fits(instance: dict[str, Any], cases: list[World]) -> bool:
    for case in cases:
        try:
            predicted = execute_trajectory_operator(instance, case.step, case.seed)
        except (RuntimeError, ValueError):
            return False
        if predicted != case.independently_verified_target():
            return False
    return True


def induce_instance(demonstrations: list[World]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [candidate_instance(stop, finalizer) for stop in STOP_PREDICATES for finalizer in FINALIZERS]
    candidates.sort(
        key=lambda candidate: hashlib.sha256(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
    )
    passing = [candidate for candidate in candidates if fits(candidate, demonstrations)]
    if not passing:
        raise RuntimeError("no trajectory-language instance fits demonstrations")
    # Frozen MDL rule: shortest canonical parameter encoding, then name.
    selected = min(
        passing,
        key=lambda candidate: (
            len(json.dumps({"stop": candidate["stop"], "finalize": candidate["finalize"]}, sort_keys=True)),
            candidate["name"],
        ),
    )
    return selected, passing


def induce_operator_family(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Anti-unify instances into one persistent executable operator schema."""
    fields = ("stop", "finalize")
    varying = [field for field in fields if len({instance[field] for instance in instances}) > 1]
    fixed = {
        field: instances[0][field]
        for field in fields
        if field not in varying
    }
    schema_body = {
        "loop": [
            "append current to trace",
            "record current in seen set",
            "compute next = transition(current)",
            "evaluate parameterized stop predicate",
            "apply parameterized finalizer or continue with current = next",
        ],
        "varying_parameters": varying,
        "fixed_parameters": fixed,
    }
    encoded = json.dumps(schema_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "lexigen-induced-operator-family-v1",
        "name": f"trajectory_fold_{hashlib.sha256(encoded).hexdigest()[:12]}",
        "operator_schema": schema_body,
        "instances": instances,
        "provenance": {
            "method": "semantic anti-unification of independently induced executable instances",
            "family_sha256": hashlib.sha256(encoded).hexdigest(),
            "human_supplied_substrate": [
                "finite trace storage",
                "state equality",
                "state-set membership",
                "set union",
                "canonical ordering",
                "black-box transition application",
            ],
        },
    }


def run(output_dir: Path) -> dict[str, Any]:
    induced: dict[str, dict[str, Any]] = {}
    episode_reports: dict[str, Any] = {}

    for mechanism in MECHANISMS:
        demonstrations = build_cases(mechanism, range(4, 7), replicas=2)
        transfer = build_cases(mechanism, range(7, 14), replicas=2)
        instance, passing = induce_instance(demonstrations)
        if not fits(instance, transfer):
            raise AssertionError(f"induced operator instance failed transfer: {mechanism}")
        induced[mechanism] = instance
        episode_reports[mechanism] = {
            "selected_instance": instance,
            "passing_instance_names": [candidate["name"] for candidate in passing],
            "transfer_case_count": len(transfer),
            "transfer_accuracy": 1.0,
        }

    family = induce_operator_family(list(induced.values()))

    separate_bytes = sum(
        len(json.dumps(instance, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for instance in induced.values()
    )
    family_bytes = len(
        json.dumps(family["operator_schema"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ) + sum(
        len(json.dumps({"stop": i["stop"], "finalize": i["finalize"]}, sort_keys=True).encode("utf-8"))
        for i in induced.values()
    )

    report = {
        "benchmark": "RIFT-2",
        "status": "induced higher-order executable language; L4 candidate, not a world breakthrough claim",
        "family": family,
        "episodes": episode_reports,
        "description_length": {
            "separate_instance_bytes": separate_bytes,
            "family_plus_parameters_bytes": family_bytes,
            "compression_ratio": separate_bytes / family_bytes,
        },
        "limitation": (
            "The higher-order operator is induced and executable, but its substrate operations remain human supplied "
            "and comparable library-learning/anti-unification ideas exist in prior research."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    family_path = output_dir / "rift2-induced-operator-family.json"
    report_path = output_dir / "rift2-report.json"
    family_path.write_text(json.dumps(family, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "family": family["name"],
        "instances": {mechanism: instance["name"] for mechanism, instance in induced.items()},
        "all_transfer_exact": all(value["transfer_accuracy"] == 1.0 for value in episode_reports.values()),
        "compression_ratio": report["description_length"]["compression_ratio"],
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift2"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
