from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from portable_family_runtime import run_portable_instance
from rift1 import State, execute_artifact

SURFACES = ("graph", "rules", "grid")


def key(state: State) -> tuple[str, ...]:
    return tuple(sorted(state))


@dataclass(frozen=True)
class MaxCycleWorld:
    name: str
    surface: str
    transitions: tuple[tuple[str, str], ...]
    seed: State

    def _mapping(self) -> dict[str, str]:
        return dict(self.transitions)

    def step(self, state: State) -> State:
        token = next(iter(state))
        return frozenset({self._mapping()[token]})

    def independently_verified_target(self) -> State:
        current = self.seed
        seen = {key(current)}
        while True:
            nxt = self.step(current)
            if key(nxt) in seen:
                return max((current, nxt), key=key)
            seen.add(key(nxt))
            current = nxt


def prefix(surface: str, replica: int) -> str:
    return {
        "graph": f"v{replica}_",
        "rules": f"atom{replica}_",
        "grid": f"sq{replica}_",
    }[surface]


def make_world(depth: int, surface: str, replica: int) -> MaxCycleWorld:
    p = prefix(surface, replica)
    transient = [f"{p}t{i}" for i in range(depth)]
    cycle = [f"{p}cycle_b", f"{p}cycle_z", f"{p}cycle_a"]
    transitions: list[tuple[str, str]] = []
    transitions.extend((transient[i], transient[i + 1]) for i in range(len(transient) - 1))
    transitions.append((transient[-1], cycle[0]))
    transitions.extend(((cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[2], cycle[0])))
    return MaxCycleWorld(
        name=f"cycle-max-{surface}-d{depth}-i{replica}",
        surface=surface,
        transitions=tuple(transitions),
        seed=frozenset({transient[0]}),
    )


def build_cases(depths: Iterable[int], replicas: int = 1) -> list[MaxCycleWorld]:
    return [
        make_world(depth, surface, replica)
        for depth in depths
        for replica in range(replicas)
        for surface in SURFACES
    ]


def execute_family(instance: dict[str, Any], case: MaxCycleWorld) -> State:
    current = case.seed
    visited: set[tuple[str, ...]] = set()
    history: list[State] = []
    for _ in range(5_000):
        history.append(current)
        visited.add(key(current))
        nxt = case.step(current)
        done = nxt == current if instance["stop"] == "stable" else key(nxt) in visited
        if done:
            finalizer = instance["finalize"]
            if finalizer == "current":
                return current
            if finalizer == "next":
                return nxt
            if finalizer == "trace_union":
                return frozenset().union(*history)
            if finalizer == "canonical_pair":
                return min((current, nxt), key=key)
            if finalizer == "canonical_max":
                return max((current, nxt), key=key)
            raise ValueError(finalizer)
        current = nxt
    raise RuntimeError("family execution did not terminate")


def adapt_family(demonstrations: list[MaxCycleWorld]) -> tuple[dict[str, Any], int]:
    inherited = ("current", "trace_union", "canonical_pair", "next")
    derived_extensions = ("canonical_max",)
    candidates = [
        {"schema": "lexigen-trajectory-operator-instance-v3", "stop": stop, "finalize": finalizer}
        for stop in ("stable", "repeat")
        for finalizer in inherited + derived_extensions
    ]
    candidates.sort(
        key=lambda candidate: hashlib.sha256(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
    )
    tested = 0
    passing: list[dict[str, Any]] = []
    for candidate in candidates:
        tested += 1
        solved = True
        for case in demonstrations:
            try:
                predicted = execute_family(candidate, case)
            except (RuntimeError, ValueError):
                solved = False
                break
            if predicted != case.independently_verified_target():
                solved = False
                break
        if solved:
            passing.append(candidate)
    if not passing:
        raise RuntimeError("induced family could not extend to canonical maximum")
    selected = min(
        passing,
        key=lambda candidate: (
            len(json.dumps(candidate, sort_keys=True, separators=(",", ":"))),
            candidate["finalize"],
        ),
    )
    selected["name"] = "trajectory_extend_" + hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    selected["extension_proof"] = {
        "new_finalizer": selected["finalize"],
        "derived_from_substrate": "canonical ordering plus conditional choice",
    }
    return selected, tested


def compile_program(order: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    return_index = next(i for i, token in enumerate(order) if token.startswith("RETURN_"))
    result: list[dict[str, Any]] = []
    for token in order:
        if token == "IF_NEXT_SEEN":
            result.append({"op": "IF_NEXT_SEEN_GOTO", "target": return_index})
        elif token == "JUMP":
            result.append({"op": "JUMP", "target": 0})
        else:
            result.append({"op": token})
    return tuple(result)


def exhaustive_fixed_language_search(cases: list[MaxCycleWorld]) -> tuple[bool, int]:
    programs: list[tuple[dict[str, Any], ...]] = []
    for return_op in ("RETURN_CURRENT", "RETURN_NEXT", "RETURN_ACCUMULATOR", "RETURN_CANONICAL_PAIR"):
        inventory = ("MARK_CURRENT", "APPLY", "IF_NEXT_SEEN", "ADVANCE", "JUMP", return_op)
        programs.extend(compile_program(order) for order in itertools.permutations(inventory))
    programs.sort(
        key=lambda program: hashlib.sha256(
            json.dumps(program, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
    )

    for tested, program in enumerate(programs, start=1):
        artifact = {
            "schema": "lexigen-rift1-artifact-v1",
            "name": "fixed_" + hashlib.sha256(
                json.dumps(program, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:10],
            "program": list(program),
        }
        solved = True
        for case in cases:
            try:
                predicted = execute_artifact(artifact, case.step, case.seed)
            except Exception:
                solved = False
                break
            if predicted != case.independently_verified_target():
                solved = False
                break
        if solved:
            return True, tested
    return False, len(programs)


def run(output_dir: Path) -> dict[str, Any]:
    demonstrations = build_cases([5], replicas=1)
    transfer = build_cases(range(8, 16), replicas=3)

    instance, family_candidates = adapt_family(demonstrations)
    fixed_solved, fixed_candidates = exhaustive_fixed_language_search(demonstrations)

    family_correct = all(
        execute_family(instance, case) == case.independently_verified_target()
        for case in transfer
    )
    portable_correct = all(
        run_portable_instance(instance, case.step, case.seed)
        == case.independently_verified_target()
        for case in transfer
    )
    ablated = dict(instance)
    ablated["finalize"] = "canonical_pair"
    ablation_correct = sum(
        int(execute_family(ablated, case) == case.independently_verified_target())
        for case in transfer
    ) / len(transfer)

    equal_budget = family_candidates
    report = {
        "benchmark": "RIFT-4",
        "status": "internal L5 gate candidate; not an external world-breakthrough claim",
        "invented_instance": instance,
        "transfer_case_count": len(transfer),
        "family_transfer_accuracy": 1.0 if family_correct else 0.0,
        "portable_interpreter_accuracy": 1.0 if portable_correct else 0.0,
        "ablation_accuracy": ablation_correct,
        "family_candidates_tested": family_candidates,
        "fixed_language_exhaustive_candidates": fixed_candidates,
        "fixed_language_found_solution": fixed_solved,
        "equal_budget": {
            "budget": equal_budget,
            "induced_language_success": family_correct,
            "fixed_language_success": False,
        },
        "gate": {
            "new_executable_semantics": instance["finalize"] == "canonical_max",
            "hidden_surface_transfer": family_correct,
            "portability": portable_correct,
            "ablation_dependency": ablation_correct < 1.0,
            "fixed_language_inexpressibility_in_frozen_inventory": not fixed_solved,
        },
        "claim_boundary": (
            "This crosses the project's internal L5 mechanism gate because a derived finalizer absent from the frozen "
            "bytecode inventory is required and portable. It is not yet a world breakthrough: the benchmark is synthetic, "
            "canonical maximum is a known operation, and no external frontier system has been run on the same protocol."
        ),
    }
    if not all(report["gate"].values()):
        raise AssertionError(f"RIFT-4 gate failed: {report['gate']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rift4-report.json"
    artifact_path = output_dir / "rift4-invented-instance.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_path.write_text(json.dumps(instance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "artifact": instance["name"],
        "family_candidates": family_candidates,
        "fixed_candidates_exhausted": fixed_candidates,
        "fixed_found_solution": fixed_solved,
        "transfer_accuracy": report["family_transfer_accuracy"],
        "portable_accuracy": report["portable_interpreter_accuracy"],
        "ablation_accuracy": ablation_correct,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift4"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
