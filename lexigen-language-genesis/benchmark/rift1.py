from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

State = frozenset[str]


class World(Protocol):
    name: str
    surface: str
    mechanism: str
    seed: State

    def step(self, state: State) -> State: ...

    def independently_verified_target(self) -> State: ...


@dataclass(frozen=True)
class ClosureWorld:
    name: str
    surface: str
    edges: tuple[tuple[str, str], ...]
    seed: State
    mechanism: str = "closure"

    def step(self, state: State) -> State:
        return frozenset(set(state) | {dst for src, dst in self.edges if src in state})

    def independently_verified_target(self) -> State:
        adjacency: dict[str, set[str]] = {}
        for src, dst in self.edges:
            adjacency.setdefault(src, set()).add(dst)
        reached = set(self.seed)
        queue = list(sorted(self.seed))
        while queue:
            current = queue.pop(0)
            for nxt in sorted(adjacency.get(current, ())):
                if nxt not in reached:
                    reached.add(nxt)
                    queue.append(nxt)
        return frozenset(reached)


@dataclass(frozen=True)
class TrajectoryUnionWorld:
    name: str
    surface: str
    transitions: tuple[tuple[str, str], ...]
    seed: State
    mechanism: str = "trajectory_union"

    def _mapping(self) -> dict[str, str]:
        return dict(self.transitions)

    def step(self, state: State) -> State:
        if len(state) != 1:
            raise ValueError("trajectory worlds require singleton states")
        token = next(iter(state))
        return frozenset({self._mapping()[token]})

    def independently_verified_target(self) -> State:
        mapping = self._mapping()
        current = next(iter(self.seed))
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            current = mapping[current]
        return frozenset(seen)


@dataclass(frozen=True)
class TwoCycleWorld:
    name: str
    surface: str
    transitions: tuple[tuple[str, str], ...]
    seed: State
    mechanism: str = "two_cycle_canonical"

    def _mapping(self) -> dict[str, str]:
        return dict(self.transitions)

    def step(self, state: State) -> State:
        if len(state) != 1:
            raise ValueError("cycle worlds require singleton states")
        token = next(iter(state))
        return frozenset({self._mapping()[token]})

    def independently_verified_target(self) -> State:
        mapping = self._mapping()
        current = next(iter(self.seed))
        seen_at: dict[str, int] = {}
        trajectory: list[str] = []
        while current not in seen_at:
            seen_at[current] = len(trajectory)
            trajectory.append(current)
            current = mapping[current]
        cycle = trajectory[seen_at[current] :]
        if len(cycle) != 2:
            raise ValueError("RIFT-1 two-cycle world must have period two")
        return frozenset({min(cycle)})


def _surface_prefix(surface: str, replica: int) -> str:
    return {
        "graph": f"g{replica}_",
        "rules": f"r{replica}_",
        "grid": f"cell{replica}_",
    }[surface]


def make_closure_world(depth: int, surface: str, replica: int) -> ClosureWorld:
    prefix = _surface_prefix(surface, replica)
    edges = tuple((f"{prefix}{i}", f"{prefix}{i + 1}") for i in range(depth))
    return ClosureWorld(
        name=f"closure-{surface}-d{depth}-i{replica}",
        surface=surface,
        edges=edges,
        seed=frozenset({f"{prefix}0"}),
    )


def make_trajectory_world(depth: int, surface: str, replica: int) -> TrajectoryUnionWorld:
    prefix = _surface_prefix(surface, replica)
    tokens = [f"{prefix}{i}" for i in range(depth + 2)]
    transitions = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    transitions.append((tokens[-1], tokens[depth // 2]))
    return TrajectoryUnionWorld(
        name=f"union-{surface}-d{depth}-i{replica}",
        surface=surface,
        transitions=tuple(transitions),
        seed=frozenset({tokens[0]}),
    )


def make_two_cycle_world(depth: int, surface: str, replica: int) -> TwoCycleWorld:
    prefix = _surface_prefix(surface, replica)
    transient = [f"{prefix}t{i}" for i in range(depth)]
    # Reverse lexical order deliberately prevents "return current" from working reliably.
    cycle_a = f"{prefix}z_cycle"
    cycle_b = f"{prefix}a_cycle"
    transitions: list[tuple[str, str]] = []
    if transient:
        transitions.extend((transient[i], transient[i + 1]) for i in range(len(transient) - 1))
        transitions.append((transient[-1], cycle_a))
        seed = frozenset({transient[0]})
    else:
        seed = frozenset({cycle_a})
    transitions.extend(((cycle_a, cycle_b), (cycle_b, cycle_a)))
    return TwoCycleWorld(
        name=f"cycle-{surface}-d{depth}-i{replica}",
        surface=surface,
        transitions=tuple(transitions),
        seed=seed,
    )


def build_cases(mechanism: str, depths: Iterable[int], replicas: int = 1) -> list[World]:
    constructors: dict[str, Callable[[int, str, int], World]] = {
        "closure": make_closure_world,
        "trajectory_union": make_trajectory_world,
        "two_cycle_canonical": make_two_cycle_world,
    }
    constructor = constructors[mechanism]
    cases: list[World] = []
    for depth in depths:
        for replica in range(replicas):
            for surface in ("graph", "rules", "grid"):
                cases.append(constructor(depth, surface, replica))
    return cases


def bounded_three(step: Callable[[State], State], seed: State) -> State:
    current = seed
    for _ in range(3):
        current = step(current)
    return current


def _canonical(state: State) -> tuple[str, ...]:
    return tuple(sorted(state))


class RuntimeErrorArtifact(RuntimeError):
    pass


def execute_artifact(
    artifact: dict[str, Any],
    step: Callable[[State], State],
    seed: State,
    *,
    max_instructions: int = 2_000,
) -> State:
    if artifact.get("schema") != "lexigen-rift1-artifact-v1":
        raise RuntimeErrorArtifact("unsupported artifact schema")
    program = artifact.get("program")
    if not isinstance(program, list) or not program:
        raise RuntimeErrorArtifact("empty program")

    current = seed
    next_state = seed
    accumulator: set[str] = set()
    seen: set[tuple[str, ...]] = set()
    pc = 0

    for _ in range(max_instructions):
        if not 0 <= pc < len(program):
            raise RuntimeErrorArtifact(f"program counter out of range: {pc}")
        instruction = program[pc]
        op = instruction.get("op")

        if op == "APPLY":
            next_state = step(current)
            pc += 1
        elif op == "ADVANCE":
            current = next_state
            pc += 1
        elif op == "MARK_CURRENT":
            seen.add(_canonical(current))
            pc += 1
        elif op == "ACCUMULATE_CURRENT":
            accumulator.update(current)
            pc += 1
        elif op == "IF_EQUAL_GOTO":
            pc = int(instruction["target"]) if next_state == current else pc + 1
        elif op == "IF_NEXT_SEEN_GOTO":
            pc = int(instruction["target"]) if _canonical(next_state) in seen else pc + 1
        elif op == "JUMP":
            pc = int(instruction["target"])
        elif op == "RETURN_CURRENT":
            return current
        elif op == "RETURN_NEXT":
            return next_state
        elif op == "RETURN_ACCUMULATOR":
            return frozenset(accumulator)
        elif op == "RETURN_CANONICAL_PAIR":
            return min((current, next_state), key=_canonical)
        else:
            raise RuntimeErrorArtifact(f"unknown opcode: {op!r}")
    raise RuntimeErrorArtifact("instruction budget exhausted")


def trace(case: World, max_steps: int = 256) -> tuple[list[State], State]:
    states = [case.seed]
    seen = {_canonical(case.seed)}
    current = case.seed
    for _ in range(max_steps):
        nxt = case.step(current)
        if _canonical(nxt) in seen:
            return states, nxt
        states.append(nxt)
        seen.add(_canonical(nxt))
        current = nxt
    raise RuntimeError("trace did not repeat")


def infer_instruction_inventory(cases: list[World]) -> tuple[str, ...]:
    """Infer required capabilities from demonstrations, but not their program order."""
    relations: set[str] = set()
    for case in cases:
        states, repeated = trace(case)
        target = case.independently_verified_target()
        last = states[-1]
        stable = repeated == last and case.step(repeated) == repeated
        union = frozenset().union(*states)
        canonical_pair = min((last, repeated), key=_canonical)
        if stable and target == repeated:
            relations.add("stable")
        elif target == union:
            relations.add("union")
        elif target == canonical_pair:
            relations.add("canonical_pair")
        else:
            relations.add("unknown")

    if relations == {"stable"}:
        return ("APPLY", "IF_EQUAL", "ADVANCE", "JUMP", "RETURN_CURRENT")
    if relations == {"union"}:
        return (
            "MARK_CURRENT",
            "ACCUMULATE_CURRENT",
            "APPLY",
            "IF_NEXT_SEEN",
            "ADVANCE",
            "JUMP",
            "RETURN_ACCUMULATOR",
        )
    if relations == {"canonical_pair"}:
        return (
            "MARK_CURRENT",
            "APPLY",
            "IF_NEXT_SEEN",
            "ADVANCE",
            "JUMP",
            "RETURN_CANONICAL_PAIR",
        )
    raise RuntimeError(f"demonstrations do not support one control law: {sorted(relations)}")


def _compile_order(order: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    return_index = next(i for i, token in enumerate(order) if token.startswith("RETURN_"))
    program: list[dict[str, Any]] = []
    for token in order:
        if token == "IF_EQUAL":
            program.append({"op": "IF_EQUAL_GOTO", "target": return_index})
        elif token == "IF_NEXT_SEEN":
            program.append({"op": "IF_NEXT_SEEN_GOTO", "target": return_index})
        elif token == "JUMP":
            program.append({"op": "JUMP", "target": 0})
        else:
            program.append({"op": token})
    return tuple(program)


def _program_bytes(program: tuple[dict[str, Any], ...]) -> bytes:
    return json.dumps(program, sort_keys=True, separators=(",", ":")).encode("utf-8")


def candidate_programs(inventory: tuple[str, ...]) -> list[tuple[dict[str, Any], ...]]:
    programs = [_compile_order(order) for order in itertools.permutations(inventory)]
    programs.sort(key=lambda program: hashlib.sha256(_program_bytes(program)).digest())
    return programs


def artifact_for(program: tuple[dict[str, Any], ...], inventory: tuple[str, ...]) -> dict[str, Any]:
    encoded = _program_bytes(program)
    return {
        "schema": "lexigen-rift1-artifact-v1",
        "name": f"rift1_{hashlib.sha256(encoded).hexdigest()[:12]}",
        "program": list(program),
        "inferred_inventory": list(inventory),
        "provenance": {
            "method": "demonstration-profiled counterexample-guided program synthesis",
            "program_sha256": hashlib.sha256(encoded).hexdigest(),
            "candidate_order": "SHA-256",
            "limitation": "opcode meanings remain human supplied",
        },
    }


def solves(artifact: dict[str, Any], cases: list[World]) -> bool:
    for case in cases:
        try:
            predicted = execute_artifact(artifact, case.step, case.seed)
        except (RuntimeErrorArtifact, ValueError):
            return False
        if predicted != case.independently_verified_target():
            return False
    return True


def synthesize(cases: list[World]) -> dict[str, Any]:
    inventory = infer_instruction_inventory(cases)
    active = [cases[0]]
    tested = 0
    rounds = 0
    programs = candidate_programs(inventory)

    while True:
        selected: dict[str, Any] | None = None
        for program in programs:
            tested += 1
            candidate = artifact_for(program, inventory)
            if solves(candidate, active):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError("no artifact solves active counterexamples")

        active_names = {case.name for case in active}
        failing = next(
            (case for case in cases if case.name not in active_names and not solves(selected, [case])),
            None,
        )
        if failing is None:
            selected["synthesis"] = {
                "programs_tested": tested,
                "counterexample_rounds": rounds,
                "active_cases": [case.name for case in active],
            }
            return selected
        active.append(failing)
        rounds += 1


def evaluate_solver(
    solver: Callable[[Callable[[State], State], State], State], cases: list[World]
) -> dict[str, Any]:
    records = []
    for case in cases:
        predicted = solver(case.step, case.seed)
        expected = case.independently_verified_target()
        records.append(
            {
                "name": case.name,
                "surface": case.surface,
                "correct": predicted == expected,
            }
        )
    return {
        "accuracy": sum(int(record["correct"]) for record in records) / len(records),
        "by_surface": {
            surface: sum(int(r["correct"]) for r in records if r["surface"] == surface)
            / sum(1 for r in records if r["surface"] == surface)
            for surface in ("graph", "rules", "grid")
        },
        "records": records,
    }


def run(output_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "benchmark": "RIFT-1",
        "status": "fixed-meta-language multi-mechanism synthesis; no breakthrough claim",
        "mechanisms": {},
    }

    for mechanism in ("closure", "trajectory_union", "two_cycle_canonical"):
        diagnostic = build_cases(mechanism, range(4, 7), replicas=2)
        transfer = build_cases(mechanism, range(7, 13), replicas=2)
        artifact = synthesize(diagnostic)

        def artifact_solver(step: Callable[[State], State], seed: State, a: dict[str, Any] = artifact) -> State:
            return execute_artifact(a, step, seed)

        baseline = evaluate_solver(bounded_three, transfer)
        synthesized = evaluate_solver(artifact_solver, transfer)
        if synthesized["accuracy"] != 1.0:
            raise AssertionError(f"{mechanism} artifact failed transfer")
        report["mechanisms"][mechanism] = {
            "artifact": artifact,
            "baseline_transfer": baseline,
            "synthesized_transfer": synthesized,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rift1-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for mechanism, result in report["mechanisms"].items():
        (output_dir / f"rift1-{mechanism}-artifact.json").write_text(
            json.dumps(result["artifact"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        mechanism: {
            "artifact": result["artifact"]["name"],
            "programs_tested": result["artifact"]["synthesis"]["programs_tested"],
            "baseline_transfer": result["baseline_transfer"]["accuracy"],
            "synthesized_transfer": result["synthesized_transfer"]["accuracy"],
        }
        for mechanism, result in report["mechanisms"].items()
    }
    summary["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift1"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
