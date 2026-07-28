from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from rift1 import State, execute_artifact
from rift2 import canonical

SURFACES = ("graph", "rules", "grid")


@dataclass(frozen=True)
class CycleQueryWorld:
    name: str
    surface: str
    transitions: tuple[tuple[str, str], ...]
    seed: State
    query: str

    def _mapping(self) -> dict[str, str]:
        return dict(self.transitions)

    def step(self, state: State) -> State:
        if len(state) != 1:
            raise ValueError("cycle query worlds require singleton states")
        token = next(iter(state))
        return frozenset({self._mapping()[token]})

    def independently_verified_target(self) -> State:
        mapping = self._mapping()
        current = self.seed
        seen = {canonical(current)}
        while True:
            nxt = frozenset({mapping[next(iter(current))]})
            if canonical(nxt) in seen:
                if self.query == "cycle_predecessor":
                    return current
                if self.query == "cycle_entry":
                    return nxt
                raise ValueError(f"unknown query: {self.query}")
            seen.add(canonical(nxt))
            current = nxt


def prefix(surface: str, replica: int) -> str:
    return {
        "graph": f"node{replica}_",
        "rules": f"fact{replica}_",
        "grid": f"tile{replica}_",
    }[surface]


def make_world(query: str, depth: int, surface: str, replica: int) -> CycleQueryWorld:
    p = prefix(surface, replica)
    transient = [f"{p}t{i}" for i in range(depth)]
    cycle = [f"{p}cycle_z", f"{p}cycle_a", f"{p}cycle_m"]
    transitions: list[tuple[str, str]] = []
    transitions.extend((transient[i], transient[i + 1]) for i in range(len(transient) - 1))
    transitions.append((transient[-1], cycle[0]))
    transitions.extend(((cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[2], cycle[0])))
    return CycleQueryWorld(
        name=f"{query}-{surface}-d{depth}-i{replica}",
        surface=surface,
        transitions=tuple(transitions),
        seed=frozenset({transient[0]}),
        query=query,
    )


def build_cases(query: str, depths: Iterable[int], replicas: int = 1) -> list[CycleQueryWorld]:
    return [
        make_world(query, depth, surface, replica)
        for depth in depths
        for replica in range(replicas)
        for surface in SURFACES
    ]


def execute_extended_family(
    instance: dict[str, Any],
    step: Callable[[State], State],
    seed: State,
    *,
    max_steps: int = 2_000,
) -> State:
    stop = instance["stop"]
    finalize = instance["finalize"]
    current = seed
    trace: list[State] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(max_steps):
        trace.append(current)
        seen.add(canonical(current))
        nxt = step(current)
        should_stop = nxt == current if stop == "stable" else canonical(nxt) in seen
        if should_stop:
            if finalize == "current":
                return current
            if finalize == "next":
                return nxt
            if finalize == "trace_union":
                return frozenset().union(*trace)
            if finalize == "canonical_pair":
                return min((current, nxt), key=canonical)
            raise ValueError(f"unknown finalizer: {finalize}")
        current = nxt
    raise RuntimeError("extended family exhausted its step budget")


def fits_family(instance: dict[str, Any], cases: list[CycleQueryWorld]) -> bool:
    for case in cases:
        try:
            predicted = execute_extended_family(instance, case.step, case.seed)
        except (RuntimeError, ValueError):
            return False
        if predicted != case.independently_verified_target():
            return False
    return True


def adapt_family(demonstrations: list[CycleQueryWorld]) -> tuple[dict[str, Any], int, bool]:
    existing_finalizers = ("current", "trace_union", "canonical_pair")
    substrate_extensions = ("next",)
    candidates = [
        {"schema": "lexigen-trajectory-operator-instance-v2", "stop": stop, "finalize": finalize}
        for stop in ("stable", "repeat")
        for finalize in existing_finalizers + substrate_extensions
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
        if fits_family(candidate, demonstrations):
            passing.append(candidate)
    if not passing:
        raise RuntimeError("family could not adapt to demonstrations")
    selected = min(
        passing,
        key=lambda candidate: (
            len(json.dumps(candidate, sort_keys=True, separators=(",", ":"))),
            candidate["finalize"],
        ),
    )
    extended_vocabulary = selected["finalize"] in substrate_extensions
    selected["name"] = "trajectory_adapt_" + hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return selected, tested, extended_vocabulary


def compile_program(order: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    return_index = next(i for i, token in enumerate(order) if token.startswith("RETURN_"))
    program = []
    for token in order:
        if token == "IF_NEXT_SEEN":
            program.append({"op": "IF_NEXT_SEEN_GOTO", "target": return_index})
        elif token == "JUMP":
            program.append({"op": "JUMP", "target": 0})
        else:
            program.append({"op": token})
    return tuple(program)


def raw_program_search(demonstrations: list[CycleQueryWorld]) -> tuple[dict[str, Any], int]:
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
            "name": "raw_" + hashlib.sha256(
                json.dumps(program, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:10],
            "program": list(program),
        }
        solved = True
        for case in demonstrations:
            try:
                predicted = execute_artifact(artifact, case.step, case.seed)
            except Exception:
                solved = False
                break
            if predicted != case.independently_verified_target():
                solved = False
                break
        if solved:
            return artifact, tested
    raise RuntimeError("raw program search failed")


def raw_fits(artifact: dict[str, Any], cases: list[CycleQueryWorld]) -> bool:
    for case in cases:
        try:
            predicted = execute_artifact(artifact, case.step, case.seed)
        except Exception:
            return False
        if predicted != case.independently_verified_target():
            return False
    return True


def run(output_dir: Path) -> dict[str, Any]:
    episodes: dict[str, Any] = {}
    for query in ("cycle_predecessor", "cycle_entry"):
        demonstrations = build_cases(query, [5], replicas=1)
        transfer = build_cases(query, range(8, 15), replicas=2)

        adapted, family_candidates_tested, vocabulary_extended = adapt_family(demonstrations)
        raw_artifact, raw_candidates_tested = raw_program_search(demonstrations)
        if not fits_family(adapted, transfer):
            raise AssertionError(f"adapted family failed transfer: {query}")
        if not raw_fits(raw_artifact, transfer):
            raise AssertionError(f"raw search artifact failed transfer: {query}")

        episodes[query] = {
            "adapted_instance": adapted,
            "family_candidates_tested": family_candidates_tested,
            "raw_program": raw_artifact,
            "raw_candidates_tested": raw_candidates_tested,
            "search_reduction": raw_candidates_tested / family_candidates_tested,
            "vocabulary_extended": vocabulary_extended,
            "transfer_accuracy": 1.0,
        }

    report = {
        "benchmark": "RIFT-3",
        "status": "compositional capability growth from an induced language; not a world breakthrough claim",
        "episodes": episodes,
        "claim_boundary": (
            "The induced family recombines previously learned semantic dimensions and adds the substrate-level "
            "finalizer `next`. This is verified capability growth, but prior meta-learning and library-learning "
            "research prevents treating it as an external AI breakthrough without stronger baselines."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rift3-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        query: {
            "adapted": result["adapted_instance"]["name"],
            "family_candidates": result["family_candidates_tested"],
            "raw_candidates": result["raw_candidates_tested"],
            "search_reduction": result["search_reduction"],
            "vocabulary_extended": result["vocabulary_extended"],
        }
        for query, result in episodes.items()
    }
    summary["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift3"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
