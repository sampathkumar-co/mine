from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Iterable

from artifact_runtime import ArtifactRuntimeError, execute_artifact
from rift0 import World


@dataclass(frozen=True)
class SynthesisResult:
    artifact: dict[str, Any]
    programs_tested: int
    counterexample_rounds: int
    active_case_names: tuple[str, ...]


def _artifact_for(program: tuple[dict[str, Any], ...], evidence: dict[str, Any]) -> dict[str, Any]:
    canonical_program = json.dumps(program, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence_bytes = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "lexigen-language-artifact-v1",
        "name": f"synth_{hashlib.sha256(canonical_program).hexdigest()[:10]}",
        "signature": "((State -> State), State) -> State",
        "termination_contract": "bounded by runtime instruction budget; monotonicity may be asserted by program",
        "program": list(program),
        "provenance": {
            "method": "counterexample-guided enumerative bytecode synthesis",
            "program_sha256": hashlib.sha256(canonical_program).hexdigest(),
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "warning": (
                "Opcode meanings and the opcode inventory remain human supplied. "
                "This is autonomous composition inside a fixed meta-language, not autonomous semantic invention."
            ),
        },
    }


def _instruction_variants(length: int) -> tuple[dict[str, Any], ...]:
    base = (
        {"op": "APPLY_STEP"},
        {"op": "RETURN_IF_STABLE"},
        {"op": "ASSERT_MONOTONE"},
        {"op": "ADVANCE"},
        {"op": "RETURN"},
    )
    jumps = tuple({"op": "JUMP", "target": target} for target in range(length))
    return base + jumps


def _candidate_programs(max_length: int = 6) -> Iterable[tuple[dict[str, Any], ...]]:
    # The synthesizer receives a generic opcode inventory, not a program template.
    # Cheap semantic coverage filters remove programs that cannot express repeated
    # transition, state advancement, conditional termination, and control flow.
    required = {"APPLY_STEP", "RETURN_IF_STABLE", "ADVANCE", "JUMP"}
    for length in range(1, max_length + 1):
        variants = _instruction_variants(length)
        for program in itertools.product(variants, repeat=length):
            ops = {str(instruction["op"]) for instruction in program}
            if not required.issubset(ops):
                continue
            if sum(1 for instruction in program if instruction["op"] == "JUMP") > 2:
                continue
            yield program


def _solves(artifact: dict[str, Any], cases: list[World], max_instructions: int) -> bool:
    for case in cases:
        try:
            predicted = execute_artifact(
                artifact,
                case.step,
                case.seed,
                max_instructions=max_instructions,
            )
        except (ArtifactRuntimeError, ValueError):
            return False
        if predicted != case.independently_verified_target():
            return False
    return True


def synthesize(
    cases: list[World],
    evidence: dict[str, Any],
    *,
    max_length: int = 6,
    max_instructions: int = 256,
) -> SynthesisResult:
    if not cases:
        raise ValueError("at least one synthesis case is required")

    active = [cases[0]]
    tested = 0
    counterexample_rounds = 0

    while True:
        candidate: dict[str, Any] | None = None
        for program in _candidate_programs(max_length=max_length):
            tested += 1
            artifact = _artifact_for(program, evidence)
            if _solves(artifact, active, max_instructions):
                candidate = artifact
                break
        if candidate is None:
            raise RuntimeError("no bytecode artifact satisfied active counterexamples")

        failing = None
        active_names = {case.name for case in active}
        for case in cases:
            if case.name in active_names:
                continue
            if not _solves(candidate, [case], max_instructions):
                failing = case
                break
        if failing is None:
            return SynthesisResult(
                artifact=candidate,
                programs_tested=tested,
                counterexample_rounds=counterexample_rounds,
                active_case_names=tuple(case.name for case in active),
            )

        active.append(failing)
        counterexample_rounds += 1
