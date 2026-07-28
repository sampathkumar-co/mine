from __future__ import annotations

from collections.abc import Callable
from typing import Any

State = frozenset[str]
Step = Callable[[State], State]


class ArtifactRuntimeError(RuntimeError):
    pass


def execute_artifact(
    artifact: dict[str, Any],
    step: Step,
    seed: State,
    *,
    max_instructions: int = 100_000,
) -> State:
    """Execute a serialized control-language artifact without the invention engine."""
    if artifact.get("schema") != "lexigen-language-artifact-v1":
        raise ArtifactRuntimeError("unsupported artifact schema")
    program = artifact.get("program")
    if not isinstance(program, list) or not program:
        raise ArtifactRuntimeError("artifact program must be a non-empty list")

    current = seed
    updated = seed
    pc = 0
    executed = 0

    while True:
        if executed >= max_instructions:
            raise ArtifactRuntimeError("instruction budget exhausted")
        if not 0 <= pc < len(program):
            raise ArtifactRuntimeError(f"program counter out of range: {pc}")
        instruction = program[pc]
        if not isinstance(instruction, dict):
            raise ArtifactRuntimeError("instruction must be an object")
        op = instruction.get("op")
        executed += 1

        if op == "APPLY_STEP":
            updated = step(current)
            pc += 1
        elif op == "RETURN_IF_STABLE":
            if updated == current:
                return current
            pc += 1
        elif op == "ASSERT_MONOTONE":
            if not current.issubset(updated):
                raise ArtifactRuntimeError("monotonicity contract violated")
            pc += 1
        elif op == "ADVANCE":
            current = updated
            pc += 1
        elif op == "JUMP":
            target = instruction.get("target")
            if not isinstance(target, int):
                raise ArtifactRuntimeError("JUMP target must be an integer")
            pc = target
        elif op == "RETURN":
            return current
        else:
            raise ArtifactRuntimeError(f"unknown opcode: {op!r}")
