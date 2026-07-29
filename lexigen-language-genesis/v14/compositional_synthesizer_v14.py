from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
V13 = ROOT / "v13"
if str(V13) not in sys.path:
    sys.path.insert(0, str(V13))

from latent_runtime_v13 import Grid, LatentRuntimeError  # noqa: E402
from latent_runtime_v13_ext4 import execute_program as execute_stage  # noqa: E402
from latent_synthesizer_v13 import canonical_json  # noqa: E402
from latent_synthesizer_v13_ext3 import (  # noqa: E402
    candidate_programs,
    description_length,
)
from latent_synthesizer_v13_final import VALIDATED_OPERATORS  # noqa: E402

Pipeline = tuple[dict[str, Any], ...]
Signature = tuple[Grid, ...]


@dataclass(frozen=True)
class CompositionResult:
    pipeline: Pipeline | None
    candidates_tested: int
    signatures_seen: int
    inventory_states: int
    exact_pipeline_count: int


def pipeline_cost(pipeline: Pipeline) -> int:
    return sum(description_length(stage) for stage in pipeline) + 2 * len(pipeline)


def pipeline_key(pipeline: Pipeline) -> tuple[int, bytes]:
    encoded = canonical_json(list(pipeline))
    return pipeline_cost(pipeline), hashlib.sha256(encoded.encode()).digest()


def execute_pipeline(pipeline: Pipeline, grid: Grid) -> Grid:
    current = grid
    for stage in pipeline:
        current = execute_stage(stage, current)
        if not current or not current[0] or len(current) > 60 or len(current[0]) > 60:
            raise LatentRuntimeError("intermediate grid exceeds v14 size budget")
    return current


def stage_inventory(signature: Signature, targets: Signature) -> tuple[dict[str, Any], ...]:
    pairs = tuple(zip(signature, targets))
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidate_programs(pairs):
        if candidate.get("operator") not in VALIDATED_OPERATORS:
            continue
        unique.setdefault(canonical_json(candidate), candidate)
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda text: hashlib.sha256(text.encode()).digest(),
        )
    )


def synthesize_composition(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    max_depth: int = 3,
    candidate_budget: int = 250_000,
) -> CompositionResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    inputs: Signature = tuple(source for source, _ in examples)
    targets: Signature = tuple(target for _, target in examples)
    if inputs == targets:
        return CompositionResult(tuple(), 0, 1, 0, 1)

    tested = 0
    inventory_states = 0
    visited: dict[Signature, Pipeline] = {inputs: tuple()}
    frontier: dict[Signature, Pipeline] = {inputs: tuple()}

    for _depth in range(1, max_depth + 1):
        next_frontier: dict[Signature, Pipeline] = {}
        exact: list[Pipeline] = []
        for signature, pipeline in sorted(frontier.items(), key=lambda item: pipeline_key(item[1])):
            inventory = stage_inventory(signature, targets)
            inventory_states += 1
            for stage in inventory:
                tested += 1
                if tested > candidate_budget:
                    chosen = min(exact, key=pipeline_key) if exact else None
                    return CompositionResult(chosen, tested - 1, len(visited), inventory_states, len(exact))
                try:
                    transformed = tuple(execute_stage(stage, grid) for grid in signature)
                except (LatentRuntimeError, ValueError, IndexError, KeyError):
                    continue
                candidate = pipeline + (stage,)
                if transformed == targets:
                    exact.append(candidate)
                    continue
                previous = visited.get(transformed)
                if previous is None or pipeline_key(candidate) < pipeline_key(previous):
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        if exact:
            return CompositionResult(min(exact, key=pipeline_key), tested, len(visited), inventory_states, len(exact))
        frontier = next_frontier
        if not frontier:
            break
    return CompositionResult(None, tested, len(visited), inventory_states, 0)
