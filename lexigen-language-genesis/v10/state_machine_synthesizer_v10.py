from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from state_machine_runtime_v10 import Grid, StateMachineError, canonical_json, execute_machine


def changed_colour_roles(examples: Sequence[tuple[Grid, Grid]]) -> tuple[list[int], list[int], list[int]]:
    backgrounds: set[int] = set()
    traces: set[int] = set()
    preserved: set[int] = set()
    for source, target in examples:
        for before_row, after_row in zip(source, target):
            for before, after in zip(before_row, after_row):
                if before != after:
                    backgrounds.add(before)
                    traces.add(after)
                elif before != after:
                    pass
                elif before not in backgrounds:
                    preserved.add(before)
    all_colours = sorted({value for pair in examples for grid in pair for row in grid for value in row})
    obstacle_candidates = sorted(set(all_colours) - backgrounds - traces)
    return sorted(backgrounds), sorted(traces), obstacle_candidates


def machine_length(machine: dict[str, Any]) -> int:
    costs = {
        "away_from_boundary": 1,
        "toward_boundary": 1,
        "forward_is_obstacle": 1,
        "forward_is_nonbackground": 2,
        "away_from_lateral_obstacle": 1,
        "toward_lateral_obstacle": 1,
        "always_left": 0,
        "always_right": 0,
    }
    return 10 + sum(
        costs[value]
        for value in (
            machine["seed"]["direction_mode"],
            machine["transition"]["trigger"],
            machine["transition"]["turn"],
        )
    )


@dataclass(frozen=True)
class StateMachineSynthesisResult:
    machine: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int


def candidate_machines(examples: Sequence[tuple[Grid, Grid]]):
    backgrounds, traces, obstacles = changed_colour_roles(examples)
    for background, trace, obstacle, seed_mode, trigger, turn in itertools.product(
        backgrounds,
        traces,
        obstacles,
        ("away_from_boundary", "toward_boundary"),
        ("forward_is_obstacle", "forward_is_nonbackground"),
        (
            "away_from_lateral_obstacle",
            "toward_lateral_obstacle",
            "always_left",
            "always_right",
        ),
    ):
        if len({background, trace, obstacle}) != 3:
            continue
        yield {
            "schema": "lexigen-sensor-state-machine-v1",
            "types": {
                "state": "(Point,Vector)",
                "sensor": "Grid×Point→Colour",
                "transition": "State×Sensors→State",
                "output": "Grid",
            },
            "colours": {
                "background": background,
                "trace": trace,
                "obstacle": obstacle,
            },
            "seed": {
                "op": "derive_heading_from_boundary_seed",
                "direction_mode": seed_mode,
            },
            "transition": {
                "op": "conditional_sensor_transition",
                "trigger": trigger,
                "turn": turn,
            },
            "execution": {
                "op": "iterate_until_outside",
                "max_steps_factor": 8,
            },
        }


def synthesize_state_machine(examples: Sequence[tuple[Grid, Grid]]) -> StateMachineSynthesisResult:
    exact: list[dict[str, Any]] = []
    tested = 0
    for machine in candidate_machines(examples):
        tested += 1
        try:
            if all(execute_machine(machine, source) == target for source, target in examples):
                exact.append(machine)
        except (StateMachineError, ValueError, IndexError, KeyError):
            continue
    if not exact:
        return StateMachineSynthesisResult(None, tested, 0)
    chosen = min(
        exact,
        key=lambda machine: (
            machine_length(machine),
            hashlib.sha256(canonical_json(machine).encode()).digest(),
        ),
    )
    machine = dict(chosen)
    digest = hashlib.sha256(canonical_json(chosen).encode()).hexdigest()
    machine["name"] = "generated_state_machine_" + digest[:12]
    machine["provenance"] = {
        "method": "typed sensor-transition grammar synthesis",
        "candidates_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_state_machine_substrate": True,
    }
    return StateMachineSynthesisResult(machine, tested, len(exact))
