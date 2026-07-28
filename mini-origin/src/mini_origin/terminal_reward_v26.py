from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

from .intervention_genesis_v22 import (
    PositionRule,
    information_lower_bound,
    intervention_response,
    observational_equivalence_certificate,
    optimal_worst_case_queries,
    position_rules,
)
from .outcome_alphabet_v23 import ACCEPT, ACTION_NAMES, LEFT, RIGHT
from .predicate_genesis_v25 import LoopEvaluation, Predicate


@dataclass(frozen=True)
class World:
    dimension: int
    root: int
    rho: float


@dataclass(frozen=True)
class TerminalProgram:
    rule: PositionRule
    predicates: tuple[Predicate, ...]
    mapping: tuple[int, ...]
    training_accuracy: float
    development_accuracy: float

    @property
    def complexity(self) -> int:
        return self.rule.complexity + sum(
            predicate.complexity for predicate in self.predicates
        ) + len(self.mapping)

    def text(self) -> str:
        predicates = ";".join(predicate.text() for predicate in self.predicates)
        table = ",".join(
            f"{code}:{ACTION_NAMES[action]}"
            for code, action in enumerate(self.mapping)
        )
        return f"rule={self.rule.name};predicates={predicates};map={table}"


def noiseless_response(
    dimension: int,
    root: int,
    rho: float,
    query: int,
) -> tuple[float | None, float | None]:
    left = None
    right = None
    if query > 0:
        left = rho if root >= query else 0.0
    if query < dimension - 1:
        right = rho if root <= query else 0.0
    return left, right


def encode(
    predicates: tuple[Predicate, ...],
    left: float | None,
    right: float | None,
) -> int:
    return sum(
        int(predicate.evaluate(left, right)) << index
        for index, predicate in enumerate(predicates)
    )


def decode(
    program: TerminalProgram,
    left: float | None,
    right: float | None,
) -> int:
    return program.mapping[encode(program.predicates, left, right)]


def exact_worlds() -> tuple[World, ...]:
    return tuple(
        World(dimension, root, rho)
        for dimension in range(2, 9)
        for rho in (0.35, 0.60, 0.85)
        for root in range(dimension)
    )


def unsupervised_observations(
    seed: int,
    samples: int = 1_600,
) -> list[tuple[float | None, float | None]]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(samples):
        dimension = int(rng.integers(3, 9))
        root = int(rng.integers(0, dimension))
        query = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.30, 0.90))
        rows.append(
            intervention_response(
                seed + index * 65_537,
                dimension,
                root,
                rho,
                query,
                replicates=384,
            )
        )
    return rows


def _predicate_signature(
    predicate: Predicate,
    observations: list[tuple[float | None, float | None]],
) -> bytes:
    bits = np.asarray(
        [predicate.evaluate(left, right) for left, right in observations],
        dtype=np.bool_,
    )
    return np.packbits(bits).tobytes()


def terminal_predicate_grammar(
    observations: list[tuple[float | None, float | None]],
) -> tuple[Predicate, ...]:
    magnitudes = [
        abs(value)
        for left, right in observations
        for value in (left, right)
        if value is not None
    ]
    scale = float(np.quantile(magnitudes, 0.90))
    thresholds = tuple(
        sorted({0.0, 0.15 * scale, 0.25 * scale, 0.35 * scale})
    )
    raw = [Predicate("present", "left"), Predicate("present", "right")]
    for threshold in thresholds:
        raw.extend(
            (
                Predicate("greater", "left", threshold, 2),
                Predicate("greater", "right", threshold, 2),
                Predicate("difference", "right-left", threshold, 3),
                Predicate("difference", "left-right", threshold, 3),
            )
        )
    unique: dict[bytes, Predicate] = {}
    for predicate in raw:
        signature = _predicate_signature(predicate, observations)
        current = unique.get(signature)
        if current is None or (
            predicate.complexity,
            abs(predicate.threshold),
            predicate.text(),
        ) < (
            current.complexity,
            abs(current.threshold),
            current.text(),
        ):
            unique[signature] = predicate
    return tuple(
        sorted(unique.values(), key=lambda value: (value.complexity, value.text()))
    )


def run_noiseless_trial(program: TerminalProgram, world: World) -> bool:
    low = 0
    high = world.dimension - 1
    queries = 0
    budget = information_lower_bound(world.dimension)
    while low < high and queries < budget:
        query = low + program.rule.offset(high - low + 1)
        left, right = noiseless_response(
            world.dimension, world.root, world.rho, query
        )
        action = decode(program, left, right)
        queries += 1
        if action == ACCEPT:
            return query == world.root
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        else:
            raise ValueError(action)
        if low > high:
            return False
    remaining = max(1, high - low + 1)
    prediction = low + (remaining - 1) // 2
    return prediction == world.root


def exact_accuracy(
    program: TerminalProgram,
    worlds: tuple[World, ...],
) -> float:
    return sum(run_noiseless_trial(program, world) for world in worlds) / len(worlds)


def _mapping_space(predicate_count: int):
    return itertools.product((LEFT, ACCEPT, RIGHT), repeat=1 << predicate_count)


def _candidate_program(
    rule: PositionRule,
    predicates: tuple[Predicate, ...],
    mapping: tuple[int, ...],
    training_accuracy: float,
    development_accuracy: float = 0.0,
) -> TerminalProgram:
    return TerminalProgram(
        rule,
        predicates,
        mapping,
        training_accuracy,
        development_accuracy,
    )


def _all_exact_two_predicate_programs(
    grammar: tuple[Predicate, ...],
    worlds: tuple[World, ...],
) -> list[TerminalProgram]:
    exact = []
    for rule in position_rules():
        for predicates in itertools.combinations(grammar, 2):
            for mapping_values in _mapping_space(2):
                mapping = tuple(int(value) for value in mapping_values)
                program = _candidate_program(rule, predicates, mapping, 1.0)
                if all(run_noiseless_trial(program, world) for world in worlds):
                    exact.append(program)
    return exact


def _best_single_predicate_program(
    grammar: tuple[Predicate, ...],
    worlds: tuple[World, ...],
) -> TerminalProgram:
    best = None
    for rule in position_rules():
        for predicate in grammar:
            for mapping_values in _mapping_space(1):
                mapping = tuple(int(value) for value in mapping_values)
                program = _candidate_program(rule, (predicate,), mapping, 0.0)
                accuracy = exact_accuracy(program, worlds)
                candidate = _candidate_program(
                    rule, (predicate,), mapping, accuracy
                )
                score = (
                    accuracy,
                    -candidate.complexity,
                    -rule.complexity,
                    rule.name,
                    predicate.text(),
                    mapping,
                )
                if best is None or score > best[0]:
                    best = (score, candidate)
    assert best is not None
    return best[1]


def run_noisy_trial(
    seed: int,
    program: TerminalProgram,
    dimension: int,
    root: int,
    rho: float,
    replicates: int,
) -> tuple[bool, int, bool, int]:
    low = 0
    high = dimension - 1
    queries = 0
    budget = information_lower_bound(dimension)
    while low < high and queries < budget:
        query = low + program.rule.offset(high - low + 1)
        left, right = intervention_response(
            seed + queries * 7_919,
            dimension,
            root,
            rho,
            query,
            replicates,
        )
        action = decode(program, left, right)
        queries += 1
        if action == ACCEPT:
            return query == root, queries, False, 1
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        else:
            raise ValueError(action)
        if low > high:
            return False, queries, True, 0
    remaining = max(1, high - low + 1)
    prediction = low + (remaining - 1) // 2
    return prediction == root, queries, False, remaining


def evaluate_noisy(
    seed: int,
    program: TerminalProgram,
    dimensions: tuple[int, ...],
    trials: int,
    replicates: int,
    rho_low: float,
    rho_high: float,
) -> LoopEvaluation:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(trials):
        dimension = int(rng.choice(dimensions))
        root = int(rng.integers(0, dimension))
        rho = float(rng.uniform(rho_low, rho_high))
        rows.append(
            run_noisy_trial(
                seed + index * 104_729,
                program,
                dimension,
                root,
                rho,
                replicates,
            )
        )
    return LoopEvaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean([row[1] for row in rows])),
        maximum_queries=max(row[1] for row in rows),
        invalid_transition_rate=float(np.mean([row[2] for row in rows])),
        mean_remaining_candidates=float(np.mean([row[3] for row in rows])),
    )


def search_terminal_programs(
    seed: int,
) -> tuple[TerminalProgram, TerminalProgram, dict[str, object]]:
    observations = unsupervised_observations(seed)
    grammar = terminal_predicate_grammar(observations)
    worlds = exact_worlds()
    single = _best_single_predicate_program(grammar, worlds)
    exact = _all_exact_two_predicate_programs(grammar, worlds)
    if not exact:
        raise RuntimeError("terminal-reward search found no exact two-predicate controller")

    exact.sort(
        key=lambda program: (
            program.complexity,
            program.rule.complexity,
            program.rule.name,
            tuple(predicate.text() for predicate in program.predicates),
            program.mapping,
        )
    )
    finalists = exact[:512]
    development_dimensions = (3, 4, 5, 6, 7, 8, 9, 11)
    evaluated = []
    for index, program in enumerate(finalists):
        evaluation = evaluate_noisy(
            seed + 2_000_003 + index * 1_009,
            program,
            development_dimensions,
            trials=1_200,
            replicates=384,
            rho_low=0.28,
            rho_high=0.92,
        )
        evaluated.append(
            _candidate_program(
                program.rule,
                program.predicates,
                program.mapping,
                1.0,
                evaluation.accuracy,
            )
        )
    best_development = max(program.development_accuracy for program in evaluated)
    eligible = [
        program
        for program in evaluated
        if program.development_accuracy >= best_development - 0.003
    ]
    selected = min(
        eligible,
        key=lambda program: (
            program.complexity,
            program.rule.complexity,
            program.rule.name,
            tuple(predicate.text() for predicate in program.predicates),
            program.mapping,
        ),
    )
    evidence = {
        "supervision": "terminal_root_success_only",
        "step_action_labels_used": False,
        "unsupervised_observation_count": len(observations),
        "grammar_size": len(grammar),
        "training_world_count": len(worlds),
        "exact_two_predicate_program_count": len(exact),
        "development_finalist_count": len(finalists),
        "selected_rule": selected.rule.name,
        "selected_predicates": [
            predicate.text() for predicate in selected.predicates
        ],
        "selected_mapping": [ACTION_NAMES[action] for action in selected.mapping],
        "training_accuracy": selected.training_accuracy,
        "development_accuracy": selected.development_accuracy,
        "single_predicate_training_accuracy": single.training_accuracy,
    }
    return selected, single, evidence


def specialist_program() -> TerminalProgram:
    rule = next(rule for rule in position_rules() if rule.name == "lower_midpoint")
    predicates = (
        Predicate("greater", "right", 0.15, 2),
        Predicate("difference", "right-left", 0.15, 3),
    )
    mapping = (RIGHT, ACCEPT, RIGHT, LEFT)
    return TerminalProgram(rule, predicates, mapping, 1.0, 1.0)


def random_mapping_control(
    seed: int,
    template: TerminalProgram,
    dimensions: tuple[int, ...],
) -> dict[str, float]:
    rng = random.Random(seed)
    scores = []
    for index in range(48):
        mapping = tuple(
            rng.choice((LEFT, ACCEPT, RIGHT))
            for _ in range(1 << len(template.predicates))
        )
        program = TerminalProgram(
            template.rule,
            template.predicates,
            mapping,
            0.0,
            0.0,
        )
        score = evaluate_noisy(
            seed + index * 1_000_003,
            program,
            dimensions,
            trials=420,
            replicates=512,
            rho_low=0.26,
            rho_high=0.95,
        ).accuracy
        scores.append(score)
    return {
        "trials": 48,
        "median_accuracy": float(np.median(scores)),
        "maximum_accuracy": max(scores),
    }


def wrong_query_control(program: TerminalProgram) -> TerminalProgram:
    rule = next(rule for rule in position_rules() if rule.name == "first")
    return TerminalProgram(
        rule,
        program.predicates,
        program.mapping,
        0.0,
        0.0,
    )


def digest(program: TerminalProgram) -> str:
    return hashlib.sha256(program.text().encode("utf-8")).hexdigest()


def serialize(program: TerminalProgram) -> dict[str, object]:
    return {
        "rule": program.rule.name,
        "rule_complexity": program.rule.complexity,
        "predicates": [predicate.text() for predicate in program.predicates],
        "mapping": [
            {"code": code, "action": ACTION_NAMES[action]}
            for code, action in enumerate(program.mapping)
        ],
        "complexity": program.complexity,
        "training_accuracy": program.training_accuracy,
        "development_accuracy": program.development_accuracy,
    }


def run(seed: int = 1101) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    selected, single, search = search_terminal_programs(seed * 10_000 + 83)
    frozen_digest = digest(selected)

    # Hidden dimensions and random trials are created only after the full query,
    # predicate and action-table controller has been frozen.
    hidden_dimensions = (13, 21, 37, 63, 127)
    candidate = evaluate_noisy(
        seed * 10_000 + 11_000_001,
        selected,
        hidden_dimensions,
        trials=3_200,
        replicates=512,
        rho_low=0.26,
        rho_high=0.95,
    )
    single_control = evaluate_noisy(
        seed * 10_000 + 11_000_003,
        single,
        hidden_dimensions,
        trials=3_200,
        replicates=512,
        rho_low=0.26,
        rho_high=0.95,
    )
    wrong_query = evaluate_noisy(
        seed * 10_000 + 11_000_005,
        wrong_query_control(selected),
        hidden_dimensions,
        trials=3_200,
        replicates=512,
        rho_low=0.26,
        rho_high=0.95,
    )
    specialist = evaluate_noisy(
        seed * 10_000 + 11_000_007,
        specialist_program(),
        hidden_dimensions,
        trials=3_200,
        replicates=512,
        rho_low=0.26,
        rho_high=0.95,
    )
    random_control = random_mapping_control(
        seed * 10_000 + 11_000_009,
        selected,
        hidden_dimensions,
    )

    single_gap = candidate.accuracy - single_control.accuracy
    query_gap = candidate.accuracy - wrong_query.accuracy
    specialist_gap = candidate.accuracy - specialist.accuracy
    random_gap = candidate.accuracy - random_control["median_accuracy"]
    query_optimality = all(
        optimal_worst_case_queries(dimension)
        == information_lower_bound(dimension)
        for dimension in hidden_dimensions
    )
    predicate_family = (
        len(selected.predicates) == 2
        and any(
            predicate.kind == "greater" and predicate.channel == "right"
            for predicate in selected.predicates
        )
        and any(
            predicate.kind == "difference"
            and predicate.channel == "right-left"
            for predicate in selected.predicates
        )
    )
    gate = (
        bool(equivalence["exact_within_tolerance"])
        and search["step_action_labels_used"] is False
        and search["supervision"] == "terminal_root_success_only"
        and selected.training_accuracy == 1.0
        and selected.rule.name == "lower_midpoint"
        and predicate_family
        and selected.development_accuracy >= 0.985
        and candidate.accuracy >= 0.985
        and candidate.invalid_transition_rate <= 0.01
        and single_gap >= 0.20
        and query_gap >= 0.20
        and specialist_gap >= -0.01
        and random_gap >= 0.45
        and query_optimality
    )
    return {
        "status": (
            "terminal_reward_controller_genesis_candidate" if gate else "not_yet"
        ),
        "claim_scope": (
            "from raw intervention responses and terminal root-identification "
            "success alone, an enumerative search jointly synthesizes the query "
            "rule, two predicates and the four-entry action table, freezes the "
            "controller, and transfers to larger hidden chains; the search space "
            "and causal family remain human supplied, so this is not a world "
            "breakthrough"
        ),
        "seed": seed,
        "candidate_gate": gate,
        "observational_equivalence": equivalence,
        "search": search,
        "selected_program": serialize(selected),
        "single_predicate_program": serialize(single),
        "frozen_program_digest": frozen_digest,
        "hidden_dimensions": list(hidden_dimensions),
        "hidden_query_optimality": query_optimality,
        "candidate": candidate.__dict__,
        "single_predicate_control": single_control.__dict__,
        "wrong_query_control": wrong_query.__dict__,
        "specialist_control": specialist.__dict__,
        "random_mapping_control": random_control,
        "single_predicate_gap": single_gap,
        "wrong_query_gap": query_gap,
        "specialist_gap": specialist_gap,
        "random_gap": random_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "rule": report["selected_program"]["rule"],
                "predicates": report["selected_program"]["predicates"],
                "development_accuracy": report["selected_program"][
                    "development_accuracy"
                ],
                "hidden_accuracy": report["candidate"]["accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
