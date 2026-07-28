from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import random

import numpy as np

from .intervention_genesis_v22 import (
    information_lower_bound,
    intervention_response,
    observational_equivalence_certificate,
    optimal_worst_case_queries,
)
from .outcome_alphabet_v23 import ACCEPT, ACTION_NAMES, LEFT, RIGHT
from .predicate_genesis_v25 import Predicate
from .terminal_reward_v26 import (
    TerminalProgram,
    World,
    _all_exact_two_predicate_programs,
    _best_single_predicate_program,
    digest,
    evaluate_noisy,
    position_rules,
    serialize,
    specialist_program,
    terminal_predicate_grammar,
    wrong_query_control,
)


@dataclass(frozen=True)
class Stratum:
    name: str
    dimensions: tuple[int, ...]
    rho_low: float
    rho_high: float
    replicates: int
    trials: int


@dataclass(frozen=True)
class StrataEvaluation:
    minimum_accuracy: float
    mean_accuracy: float
    maximum_invalid_transition_rate: float
    rows: tuple[dict[str, object], ...]


def robust_exact_worlds() -> tuple[World, ...]:
    return tuple(
        World(dimension, root, rho)
        for dimension in range(2, 10)
        for rho in (0.24, 0.40, 0.65, 0.90)
        for root in range(dimension)
    )


def support_observations(
    seed: int,
    samples: int = 2_000,
) -> list[tuple[float | None, float | None]]:
    rng = np.random.default_rng(seed)
    rows = []
    replicate_choices = (256, 384, 512)
    for index in range(samples):
        dimension = int(rng.integers(3, 10))
        root = int(rng.integers(0, dimension))
        query = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.24, 0.92))
        replicates = int(rng.choice(replicate_choices))
        rows.append(
            intervention_response(
                seed + index * 65_537,
                dimension,
                root,
                rho,
                query,
                replicates,
            )
        )
    return rows


def development_strata() -> tuple[Stratum, ...]:
    dimensions = (3, 4, 5, 6, 7, 8, 9, 11)
    return (
        Stratum("low-coupling", dimensions, 0.24, 0.34, 512, 900),
        Stratum("transition", dimensions, 0.26, 0.42, 384, 900),
        Stratum("medium", dimensions, 0.34, 0.60, 384, 900),
        Stratum("high", dimensions, 0.60, 0.96, 256, 900),
    )


def hidden_strata() -> tuple[Stratum, ...]:
    dimensions = (17, 31, 63, 127, 255)
    return (
        Stratum("hidden-low", dimensions, 0.24, 0.34, 512, 2_400),
        Stratum("hidden-transition", dimensions, 0.26, 0.42, 384, 2_400),
        Stratum("hidden-medium", dimensions, 0.34, 0.60, 384, 2_400),
        Stratum("hidden-high", dimensions, 0.60, 0.96, 256, 2_400),
    )


def evaluate_strata(
    seed: int,
    program: TerminalProgram,
    strata: tuple[Stratum, ...],
) -> StrataEvaluation:
    rows = []
    for index, stratum in enumerate(strata):
        evaluation = evaluate_noisy(
            seed + index * 1_000_003,
            program,
            stratum.dimensions,
            stratum.trials,
            stratum.replicates,
            stratum.rho_low,
            stratum.rho_high,
        )
        rows.append(
            {
                "name": stratum.name,
                "rho_low": stratum.rho_low,
                "rho_high": stratum.rho_high,
                "replicates": stratum.replicates,
                "accuracy": evaluation.accuracy,
                "invalid_transition_rate": evaluation.invalid_transition_rate,
                "mean_queries": evaluation.mean_queries,
            }
        )
    return StrataEvaluation(
        minimum_accuracy=min(float(row["accuracy"]) for row in rows),
        mean_accuracy=float(np.mean([float(row["accuracy"]) for row in rows])),
        maximum_invalid_transition_rate=max(
            float(row["invalid_transition_rate"]) for row in rows
        ),
        rows=tuple(rows),
    )


def search_support_robust_programs(
    seed: int,
) -> tuple[TerminalProgram, TerminalProgram, StrataEvaluation, dict[str, object]]:
    observations = support_observations(seed)
    grammar = terminal_predicate_grammar(observations)
    worlds = robust_exact_worlds()
    single = _best_single_predicate_program(grammar, worlds)
    exact = _all_exact_two_predicate_programs(grammar, worlds)
    if not exact:
        raise RuntimeError("support-robust search found no exact controller")
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
    strata = development_strata()
    evaluated = []
    for program in finalists:
        score = evaluate_strata(seed + 2_000_003, program, strata)
        evaluated.append((program, score))

    best_minimum = max(score.minimum_accuracy for _, score in evaluated)
    near_minimum = [
        (program, score)
        for program, score in evaluated
        if score.minimum_accuracy >= best_minimum - 0.001
    ]
    best_mean = max(score.mean_accuracy for _, score in near_minimum)
    eligible = [
        (program, score)
        for program, score in near_minimum
        if score.mean_accuracy >= best_mean - 0.001
    ]
    selected, selected_score = min(
        eligible,
        key=lambda item: (
            item[0].complexity,
            item[0].rule.complexity,
            item[0].rule.name,
            tuple(predicate.text() for predicate in item[0].predicates),
            item[0].mapping,
        ),
    )
    selected = TerminalProgram(
        selected.rule,
        selected.predicates,
        selected.mapping,
        1.0,
        selected_score.mean_accuracy,
    )
    evidence = {
        "supervision": "terminal_root_success_only",
        "step_action_labels_used": False,
        "selection_objective": "maximise_worst_support_stratum_then_mean",
        "observation_count": len(observations),
        "grammar_size": len(grammar),
        "training_world_count": len(worlds),
        "exact_program_count": len(exact),
        "finalist_count": len(finalists),
        "selected_rule": selected.rule.name,
        "selected_predicates": [
            predicate.text() for predicate in selected.predicates
        ],
        "selected_mapping": [ACTION_NAMES[action] for action in selected.mapping],
        "development_worst_accuracy": selected_score.minimum_accuracy,
        "development_mean_accuracy": selected_score.mean_accuracy,
        "development_rows": list(selected_score.rows),
        "single_predicate_training_accuracy": single.training_accuracy,
    }
    return selected, single, selected_score, evidence


def robust_random_mapping_control(
    seed: int,
    template: TerminalProgram,
    strata: tuple[Stratum, ...],
) -> dict[str, float]:
    rng = random.Random(seed)
    scores = []
    for index in range(32):
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
        score = evaluate_strata(
            seed + index * 1_000_003,
            program,
            tuple(
                Stratum(
                    stratum.name,
                    stratum.dimensions,
                    stratum.rho_low,
                    stratum.rho_high,
                    stratum.replicates,
                    480,
                )
                for stratum in strata
            ),
        )
        scores.append(score.minimum_accuracy)
    return {
        "trials": len(scores),
        "median_worst_accuracy": float(np.median(scores)),
        "maximum_worst_accuracy": max(scores),
    }


def run(seed: int = 1201) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    selected, single, development, search = search_support_robust_programs(
        seed * 10_000 + 89
    )
    frozen_digest = digest(selected)

    # Hidden support strata are instantiated only after the controller and
    # max-min selection evidence have been frozen.
    strata = hidden_strata()
    candidate = evaluate_strata(
        seed * 10_000 + 12_000_001,
        selected,
        strata,
    )
    single_control = evaluate_strata(
        seed * 10_000 + 12_000_003,
        single,
        strata,
    )
    query_control = evaluate_strata(
        seed * 10_000 + 12_000_005,
        wrong_query_control(selected),
        strata,
    )
    specialist = evaluate_strata(
        seed * 10_000 + 12_000_007,
        specialist_program(),
        strata,
    )
    random_control = robust_random_mapping_control(
        seed * 10_000 + 12_000_009,
        selected,
        strata,
    )

    single_gap = candidate.minimum_accuracy - single_control.minimum_accuracy
    query_gap = candidate.minimum_accuracy - query_control.minimum_accuracy
    specialist_gap = candidate.minimum_accuracy - specialist.minimum_accuracy
    random_gap = (
        candidate.minimum_accuracy - random_control["median_worst_accuracy"]
    )
    dimensions = hidden_strata()[0].dimensions
    query_optimality = all(
        optimal_worst_case_queries(dimension)
        == information_lower_bound(dimension)
        for dimension in dimensions
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
        and search["selection_objective"]
        == "maximise_worst_support_stratum_then_mean"
        and selected.training_accuracy == 1.0
        and selected.rule.name == "lower_midpoint"
        and predicate_family
        and development.minimum_accuracy >= 0.985
        and candidate.minimum_accuracy >= 0.985
        and candidate.maximum_invalid_transition_rate <= 0.01
        and single_gap >= 0.20
        and query_gap >= 0.20
        and specialist_gap >= -0.01
        and random_gap >= 0.45
        and query_optimality
    )
    return {
        "status": (
            "support_robust_terminal_genesis_candidate" if gate else "not_yet"
        ),
        "claim_scope": (
            "terminal-reward-only controller synthesis is selected by worst-case "
            "performance across preregistered coupling and noise strata, then "
            "transfers to hidden chains up to size 255; the grammar and causal "
            "family remain human supplied, so this is not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": gate,
        "observational_equivalence": equivalence,
        "search": search,
        "selected_program": serialize(selected),
        "single_predicate_program": serialize(single),
        "frozen_program_digest": frozen_digest,
        "hidden_dimensions": list(dimensions),
        "hidden_query_optimality": query_optimality,
        "candidate": candidate.__dict__,
        "single_predicate_control": single_control.__dict__,
        "wrong_query_control": query_control.__dict__,
        "specialist_control": specialist.__dict__,
        "random_mapping_control": random_control,
        "single_predicate_gap": single_gap,
        "wrong_query_gap": query_gap,
        "specialist_gap": specialist_gap,
        "random_gap": random_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1201)
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
                "development_worst": report["search"][
                    "development_worst_accuracy"
                ],
                "hidden_worst": report["candidate"]["minimum_accuracy"],
                "specialist_gap": report["specialist_gap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
