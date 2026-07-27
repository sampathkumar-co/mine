from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import coded_memory_v10 as v10


WRITE_LIMIT = 0.40


def specialist_expander_program() -> v10.CodeProgram:
    return v10.CodeProgram(
        density=0.34,
        systematic=False,
        coefficient_mode="rademacher",
        balanced=True,
        ridge=1e-6,
        learning_rate=0.24,
        seed_salt=7_777,
    )


def constrained_programs(
    rng: np.random.Generator,
    count: int = 620,
) -> list[v10.CodeProgram]:
    values: list[v10.CodeProgram] = []
    for _ in range(count):
        values.append(
            v10.CodeProgram(
                density=float(rng.uniform(0.16, 0.39)),
                systematic=bool(rng.random() < 0.18),
                coefficient_mode=str(rng.choice(("rademacher", "gaussian"))),
                balanced=bool(rng.integers(0, 2)),
                ridge=float(10.0 ** rng.uniform(-7.5, -3.5)),
                learning_rate=float(rng.uniform(0.18, 0.38)),
                seed_salt=int(rng.integers(1, 2_000_000)),
            )
        )
    return values


@dataclass(frozen=True)
class FeasibleRankedCode:
    program: v10.CodeProgram
    score: float
    evaluations: dict[str, v10.CodeEvaluation]
    expander_gap: float


def _evaluate_expander(scenario: v10.CodeScenario) -> v10.CodeEvaluation:
    return v10.evaluate_program(specialist_expander_program(), scenario)


def rank_feasible(
    programs: Iterable[v10.CodeProgram],
    scenarios: list[v10.CodeScenario],
    limit: int,
) -> list[FeasibleRankedCode]:
    expander = {
        scenario.label(): _evaluate_expander(scenario)
        for scenario in scenarios
    }
    ranked: list[FeasibleRankedCode] = []
    for program in programs:
        evaluations = {
            scenario.label(): v10.evaluate_program(program, scenario)
            for scenario in scenarios
        }
        feasible = all(
            evaluation.write_fraction <= WRITE_LIMIT + 1e-12
            and evaluation.surviving_rank == scenario.contexts
            for evaluation, scenario in zip(evaluations.values(), scenarios)
        )
        if not feasible:
            continue
        post = [value.post_damage for value in evaluations.values()]
        retention = [value.retention for value in evaluations.values()]
        writes = [value.write_fraction for value in evaluations.values()]
        gaps = [
            evaluations[label].post_damage - expander[label].post_damage
            for label in evaluations
        ]
        score = (
            0.62 * v10.robust_score(post)
            + 0.18 * v10.robust_score(retention)
            + 0.12 * (1.0 - max(writes))
            + 0.08 * min(gaps)
        )
        ranked.append(
            FeasibleRankedCode(
                program=program,
                score=float(score),
                evaluations=evaluations,
                expander_gap=float(min(gaps)),
            )
        )
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


def hidden_scenarios(seed: int) -> list[v10.CodeScenario]:
    return [
        v10.CodeScenario(seed + 501, 8, 12, 2.70, 80, 0.050, 0.55),
        v10.CodeScenario(seed + 502, 10, 14, 2.80, 86, 0.060, 0.58),
        v10.CodeScenario(seed + 503, 12, 16, 2.90, 92, 0.070, 0.60),
        v10.CodeScenario(seed + 504, 14, 18, 3.00, 98, 0.075, 0.62),
    ]


@dataclass
class ConstrainedCodeResult:
    seed: int
    program: v10.CodeProgram
    hidden: dict[str, v10.CodeEvaluation]
    dense: dict[str, v10.CodeEvaluation]
    replication: dict[str, v10.CodeEvaluation]
    expander: dict[str, v10.CodeEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_dense(self) -> float:
        return float(min(value.post_damage for value in self.dense.values()))

    @property
    def strict_replication(self) -> float:
        return float(min(value.post_damage for value in self.replication.values()))

    @property
    def strict_expander(self) -> float:
        return float(min(value.post_damage for value in self.expander.values()))

    @property
    def max_write(self) -> float:
        return float(max(value.write_fraction for value in self.hidden.values()))

    @property
    def min_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def candidate(self) -> bool:
        return (
            self.strict_post >= 0.90
            and self.min_retention >= 0.95
            and self.max_write <= WRITE_LIMIT
            and self.strict_post >= self.strict_dense - 0.04
            and self.strict_post >= self.strict_replication + 0.15
            and self.strict_post >= self.strict_expander - 0.015
        )

    def to_dict(self) -> dict[str, object]:
        def serialize(values: dict[str, v10.CodeEvaluation]) -> dict[str, dict[str, float | int]]:
            return {
                key: {
                    "score": value.score,
                    "pre_damage": value.pre_damage,
                    "post_damage": value.post_damage,
                    "retention": value.retention,
                    "write_fraction": value.write_fraction,
                    "surviving_rank": value.surviving_rank,
                }
                for key, value in values.items()
            }

        return {
            "status": (
                "hard_constrained_sparse_code_candidate"
                if self.candidate
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided synthesis of a rank-preserving online sparse memory code under "
                "a hard write-operation budget, compared with dense, replication and hand sparse-expander "
                "controls; coding is established, so external significance requires specialist review"
            ),
            "seed": self.seed,
            "program": self.program.text(),
            "strict_post_damage": self.strict_post,
            "strict_dense_post_damage": self.strict_dense,
            "strict_replication_post_damage": self.strict_replication,
            "strict_expander_post_damage": self.strict_expander,
            "max_write_fraction": self.max_write,
            "min_retention": self.min_retention,
            "hidden": serialize(self.hidden),
            "dense": serialize(self.dense),
            "replication": serialize(self.replication),
            "expander": serialize(self.expander),
            "history": self.history,
        }


def run_constrained_search(seed: int = 111) -> ConstrainedCodeResult:
    rng = np.random.default_rng(seed)
    curriculum = v10.training_scenarios(seed * 10_000)
    candidates = constrained_programs(rng, count=620)
    ranked = rank_feasible(candidates, curriculum, limit=96)
    if not ranked:
        raise RuntimeError("no feasible sparse codes survived the initial curriculum")
    shortlist = [value.program for value in ranked]
    history: list[dict[str, object]] = []

    for iteration, scenario in enumerate(v10.counterexamples(seed * 10_000)):
        ranked = rank_feasible(shortlist, curriculum, limit=96)
        if not ranked:
            raise RuntimeError("all sparse codes became infeasible")
        champion = ranked[0]
        outcome = v10.evaluate_program(champion.program, scenario)
        expander = _evaluate_expander(scenario)
        history.append(
            {
                "iteration": iteration,
                "program": champion.program.text(),
                "curriculum_score": champion.score,
                "counterexample": scenario.label(),
                "counterexample_post_damage": outcome.post_damage,
                "counterexample_write_fraction": outcome.write_fraction,
                "counterexample_expander_gap": outcome.post_damage - expander.post_damage,
            }
        )
        curriculum.append(scenario)

    final = rank_feasible(shortlist, curriculum, limit=1)
    if not final:
        raise RuntimeError("no final feasible sparse code")
    program = final[0].program
    hidden = hidden_scenarios(seed * 10_000)
    return ConstrainedCodeResult(
        seed=seed,
        program=program,
        hidden={scenario.label(): v10.evaluate_program(program, scenario) for scenario in hidden},
        dense={scenario.label(): v10.evaluate_dense(scenario) for scenario in hidden},
        replication={scenario.label(): v10.evaluate_replication(scenario) for scenario in hidden},
        expander={scenario.label(): _evaluate_expander(scenario) for scenario in hidden},
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_constrained_search(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "program": payload["program"],
                "strict_post_damage": payload["strict_post_damage"],
                "strict_dense_post_damage": payload["strict_dense_post_damage"],
                "strict_replication_post_damage": payload["strict_replication_post_damage"],
                "strict_expander_post_damage": payload["strict_expander_post_damage"],
                "max_write_fraction": payload["max_write_fraction"],
                "min_retention": payload["min_retention"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
