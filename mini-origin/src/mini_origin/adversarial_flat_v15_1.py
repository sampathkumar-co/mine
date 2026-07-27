from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import json
from pathlib import Path

import numpy as np

from . import rotating_sketch_v12 as v12
from . import information_routing_v14 as v14
from . import adversarial_flat_v15 as v15


WRITE_LIMIT = 0.20


def fixed_budget_programs(
    rng: np.random.Generator,
    count: int = 140,
) -> list[v15.FlatProgram]:
    values = [
        replace(program, density=WRITE_LIMIT)
        for program in v15.random_programs(rng, count=count)
    ]
    unique: dict[str, v15.FlatProgram] = {}
    for program in values:
        unique.setdefault(program.text(), program)
    return list(unique.values())


def iid_program() -> v12.SketchProgram:
    return v12.SketchProgram(WRITE_LIMIT, "iid", 1, 1, 1, 2.5e-5, 610_020)


def balanced_program() -> v12.SketchProgram:
    return v12.SketchProgram(WRITE_LIMIT, "balanced", 1, 3, 5, 2.5e-5, 615_020)


def antithetic_program() -> v12.SketchProgram:
    return v12.SketchProgram(WRITE_LIMIT, "antithetic", 1, 1, 1, 2.5e-5, 620_020)


def dense_program() -> v12.SketchProgram:
    return v12.SketchProgram(1.0, "balanced", 1, 1, 1, 2.5e-5, 625_100)


@dataclass
class CorrectedResult:
    seed: int
    best: v15.FlatProgram
    hidden: dict[str, v15.FlatEvaluation]
    dense: dict[str, v12.SketchEvaluation]
    iid: dict[str, v12.SketchEvaluation]
    balanced: dict[str, v12.SketchEvaluation]
    antithetic: dict[str, v12.SketchEvaluation]
    information: dict[str, v14.RoutingEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_dense(self) -> float:
        return float(min(value.post_damage for value in self.dense.values()))

    @property
    def dense_fraction(self) -> float:
        return self.strict_post / max(self.strict_dense, 1e-12)

    @property
    def strict_iid(self) -> float:
        return float(min(value.post_damage for value in self.iid.values()))

    @property
    def strict_specialist(self) -> float:
        return float(
            max(
                min(value.post_damage for value in self.balanced.values()),
                min(value.post_damage for value in self.antithetic.values()),
                min(value.post_damage for value in self.information.values()),
            )
        )

    @property
    def strict_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def candidate(self) -> bool:
        return (
            self.dense_fraction >= 0.97
            and self.strict_post >= self.strict_iid + 0.025
            and self.strict_post >= self.strict_specialist + 0.020
            and self.strict_retention >= 0.90
            and all(
                abs(value.write_fraction - (
                    max(1, int(np.floor(WRITE_LIMIT * scenario.cells))) / scenario.cells
                )) <= 1e-12
                for scenario, value in zip(
                    v15.hidden_scenarios(self.seed * 10_000),
                    self.hidden.values(),
                )
            )
        )

    def to_dict(self) -> dict[str, object]:
        def values(items):
            return {label: value.__dict__ for label, value in items.items()}

        return {
            "status": (
                "corrected_flat_memory_candidate"
                if self.candidate
                else "not_yet"
            ),
            "claim_scope": (
                "fixed-budget online routing minimises counterfactual deletion attack surface; "
                "the gate is relative to dense recovery and requires gaps over equal-write iid, "
                "balanced, antithetic and information-routing controls; external status still "
                "requires coding-theory baselines and outside reproduction"
            ),
            "seed": self.seed,
            "best_program": self.best.text(),
            "strict_post_damage": self.strict_post,
            "strict_dense_post_damage": self.strict_dense,
            "fraction_of_dense": self.dense_fraction,
            "strict_iid_post_damage": self.strict_iid,
            "strict_specialist_post_damage": self.strict_specialist,
            "strict_retention": self.strict_retention,
            "hidden": values(self.hidden),
            "dense": values(self.dense),
            "iid": values(self.iid),
            "balanced": values(self.balanced),
            "antithetic": values(self.antithetic),
            "information": values(self.information),
            "history": self.history,
        }


def run(seed: int = 161) -> CorrectedResult:
    rng = np.random.default_rng(seed)
    candidates = fixed_budget_programs(rng)
    curriculum = v15.training_scenarios(seed * 10_000)
    ranked = v15.rank_programs(candidates, curriculum, limit=36)
    shortlist = [value.program for value in ranked]
    best = ranked[0]
    history: list[dict[str, object]] = []

    for iteration, counterexample in enumerate(v15.counterexamples(seed * 10_000)):
        pre = v15.evaluate_program(best.program, counterexample)
        history.append(
            {
                "iteration": iteration,
                "counterexample": counterexample.label(),
                "pre_score": pre.score,
                "pre_post_damage": pre.post_damage,
                "program": best.program.text(),
            }
        )
        curriculum.append(counterexample)
        best = v15.rank_programs(shortlist, curriculum, limit=1)[0]

    hidden_scenarios = v15.hidden_scenarios(seed * 10_000)
    return CorrectedResult(
        seed=seed,
        best=best.program,
        hidden={
            scenario.label(): v15.evaluate_program(best.program, scenario)
            for scenario in hidden_scenarios
        },
        dense={
            scenario.label(): v12.evaluate_program(dense_program(), scenario)
            for scenario in hidden_scenarios
        },
        iid={
            scenario.label(): v12.evaluate_program(iid_program(), scenario)
            for scenario in hidden_scenarios
        },
        balanced={
            scenario.label(): v12.evaluate_program(balanced_program(), scenario)
            for scenario in hidden_scenarios
        },
        antithetic={
            scenario.label(): v12.evaluate_program(antithetic_program(), scenario)
            for scenario in hidden_scenarios
        },
        information={
            scenario.label(): v14.evaluate_program(v14.hand_information_program(), scenario)
            for scenario in hidden_scenarios
        },
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=161)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strict_post_damage": payload["strict_post_damage"],
                "strict_dense_post_damage": payload["strict_dense_post_damage"],
                "fraction_of_dense": payload["fraction_of_dense"],
                "strict_iid_post_damage": payload["strict_iid_post_damage"],
                "strict_specialist_post_damage": payload[
                    "strict_specialist_post_damage"
                ],
                "strict_retention": payload["strict_retention"],
                "best_program": payload["best_program"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
