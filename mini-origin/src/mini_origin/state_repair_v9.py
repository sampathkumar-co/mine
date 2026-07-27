from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import state_invention_v8 as v8


@dataclass(frozen=True)
class RepairProgram:
    base: v8.StateProgram
    spawn_init: str
    reset_replacement: bool
    reconstruction_mix: float

    def text(self) -> str:
        return (
            self.base.text()
            + f";init={self.spawn_init};reset={int(self.reset_replacement)};"
            + f"recon={self.reconstruction_mix:.2f}"
        )

    def complexity(self) -> float:
        extra = 1.0 if self.spawn_init != "zero" else 0.0
        extra += 1.0 if self.reset_replacement else 0.0
        return self.base.complexity() + extra


class RepairingStateLearner(v8.DynamicStateLearner):
    def __init__(
        self,
        program: RepairProgram,
        dimension: int,
        context_dimension: int,
        rng: np.random.Generator,
    ):
        self.repair_program = program
        super().__init__(program.base, dimension, context_dimension, rng)

    def _initial_weight(self, context: np.ndarray) -> np.ndarray:
        mode = self.repair_program.spawn_init
        if mode == "zero" or not self.weights:
            return np.zeros(self.dimension, dtype=np.float64)
        if mode == "mean":
            return np.mean(np.stack(self.weights), axis=0)
        distances = self._distances(context)
        nearest = int(np.argmin(distances))
        if mode == "nearest":
            return self.weights[nearest].copy()
        if mode == "confidence_mean":
            confidence = np.asarray(self.confidence, dtype=np.float64) + 0.02
            confidence /= np.sum(confidence)
            return np.sum(
                confidence[:, None] * np.stack(self.weights),
                axis=0,
            )
        if mode == "context_mix":
            logits = -self.program.temperature * distances
            logits -= np.max(logits)
            coefficients = np.exp(np.clip(logits, -40.0, 40.0))
            coefficients /= max(float(np.sum(coefficients)), 1e-12)
            mixture = np.sum(
                coefficients[:, None] * np.stack(self.weights),
                axis=0,
            )
            nearest_weight = self.weights[nearest]
            mix = self.repair_program.reconstruction_mix
            return (1.0 - mix) * nearest_weight + mix * mixture
        raise ValueError(mode)

    def _merge_or_replace(self, context: np.ndarray) -> int:
        distances = self._distances(context)
        nearest = int(np.argmin(distances))
        if distances[nearest] < self.program.merge_threshold:
            return nearest

        initial = self._initial_weight(context)
        if len(self.weights) < self.program.max_slots:
            self.weights.append(initial)
            self.prototypes.append(context.copy())
            self.confidence.append(0.01)
            self.usage.append(0)
            self.age.append(0)
            self.created_slots += 1
            return len(self.weights) - 1

        if self.program.assignment == "least_confident":
            selected = int(np.argmin(self.confidence))
        elif self.program.assignment == "least_used":
            selected = int(np.argmin(self.usage))
        elif self.program.assignment == "oldest":
            selected = int(np.argmax(self.age))
        else:
            selected = nearest

        if self.repair_program.reset_replacement:
            self.weights[selected] = initial
            self.prototypes[selected] = context.copy()
            self.confidence[selected] = 0.01
            self.usage[selected] = 0
            self.age[selected] = 0
        return selected


def evaluate_program(
    program: RepairProgram,
    scenario: v8.ContextScenario,
) -> dict[str, float]:
    rng = np.random.default_rng(scenario.seed + 99_999)
    learner = RepairingStateLearner(
        program,
        scenario.dimension,
        scenario.context_dimension,
        rng,
    )
    errors: list[float] = []
    baseline_errors: list[float] = []
    switch_errors: list[float] = []
    damage_errors: list[float] = []
    recovery_errors: list[float] = []
    previous_context = None
    damage_recovery_end = scenario.damage_step + 2 * scenario.phase_length

    for step, context_index, feature, context, target in v8._scenario_stream(scenario):
        if step == scenario.damage_step:
            learner.damage(scenario.damage_fraction)
        prediction, _, _ = learner.learn(feature, context, target)
        squared = (prediction - target) ** 2
        errors.append(squared)
        baseline_errors.append(target * target)
        if previous_context is not None and context_index != previous_context:
            switch_errors.append(squared)
        if step >= scenario.damage_step:
            damage_errors.append(squared)
        if scenario.damage_step <= step < damage_recovery_end:
            recovery_errors.append(squared)
        previous_context = context_index

    burn = min(len(errors) // 4, max(20, scenario.phase_length))
    baseline = float(np.mean(baseline_errors[burn:]))

    def skill(values: list[float], fallback: list[float]) -> float:
        selected = values if values else fallback
        mse = float(np.mean(selected))
        return float(np.clip(1.0 - mse / max(baseline, 1e-12), 0.0, 1.0))

    overall = skill(errors[burn:], errors)
    switches = skill(switch_errors, errors[burn:])
    damage = skill(damage_errors, errors[burn:])
    recovery = skill(recovery_errors, damage_errors)
    score = float(
        0.45 * overall
        + 0.20 * switches
        + 0.18 * damage
        + 0.17 * recovery
    )
    return {
        "score": score,
        "skill": overall,
        "switch_skill": switches,
        "damage_skill": damage,
        "recovery_skill": recovery,
        "created_slots": float(learner.created_slots),
        "remaining_slots": float(len(learner.weights)),
    }


def mutate_base(
    program: RepairProgram,
    rng: np.random.Generator,
    scale: float,
) -> RepairProgram:
    base = program.base
    categorical = lambda current, options, probability=0.18: (
        str(rng.choice(options)) if rng.random() < probability else current
    )
    integer = base.max_slots
    if rng.random() < 0.22:
        integer = int(np.clip(integer + int(rng.choice((-1, 1))), 2, 7))

    def scalar(value: float, low: float, high: float, sigma: float) -> float:
        if rng.random() >= 0.55:
            return value
        return float(np.clip(value + rng.normal(0.0, sigma * scale), low, high))

    mutated_base = v8.StateProgram(
        max_slots=integer,
        spawn_logic=categorical(
            base.spawn_logic,
            ("novelty", "surprise", "and", "or", "product"),
        ),
        assignment=categorical(
            base.assignment,
            ("nearest", "least_confident", "least_used", "oldest"),
        ),
        read_mode=categorical(base.read_mode, ("hard", "soft", "confidence")),
        novelty_threshold=scalar(base.novelty_threshold, 0.08, 1.10, 0.12),
        surprise_threshold=scalar(base.surprise_threshold, 0.05, 1.10, 0.12),
        temperature=scalar(base.temperature, 1.0, 20.0, 2.0),
        weight_rate=scalar(base.weight_rate, 0.015, 0.42, 0.04),
        prototype_rate=scalar(base.prototype_rate, 0.015, 0.45, 0.04),
        confidence_rate=scalar(base.confidence_rate, 0.015, 0.45, 0.04),
        decay=scalar(base.decay, 0.0, 0.035, 0.004),
        merge_threshold=scalar(base.merge_threshold, 0.03, 0.70, 0.07),
    )
    init = categorical(
        program.spawn_init,
        ("zero", "nearest", "mean", "confidence_mean", "context_mix"),
        probability=0.24,
    )
    reset = program.reset_replacement
    if rng.random() < 0.15:
        reset = not reset
    return RepairProgram(
        base=mutated_base,
        spawn_init=init,
        reset_replacement=reset,
        reconstruction_mix=scalar(program.reconstruction_mix, 0.0, 1.0, 0.14),
    )


def random_program(rng: np.random.Generator) -> RepairProgram:
    base = v8.dynamic_programs(rng, count=1)[0]
    return RepairProgram(
        base=base,
        spawn_init=str(
            rng.choice(("zero", "nearest", "mean", "confidence_mean", "context_mix"))
        ),
        reset_replacement=bool(rng.integers(0, 2)),
        reconstruction_mix=float(rng.uniform(0.0, 1.0)),
    )


def one_state_control(program: RepairProgram) -> RepairProgram:
    return RepairProgram(
        base=v8.no_spawn_control(program.base),
        spawn_init="zero",
        reset_replacement=False,
        reconstruction_mix=0.0,
    )


def hand_repair_program() -> RepairProgram:
    return RepairProgram(
        base=v8.hand_dynamic_program(),
        spawn_init="context_mix",
        reset_replacement=True,
        reconstruction_mix=0.55,
    )


def robust_score(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    tail = ordered[: max(1, int(math.ceil(len(ordered) * 0.4)))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


@dataclass(frozen=True)
class RankedRepair:
    program: RepairProgram
    score: float
    details: dict[str, dict[str, float]]


def rank_programs(
    programs: Iterable[RepairProgram],
    scenarios: list[v8.ContextScenario],
    limit: int,
) -> list[RankedRepair]:
    unique: dict[str, RepairProgram] = {program.text(): program for program in programs}
    ranked: list[RankedRepair] = []
    for program in unique.values():
        details = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        score = robust_score(value["score"] for value in details.values())
        score -= 0.0015 * program.complexity()
        ranked.append(RankedRepair(program, float(score), details))
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


@dataclass
class RepairResult:
    seed: int
    invented: RepairProgram
    one_state: RepairProgram
    no_repair: RepairProgram
    hand: RepairProgram
    hidden_scores: dict[str, float]
    one_state_scores: dict[str, float]
    no_repair_scores: dict[str, float]
    hand_scores: dict[str, float]
    history: list[dict[str, object]]

    @property
    def strict_invented(self) -> float:
        return float(min(self.hidden_scores.values()))

    @property
    def strict_one(self) -> float:
        return float(min(self.one_state_scores.values()))

    @property
    def strict_no_repair(self) -> float:
        return float(min(self.no_repair_scores.values()))

    @property
    def strict_hand(self) -> float:
        return float(min(self.hand_scores.values()))

    @property
    def external_candidate(self) -> bool:
        return (
            self.strict_invented >= 0.72
            and self.strict_invented >= self.strict_one + 0.35
            and self.strict_invented >= self.strict_no_repair + 0.20
            and self.strict_invented >= self.strict_hand + 0.06
            and self.invented.base.max_slots >= 3
            and self.invented.spawn_init != "zero"
            and self.invented.reset_replacement
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "adversarial_state_repair_candidate"
                if self.external_candidate
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided structural evolution synthesizes state allocation and "
                "reconstruction semantics that recover recurring contextual mappings after targeted "
                "state deletion; external acceptance still requires independent reproduction and review"
            ),
            "seed": self.seed,
            "invented_program": self.invented.text(),
            "one_state_program": self.one_state.text(),
            "no_repair_program": self.no_repair.text(),
            "hand_program": self.hand.text(),
            "strict_invented_score": self.strict_invented,
            "strict_one_state_score": self.strict_one,
            "strict_no_repair_score": self.strict_no_repair,
            "strict_hand_score": self.strict_hand,
            "hidden_scores": self.hidden_scores,
            "one_state_scores": self.one_state_scores,
            "no_repair_scores": self.no_repair_scores,
            "hand_scores": self.hand_scores,
            "history": self.history,
        }


def run_state_repair(seed: int = 91) -> RepairResult:
    rng = np.random.default_rng(seed)
    training = v8.training_scenarios(seed * 10_000)
    adversaries = v8.counterexample_scenarios(seed * 10_000)
    population = [random_program(rng) for _ in range(180)]
    history: list[dict[str, object]] = []
    curriculum = list(training)

    for generation in range(9):
        ranked = rank_programs(population, curriculum, limit=24)
        best = ranked[0]
        history.append(
            {
                "generation": generation,
                "curriculum_size": len(curriculum),
                "score": best.score,
                "program": best.program.text(),
            }
        )
        if generation < len(adversaries):
            # Add the adversary on which the current champion performs worst.
            remaining = adversaries[generation:]
            scored = [
                (evaluate_program(best.program, scenario)["score"], scenario)
                for scenario in remaining
            ]
            scored.sort(key=lambda value: value[0])
            curriculum.append(scored[0][1])

        elites = [value.program for value in ranked[:12]]
        next_population = list(elites)
        scale = 1.0 - 0.75 * generation / 8.0
        while len(next_population) < 96:
            parent = elites[int(rng.integers(0, len(elites)))]
            next_population.append(mutate_base(parent, rng, scale))
        if generation < 3:
            next_population.extend(random_program(rng) for _ in range(24))
        population = next_population

    invented = rank_programs(population, curriculum, limit=1)[0].program
    one_state = one_state_control(invented)
    no_repair = RepairProgram(
        base=invented.base,
        spawn_init="zero",
        reset_replacement=False,
        reconstruction_mix=0.0,
    )
    hand = hand_repair_program()
    hidden = v8.hidden_scenarios(seed * 10_000)

    def scores(program: RepairProgram) -> dict[str, float]:
        return {
            scenario.label(): evaluate_program(program, scenario)["score"]
            for scenario in hidden
        }

    return RepairResult(
        seed=seed,
        invented=invented,
        one_state=one_state,
        no_repair=no_repair,
        hand=hand,
        hidden_scores=scores(invented),
        one_state_scores=scores(one_state),
        no_repair_scores=scores(no_repair),
        hand_scores=scores(hand),
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_state_repair(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "invented_program": payload["invented_program"],
                "strict_invented_score": payload["strict_invented_score"],
                "strict_one_state_score": payload["strict_one_state_score"],
                "strict_no_repair_score": payload["strict_no_repair_score"],
                "strict_hand_score": payload["strict_hand_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
