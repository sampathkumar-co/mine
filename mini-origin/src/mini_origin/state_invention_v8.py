from __future__ import annotations

from dataclasses import dataclass
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StateProgram:
    max_slots: int
    spawn_logic: str
    assignment: str
    read_mode: str
    novelty_threshold: float
    surprise_threshold: float
    temperature: float
    weight_rate: float
    prototype_rate: float
    confidence_rate: float
    decay: float
    merge_threshold: float

    def text(self) -> str:
        return (
            f"slots={self.max_slots};spawn={self.spawn_logic};assign={self.assignment};"
            f"read={self.read_mode};nov={self.novelty_threshold:.2f};"
            f"sur={self.surprise_threshold:.2f};temp={self.temperature:.1f};"
            f"wlr={self.weight_rate:.2f};plr={self.prototype_rate:.2f};"
            f"clr={self.confidence_rate:.2f};decay={self.decay:.3f};"
            f"merge={self.merge_threshold:.2f}"
        )

    def complexity(self) -> float:
        structural = self.max_slots - 1
        structural += 1 if self.spawn_logic != "never" else 0
        structural += 1 if self.assignment != "nearest" else 0
        structural += 1 if self.read_mode != "hard" else 0
        return float(structural)


@dataclass(frozen=True)
class ContextScenario:
    seed: int
    contexts: int
    dimension: int
    context_dimension: int
    steps: int
    phase_length: int
    context_noise: float
    target_noise: float
    damage_step: int
    damage_fraction: float
    drift: float

    def label(self) -> str:
        return (
            f"ctx{self.contexts}:d{self.dimension}:phase{self.phase_length}:"
            f"damage{self.damage_fraction:.2f}:drift{self.drift:.3f}:s{self.seed}"
        )


class DynamicStateLearner:
    def __init__(
        self,
        program: StateProgram,
        dimension: int,
        context_dimension: int,
        rng: np.random.Generator,
    ):
        self.program = program
        self.dimension = dimension
        self.context_dimension = context_dimension
        self.rng = rng
        self.weights: list[np.ndarray] = [rng.normal(0.0, 0.01, dimension)]
        self.prototypes: list[np.ndarray] = [np.zeros(context_dimension)]
        self.confidence: list[float] = [0.05]
        self.usage: list[int] = [0]
        self.age: list[int] = [0]
        self.created_slots = 1
        self.damage_events = 0

    def _distances(self, context: np.ndarray) -> np.ndarray:
        return np.asarray(
            [np.linalg.norm(context - prototype) for prototype in self.prototypes],
            dtype=np.float64,
        )

    def _read_weights(self, context: np.ndarray) -> np.ndarray:
        distances = self._distances(context)
        if self.program.read_mode == "hard":
            values = np.zeros(len(self.weights), dtype=np.float64)
            values[int(np.argmin(distances))] = 1.0
            return values

        logits = -self.program.temperature * distances
        if self.program.read_mode == "confidence":
            logits += np.log(np.asarray(self.confidence) + 0.05)
        logits -= np.max(logits)
        values = np.exp(np.clip(logits, -40.0, 40.0))
        return values / max(float(np.sum(values)), 1e-12)

    def predict(self, feature: np.ndarray, context: np.ndarray) -> tuple[float, np.ndarray]:
        read_weights = self._read_weights(context)
        slot_predictions = np.asarray(
            [float(weight @ feature) for weight in self.weights],
            dtype=np.float64,
        )
        prediction = float(read_weights @ slot_predictions)
        return prediction, read_weights

    def _should_spawn(self, novelty: float, surprise: float) -> bool:
        novelty_high = novelty > self.program.novelty_threshold
        surprise_high = surprise > self.program.surprise_threshold
        logic = self.program.spawn_logic
        if logic == "never":
            return False
        if logic == "novelty":
            return novelty_high
        if logic == "surprise":
            return surprise_high
        if logic == "and":
            return novelty_high and surprise_high
        if logic == "or":
            return novelty_high or surprise_high
        if logic == "product":
            return novelty * surprise > (
                self.program.novelty_threshold * self.program.surprise_threshold
            )
        raise ValueError(logic)

    def _merge_or_replace(self, context: np.ndarray) -> int:
        distances = self._distances(context)
        nearest = int(np.argmin(distances))
        if distances[nearest] < self.program.merge_threshold:
            return nearest
        if len(self.weights) < self.program.max_slots:
            self.weights.append(np.zeros(self.dimension, dtype=np.float64))
            self.prototypes.append(context.copy())
            self.confidence.append(0.01)
            self.usage.append(0)
            self.age.append(0)
            self.created_slots += 1
            return len(self.weights) - 1

        if self.program.assignment == "least_confident":
            return int(np.argmin(self.confidence))
        if self.program.assignment == "least_used":
            return int(np.argmin(self.usage))
        if self.program.assignment == "oldest":
            return int(np.argmax(self.age))
        return nearest

    def learn(
        self,
        feature: np.ndarray,
        context: np.ndarray,
        target: float,
    ) -> tuple[float, float, int]:
        prediction, read_weights = self.predict(feature, context)
        error = float(target - prediction)
        surprise = min(2.0, abs(error))
        novelty = float(np.min(self._distances(context)))

        if self._should_spawn(novelty, surprise):
            selected = self._merge_or_replace(context)
            read_weights = np.zeros(len(self.weights), dtype=np.float64)
            read_weights[selected] = 1.0
        else:
            selected = int(np.argmax(read_weights))

        for index in range(len(self.age)):
            self.age[index] += 1
        self.age[selected] = 0
        self.usage[selected] += 1

        local_prediction = float(self.weights[selected] @ feature)
        local_error = float(target - local_prediction)
        self.weights[selected] = np.clip(
            (1.0 - self.program.decay) * self.weights[selected]
            + self.program.weight_rate * local_error * feature,
            -4.0,
            4.0,
        )
        self.prototypes[selected] = (
            (1.0 - self.program.prototype_rate) * self.prototypes[selected]
            + self.program.prototype_rate * context
        )
        quality = math.exp(-3.0 * local_error * local_error)
        self.confidence[selected] = float(
            (1.0 - self.program.confidence_rate) * self.confidence[selected]
            + self.program.confidence_rate * quality
        )
        return prediction, error, selected

    def damage(self, fraction: float) -> int:
        if len(self.weights) <= 1:
            return 0
        count = min(
            len(self.weights) - 1,
            max(1, int(round(len(self.weights) * fraction))),
        )
        # Target the most confident states rather than random redundant copies.
        order = np.argsort(np.asarray(self.confidence))[::-1]
        remove = sorted((int(index) for index in order[:count]), reverse=True)
        for index in remove:
            del self.weights[index]
            del self.prototypes[index]
            del self.confidence[index]
            del self.usage[index]
            del self.age[index]
        self.damage_events += 1
        return count


def _normalised(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=dimension)
    return value / max(np.linalg.norm(value), 1e-12)


def _scenario_stream(scenario: ContextScenario):
    rng = np.random.default_rng(scenario.seed)
    centres = np.stack(
        [_normalised(rng, scenario.context_dimension) for _ in range(scenario.contexts)]
    )
    mappings = np.stack(
        [_normalised(rng, scenario.dimension) for _ in range(scenario.contexts)]
    )
    current_mappings = mappings.copy()

    # Recurrent schedule returns to old contexts after long gaps.
    schedule = list(range(scenario.contexts))
    if scenario.contexts >= 3:
        schedule += [0, scenario.contexts - 1, 1, 0]
    else:
        schedule += [0, 1, 0, 1]

    for step in range(scenario.steps):
        phase = step // scenario.phase_length
        context_index = schedule[phase % len(schedule)]
        if scenario.drift > 0.0:
            drift_noise = rng.normal(0.0, scenario.drift, scenario.dimension)
            current_mappings[context_index] += drift_noise
            current_mappings[context_index] /= max(
                np.linalg.norm(current_mappings[context_index]),
                1e-12,
            )
        feature = _normalised(rng, scenario.dimension)
        context = centres[context_index] + rng.normal(
            0.0,
            scenario.context_noise,
            scenario.context_dimension,
        )
        context /= max(np.linalg.norm(context), 1e-12)
        target = float(
            current_mappings[context_index] @ feature
            + rng.normal(0.0, scenario.target_noise)
        )
        yield step, context_index, feature, context, target


def evaluate_program(
    program: StateProgram,
    scenario: ContextScenario,
) -> dict[str, float]:
    rng = np.random.default_rng(scenario.seed + 99_999)
    learner = DynamicStateLearner(
        program,
        scenario.dimension,
        scenario.context_dimension,
        rng,
    )
    errors: list[float] = []
    baseline_errors: list[float] = []
    post_switch_errors: list[float] = []
    post_damage_errors: list[float] = []
    previous_context = None

    for step, context_index, feature, context, target in _scenario_stream(scenario):
        if step == scenario.damage_step:
            learner.damage(scenario.damage_fraction)
        prediction, error, _ = learner.learn(feature, context, target)
        squared = (prediction - target) ** 2
        errors.append(squared)
        baseline_errors.append(target * target)
        if previous_context is not None and context_index != previous_context:
            post_switch_errors.append(squared)
        if step >= scenario.damage_step:
            post_damage_errors.append(squared)
        previous_context = context_index

    burn = min(len(errors) // 4, max(20, scenario.phase_length))
    mse = float(np.mean(errors[burn:]))
    baseline = float(np.mean(baseline_errors[burn:]))
    skill = float(np.clip(1.0 - mse / max(baseline, 1e-12), 0.0, 1.0))
    switch_mse = float(np.mean(post_switch_errors)) if post_switch_errors else mse
    switch_skill = float(
        np.clip(1.0 - switch_mse / max(baseline, 1e-12), 0.0, 1.0)
    )
    damage_mse = float(np.mean(post_damage_errors)) if post_damage_errors else mse
    damage_skill = float(
        np.clip(1.0 - damage_mse / max(baseline, 1e-12), 0.0, 1.0)
    )
    score = float(0.55 * skill + 0.25 * switch_skill + 0.20 * damage_skill)
    return {
        "score": score,
        "skill": skill,
        "switch_skill": switch_skill,
        "damage_skill": damage_skill,
        "created_slots": float(learner.created_slots),
        "remaining_slots": float(len(learner.weights)),
    }


def robust_score(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    if ordered.size == 0:
        return 0.0
    tail = ordered[: max(1, int(math.ceil(ordered.size * 0.4)))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


def one_state_programs() -> list[StateProgram]:
    values = []
    for read_mode, weight_rate, decay in itertools.product(
        ("hard", "soft"),
        (0.04, 0.08, 0.16, 0.28),
        (0.0, 0.002, 0.01),
    ):
        values.append(
            StateProgram(
                max_slots=1,
                spawn_logic="never",
                assignment="nearest",
                read_mode=read_mode,
                novelty_threshold=0.5,
                surprise_threshold=0.5,
                temperature=6.0,
                weight_rate=weight_rate,
                prototype_rate=0.10,
                confidence_rate=0.10,
                decay=decay,
                merge_threshold=0.25,
            )
        )
    return values


def dynamic_programs(rng: np.random.Generator, count: int = 420) -> list[StateProgram]:
    values: list[StateProgram] = []
    for _ in range(count):
        values.append(
            StateProgram(
                max_slots=int(rng.integers(2, 7)),
                spawn_logic=str(rng.choice(("novelty", "surprise", "and", "or", "product"))),
                assignment=str(rng.choice(("nearest", "least_confident", "least_used", "oldest"))),
                read_mode=str(rng.choice(("hard", "soft", "confidence"))),
                novelty_threshold=float(rng.uniform(0.12, 0.95)),
                surprise_threshold=float(rng.uniform(0.08, 0.90)),
                temperature=float(rng.uniform(1.5, 16.0)),
                weight_rate=float(rng.uniform(0.025, 0.34)),
                prototype_rate=float(rng.uniform(0.025, 0.35)),
                confidence_rate=float(rng.uniform(0.025, 0.35)),
                decay=float(rng.uniform(0.0, 0.025)),
                merge_threshold=float(rng.uniform(0.05, 0.55)),
            )
        )
    return values


def hand_dynamic_program() -> StateProgram:
    return StateProgram(
        max_slots=6,
        spawn_logic="and",
        assignment="least_confident",
        read_mode="confidence",
        novelty_threshold=0.42,
        surprise_threshold=0.24,
        temperature=9.0,
        weight_rate=0.14,
        prototype_rate=0.12,
        confidence_rate=0.10,
        decay=0.001,
        merge_threshold=0.24,
    )


def no_spawn_control(program: StateProgram) -> StateProgram:
    return StateProgram(
        max_slots=1,
        spawn_logic="never",
        assignment=program.assignment,
        read_mode=program.read_mode,
        novelty_threshold=program.novelty_threshold,
        surprise_threshold=program.surprise_threshold,
        temperature=program.temperature,
        weight_rate=program.weight_rate,
        prototype_rate=program.prototype_rate,
        confidence_rate=program.confidence_rate,
        decay=program.decay,
        merge_threshold=program.merge_threshold,
    )


def training_scenarios(seed: int) -> list[ContextScenario]:
    return [
        ContextScenario(seed + 1, 2, 5, 3, 520, 65, 0.05, 0.025, 330, 0.50, 0.0),
        ContextScenario(seed + 2, 3, 6, 4, 680, 68, 0.06, 0.030, 420, 0.55, 0.0),
        ContextScenario(seed + 3, 3, 7, 4, 720, 72, 0.07, 0.035, 450, 0.55, 0.003),
    ]


def counterexample_scenarios(seed: int) -> list[ContextScenario]:
    return [
        ContextScenario(seed + 101, 4, 8, 5, 900, 75, 0.08, 0.04, 540, 0.60, 0.0),
        ContextScenario(seed + 102, 4, 9, 5, 960, 60, 0.09, 0.05, 580, 0.65, 0.004),
        ContextScenario(seed + 103, 5, 10, 6, 1050, 70, 0.10, 0.05, 630, 0.65, 0.006),
    ]


def hidden_scenarios(seed: int) -> list[ContextScenario]:
    return [
        ContextScenario(seed + 501, 4, 10, 6, 1150, 58, 0.10, 0.055, 680, 0.65, 0.0),
        ContextScenario(seed + 502, 5, 12, 7, 1350, 54, 0.12, 0.060, 780, 0.70, 0.005),
        ContextScenario(seed + 503, 6, 14, 8, 1550, 50, 0.14, 0.070, 860, 0.75, 0.008),
    ]


@dataclass(frozen=True)
class RankedStateProgram:
    program: StateProgram
    score: float
    details: dict[str, dict[str, float]]


def rank_programs(
    programs: Iterable[StateProgram],
    scenarios: list[ContextScenario],
    limit: int,
) -> list[RankedStateProgram]:
    ranked = []
    for program in programs:
        details = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        score = robust_score(value["score"] for value in details.values())
        score -= 0.002 * program.complexity()
        ranked.append(RankedStateProgram(program, float(score), details))
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


@dataclass
class StateInventionResult:
    seed: int
    one_state: StateProgram
    invented: StateProgram
    hand_dynamic: StateProgram
    hidden_scores: dict[str, float]
    one_state_scores: dict[str, float]
    no_spawn_scores: dict[str, float]
    hand_scores: dict[str, float]
    history: list[dict[str, object]]

    @property
    def strict_invented(self) -> float:
        return float(min(self.hidden_scores.values()))

    @property
    def strict_one_state(self) -> float:
        return float(min(self.one_state_scores.values()))

    @property
    def strict_no_spawn(self) -> float:
        return float(min(self.no_spawn_scores.values()))

    @property
    def strict_hand(self) -> float:
        return float(min(self.hand_scores.values()))

    @property
    def external_candidate(self) -> bool:
        return (
            self.strict_invented >= 0.72
            and self.strict_invented >= self.strict_one_state + 0.35
            and self.strict_invented >= self.strict_no_spawn + 0.30
            and self.strict_invented >= self.strict_hand + 0.06
            and self.invented.max_slots >= 3
            and self.invented.spawn_logic != "never"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "counterexample_driven_state_invention_candidate"
                if self.external_candidate
                else "not_yet"
            ),
            "claim_scope": (
                "a stateless learner is diagnosed through counterexamples, its language is "
                "expanded with conditional state creation, and an executable dynamic-state "
                "program transfers to more contexts, drift and targeted state deletion; external "
                "acceptance still requires independent reproduction and peer review"
            ),
            "seed": self.seed,
            "one_state_program": self.one_state.text(),
            "invented_program": self.invented.text(),
            "hand_dynamic_program": self.hand_dynamic.text(),
            "strict_invented_score": self.strict_invented,
            "strict_one_state_score": self.strict_one_state,
            "strict_no_spawn_score": self.strict_no_spawn,
            "strict_hand_score": self.strict_hand,
            "hidden_scores": self.hidden_scores,
            "one_state_scores": self.one_state_scores,
            "no_spawn_scores": self.no_spawn_scores,
            "hand_scores": self.hand_scores,
            "history": self.history,
        }


def run_state_invention(seed: int = 81) -> StateInventionResult:
    training = training_scenarios(seed * 10_000)
    one_ranked = rank_programs(one_state_programs(), training, limit=1)[0]
    history: list[dict[str, object]] = [
        {
            "stage": "stateless_search",
            "program": one_ranked.program.text(),
            "score": one_ranked.score,
            "decision": "expand_state_language" if one_ranked.score < 0.70 else "retain",
        }
    ]

    rng = np.random.default_rng(seed)
    dynamic_candidates = dynamic_programs(rng, count=420)
    dynamic_ranked = rank_programs(dynamic_candidates, training, limit=72)
    curriculum = list(training)
    counterexamples = counterexample_scenarios(seed * 10_000)
    best = dynamic_ranked[0]
    shortlist = [value.program for value in dynamic_ranked]
    for iteration, counterexample in enumerate(counterexamples):
        outcome = evaluate_program(best.program, counterexample)
        history.append(
            {
                "stage": "counterexample",
                "iteration": iteration,
                "scenario": counterexample.label(),
                "pre_score": outcome["score"],
                "program": best.program.text(),
            }
        )
        curriculum.append(counterexample)
        best = rank_programs(shortlist, curriculum, limit=1)[0]

    hidden = hidden_scenarios(seed * 10_000)
    hand = hand_dynamic_program()
    no_spawn = no_spawn_control(best.program)

    def scores(program: StateProgram) -> dict[str, float]:
        return {
            scenario.label(): evaluate_program(program, scenario)["score"]
            for scenario in hidden
        }

    return StateInventionResult(
        seed=seed,
        one_state=one_ranked.program,
        invented=best.program,
        hand_dynamic=hand,
        hidden_scores=scores(best.program),
        one_state_scores=scores(one_ranked.program),
        no_spawn_scores=scores(no_spawn),
        hand_scores=scores(hand),
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_state_invention(args.seed)
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
                "strict_no_spawn_score": payload["strict_no_spawn_score"],
                "strict_hand_score": payload["strict_hand_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
