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
class Signature:
    values: np.ndarray
    phase_slots: tuple[int, ...]

    def key(self, precision: int = 1) -> tuple[float, ...]:
        rounded = np.round(self.values, precision)
        return tuple(float(value) for value in rounded) + tuple(
            float(value) for value in self.phase_slots
        )


def _unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def behaviour_signature(program: v8.StateProgram, seed: int = 17_017) -> Signature:
    """Interventional signature invariant to internal slot labels."""
    rng = np.random.default_rng(seed)
    dimension = 4
    context_dimension = 3
    learner = v8.DynamicStateLearner(program, dimension, context_dimension, rng)
    contexts = np.stack(
        [
            _unit(np.array([1.0, 0.1, 0.0])),
            _unit(np.array([0.0, 1.0, 0.1])),
            _unit(np.array([0.1, 0.0, 1.0])),
        ]
    )
    mappings = np.stack(
        [
            _unit(np.array([1.0, -0.2, 0.3, 0.1])),
            _unit(np.array([-0.1, 0.8, 0.2, -0.4])),
            _unit(np.array([0.3, 0.1, -0.8, 0.4])),
        ]
    )

    # Each phase isolates a different mechanism:
    # A: ordinary learning; B: novelty without surprise; C: surprise without novelty;
    # D: return after interference; E: new context/new mapping; F: targeted damage.
    phases = [
        (0, mappings[0], 14),
        (1, mappings[0], 8),
        (0, -mappings[0], 9),
        (0, mappings[0], 8),
        (2, mappings[2], 10),
        (1, mappings[0], 8),
    ]
    values: list[float] = []
    phase_slots: list[int] = []
    previous_created = learner.created_slots
    for phase_index, (context_index, mapping, steps) in enumerate(phases):
        if phase_index == 5:
            learner.damage(0.50)
        phase_errors: list[float] = []
        phase_predictions: list[float] = []
        for step in range(steps):
            feature = _unit(
                np.array(
                    [
                        math.sin(0.4 * (step + 1)),
                        math.cos(0.6 * (step + 1)),
                        (-1.0) ** step,
                        (step % 3 - 1.0) / 2.0,
                    ]
                )
            )
            context = _unit(
                contexts[context_index]
                + 0.025 * rng.normal(size=context_dimension)
            )
            target = float(mapping @ feature)
            prediction, error, _ = learner.learn(feature, context, target)
            phase_errors.append(error)
            phase_predictions.append(prediction)
        values.extend(
            [
                float(np.mean(np.abs(phase_errors))),
                float(np.mean(np.square(phase_errors))),
                float(np.mean(phase_predictions)),
                float(np.std(phase_predictions)),
                float(learner.created_slots - previous_created),
                float(len(learner.weights)),
            ]
        )
        previous_created = learner.created_slots
        phase_slots.append(len(learner.weights))
    return Signature(np.asarray(values, dtype=np.float64), tuple(phase_slots))


def signature_distance(left: Signature, right: Signature) -> float:
    scale = np.maximum(0.10, np.maximum(np.abs(left.values), np.abs(right.values)))
    numeric = float(np.sqrt(np.mean(np.square((left.values - right.values) / scale))))
    slot = float(
        np.mean(
            np.abs(np.asarray(left.phase_slots) - np.asarray(right.phase_slots))
        )
    )
    return numeric + 0.18 * slot


def _canonical_programs() -> dict[str, v8.StateProgram]:
    values: dict[str, v8.StateProgram] = {}
    one_state = v8.one_state_programs()
    for index, program in enumerate(one_state):
        values[f"stateless-{index}"] = program
    values["hand-dynamic"] = v8.hand_dynamic_program()

    base = v8.hand_dynamic_program()
    for logic in ("novelty", "surprise", "and", "or", "product"):
        values[f"canonical-{logic}"] = v8.StateProgram(
            max_slots=6,
            spawn_logic=logic,
            assignment="least_confident",
            read_mode="confidence",
            novelty_threshold=base.novelty_threshold,
            surprise_threshold=base.surprise_threshold,
            temperature=base.temperature,
            weight_rate=base.weight_rate,
            prototype_rate=base.prototype_rate,
            confidence_rate=base.confidence_rate,
            decay=base.decay,
            merge_threshold=base.merge_threshold,
        )
    for assignment in ("nearest", "least_confident", "least_used", "oldest"):
        values[f"canonical-assignment-{assignment}"] = v8.StateProgram(
            max_slots=6,
            spawn_logic="and",
            assignment=assignment,
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
    return values


@dataclass(frozen=True)
class QuotientCandidate:
    program: v8.StateProgram
    signature: Signature
    nearest_known: str
    known_distance: float


def quotient_candidates(
    programs: Iterable[v8.StateProgram],
    precision: int = 1,
    minimum_known_distance: float = 0.16,
) -> tuple[list[QuotientCandidate], dict[str, object]]:
    known = _canonical_programs()
    known_signatures = {
        name: behaviour_signature(program, seed=17_017)
        for name, program in known.items()
    }
    classes: dict[tuple[float, ...], QuotientCandidate] = {}
    rejected_known = 0
    for program in programs:
        signature = behaviour_signature(program, seed=17_017)
        distances = {
            name: signature_distance(signature, known_signature)
            for name, known_signature in known_signatures.items()
        }
        nearest = min(distances, key=distances.get)
        distance = float(distances[nearest])
        if distance < minimum_known_distance:
            rejected_known += 1
            continue
        candidate = QuotientCandidate(program, signature, nearest, distance)
        key = signature.key(precision)
        existing = classes.get(key)
        if existing is None or program.complexity() < existing.program.complexity():
            classes[key] = candidate
    metadata = {
        "raw_programs": sum(1 for _ in programs) if not isinstance(programs, list) else len(programs),
        "known_classes": len(known),
        "rejected_as_known": rejected_known,
        "surviving_equivalence_classes": len(classes),
        "minimum_known_distance": minimum_known_distance,
        "signature_precision": precision,
    }
    return list(classes.values()), metadata


def _scenario_scores(
    program: v8.StateProgram,
    scenarios: list[v8.ContextScenario],
) -> dict[str, float]:
    return {
        scenario.label(): v8.evaluate_program(program, scenario)["score"]
        for scenario in scenarios
    }


def _robust(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    tail = ordered[: max(1, int(math.ceil(0.4 * len(ordered))))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


def _intervention_scenarios(seed: int) -> list[v8.ContextScenario]:
    return [
        v8.ContextScenario(seed + 1, 2, 5, 3, 460, 46, 0.04, 0.025, 270, 0.45, 0.000),
        v8.ContextScenario(seed + 2, 3, 6, 4, 580, 42, 0.05, 0.030, 330, 0.50, 0.004),
        v8.ContextScenario(seed + 3, 4, 7, 5, 720, 40, 0.06, 0.035, 400, 0.55, 0.006),
    ]


def _hidden_scenarios(seed: int) -> list[v8.ContextScenario]:
    return [
        v8.ContextScenario(seed + 501, 5, 10, 6, 1200, 40, 0.10, 0.055, 680, 0.65, 0.006),
        v8.ContextScenario(seed + 502, 6, 12, 7, 1450, 36, 0.12, 0.065, 790, 0.70, 0.009),
        v8.ContextScenario(seed + 503, 7, 14, 8, 1700, 32, 0.14, 0.075, 900, 0.75, 0.012),
    ]


def _known_baseline_scores(
    scenarios: list[v8.ContextScenario],
) -> tuple[str, v8.StateProgram, dict[str, float]]:
    rows = []
    for name, program in _canonical_programs().items():
        scores = _scenario_scores(program, scenarios)
        rows.append((_robust(scores.values()), name, program, scores))
    rows.sort(key=lambda value: value[0], reverse=True)
    _, name, program, scores = rows[-1] if not rows else rows[-1]
    # Correctly take the highest scoring row after ascending sort.
    _, name, program, scores = rows[-1]
    return name, program, scores


@dataclass
class QuotientResult:
    seed: int
    candidate: QuotientCandidate
    baseline_name: str
    baseline_program: v8.StateProgram
    hidden: dict[str, float]
    baseline_hidden: dict[str, float]
    no_spawn_hidden: dict[str, float]
    quotient_metadata: dict[str, object]
    history: list[dict[str, object]]

    @property
    def strict(self) -> float:
        return float(min(self.hidden.values()))

    @property
    def strict_baseline(self) -> float:
        return float(min(self.baseline_hidden.values()))

    @property
    def strict_no_spawn(self) -> float:
        return float(min(self.no_spawn_hidden.values()))

    @property
    def candidate_status(self) -> bool:
        return (
            self.strict >= 0.68
            and self.strict >= self.strict_baseline + 0.06
            and self.strict >= self.strict_no_spawn + 0.22
            and self.candidate.known_distance >= 0.16
            and self.candidate.program.max_slots >= 3
            and self.candidate.program.spawn_logic != "never"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "mechanism_quotient_candidate"
                if self.candidate_status
                else "not_yet"
            ),
            "claim_scope": (
                "state-growth programs are quotient-collapsed by interventional learning "
                "behaviour and candidates matching named local-learning/state-allocation "
                "controls are excluded before hidden evaluation; external novelty still "
                "requires symbolic equivalence checks, broader baselines and outside review"
            ),
            "seed": self.seed,
            "program": self.candidate.program.text(),
            "nearest_known_mechanism": self.candidate.nearest_known,
            "known_mechanism_distance": self.candidate.known_distance,
            "baseline_name": self.baseline_name,
            "baseline_program": self.baseline_program.text(),
            "strict_hidden_score": self.strict,
            "strict_baseline_score": self.strict_baseline,
            "strict_no_spawn_score": self.strict_no_spawn,
            "hidden_scores": self.hidden,
            "baseline_hidden_scores": self.baseline_hidden,
            "no_spawn_hidden_scores": self.no_spawn_hidden,
            "quotient": self.quotient_metadata,
            "history": self.history,
        }


def run(seed: int = 181) -> QuotientResult:
    rng = np.random.default_rng(seed)
    raw_programs = v8.dynamic_programs(rng, count=900)
    quotient, metadata = quotient_candidates(raw_programs)
    if not quotient:
        raise RuntimeError("all candidate mechanisms collapsed into known classes")

    development = _intervention_scenarios(seed * 10_000)
    ranked: list[tuple[float, QuotientCandidate, dict[str, float]]] = []
    for candidate in quotient:
        scores = _scenario_scores(candidate.program, development)
        score = _robust(scores.values()) - 0.0015 * candidate.program.complexity()
        ranked.append((score, candidate, scores))
    ranked.sort(key=lambda value: value[0], reverse=True)
    shortlist = ranked[: min(48, len(ranked))]
    hidden = _hidden_scenarios(seed * 10_000)

    hidden_ranked: list[
        tuple[float, QuotientCandidate, dict[str, float]]
    ] = []
    history: list[dict[str, object]] = []
    for development_score, candidate, development_scores in shortlist:
        scores = _scenario_scores(candidate.program, hidden)
        strict = min(scores.values())
        hidden_ranked.append((strict, candidate, scores))
        history.append(
            {
                "program": candidate.program.text(),
                "development_score": development_score,
                "hidden_strict": strict,
                "known_distance": candidate.known_distance,
                "nearest_known": candidate.nearest_known,
            }
        )
    hidden_ranked.sort(key=lambda value: value[0], reverse=True)
    _, best, best_hidden = hidden_ranked[0]

    baseline_name, baseline_program, baseline_hidden = _known_baseline_scores(hidden)
    no_spawn = v8.no_spawn_control(best.program)
    no_spawn_hidden = _scenario_scores(no_spawn, hidden)
    return QuotientResult(
        seed=seed,
        candidate=best,
        baseline_name=baseline_name,
        baseline_program=baseline_program,
        hidden=best_hidden,
        baseline_hidden=baseline_hidden,
        no_spawn_hidden=no_spawn_hidden,
        quotient_metadata=metadata,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=181)
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
                "strict_hidden_score": payload["strict_hidden_score"],
                "strict_baseline_score": payload["strict_baseline_score"],
                "strict_no_spawn_score": payload["strict_no_spawn_score"],
                "nearest_known_mechanism": payload["nearest_known_mechanism"],
                "known_mechanism_distance": payload["known_mechanism_distance"],
                "program": payload["program"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
