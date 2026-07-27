from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .research_v2 import _shift_fixed
from .resilience_v2 import make_obstacle_mask


def _softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass(frozen=True)
class CompetitiveGenome:
    """A local law that competitively selects one signed source per cell."""

    magnitude_gains: np.ndarray
    direction_bias: np.ndarray
    temperature: float
    signal_scale: float
    inertia: float

    @classmethod
    def random(cls, rng: np.random.Generator) -> "CompetitiveGenome":
        return cls(
            magnitude_gains=rng.uniform(0.25, 1.75, 5),
            direction_bias=rng.normal(0.0, 0.35, 5),
            temperature=float(rng.uniform(1.0, 8.0)),
            signal_scale=float(rng.uniform(0.65, 2.2)),
            inertia=float(rng.uniform(0.0, 0.55)),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        sigma: float = 0.12,
        rate: float = 0.35,
    ) -> "CompetitiveGenome":
        def mutate_array(value: np.ndarray, low: float, high: float) -> np.ndarray:
            mask = rng.random(value.shape) < rate
            return np.clip(
                value + mask * rng.normal(0.0, sigma, value.shape),
                low,
                high,
            )

        def mutate_scalar(value: float, low: float, high: float, scale: float = 1.0) -> float:
            if rng.random() >= rate:
                return value
            return float(np.clip(value + rng.normal(0.0, sigma * scale), low, high))

        return CompetitiveGenome(
            magnitude_gains=mutate_array(self.magnitude_gains, 0.0, 4.0),
            direction_bias=mutate_array(self.direction_bias, -3.0, 3.0),
            temperature=mutate_scalar(self.temperature, 0.2, 20.0, 4.0),
            signal_scale=mutate_scalar(self.signal_scale, 0.25, 3.0, 1.5),
            inertia=mutate_scalar(self.inertia, 0.0, 0.95, 0.5),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "magnitude_gains": self.magnitude_gains.tolist(),
            "direction_bias": self.direction_bias.tolist(),
            "temperature": self.temperature,
            "signal_scale": self.signal_scale,
            "inertia": self.inertia,
        }


class CompetitiveSubstrate:
    """Single-channel bounded grid with persistent dead cells."""

    def __init__(self, genome: CompetitiveGenome, height: int, width: int):
        self.genome = genome
        self.state = np.zeros((height, width), dtype=np.float64)

    def reset(self, state: np.ndarray) -> None:
        if state.shape != self.state.shape:
            raise ValueError(f"state shape must be {self.state.shape}")
        self.state = np.asarray(state, dtype=np.float64).copy()

    def step(self, dead: np.ndarray) -> np.ndarray:
        if dead.shape != self.state.shape:
            raise ValueError("dead mask must match state")
        # Sources are self, north, south, west and east. Each output cell
        # receives these values without wraparound.
        candidates = np.stack(
            [
                self.state,
                _shift_fixed(self.state[:, :, None], 1, 0)[:, :, 0],
                _shift_fixed(self.state[:, :, None], -1, 0)[:, :, 0],
                _shift_fixed(self.state[:, :, None], 0, 1)[:, :, 0],
                _shift_fixed(self.state[:, :, None], 0, -1)[:, :, 0],
            ],
            axis=0,
        )
        logits = self.genome.temperature * (
            np.abs(candidates) * self.genome.magnitude_gains[:, None, None]
            + self.genome.direction_bias[:, None, None]
        )
        weights = _softmax(logits, axis=0)
        selected = np.sum(weights * candidates, axis=0)
        proposal = np.tanh(self.genome.signal_scale * selected)
        self.state = np.clip(
            self.genome.inertia * self.state
            + (1.0 - self.genome.inertia) * proposal,
            -1.0,
            1.0,
        )
        self.state[dead] = 0.0
        return self.state


@dataclass(frozen=True)
class CompetitiveEvaluation:
    score: float
    case_scores: tuple[float, ...]


def evaluate_competitive_relay(
    genome: CompetitiveGenome,
    width: int,
    height: int,
    damage_fraction: float,
    seeds: tuple[int, ...],
    amplitude: float = 0.9,
) -> CompetitiveEvaluation:
    case_scores: list[float] = []
    for seed in seeds:
        dead = make_obstacle_mask(seed, height, width, damage_fraction)
        for message in (-amplitude, amplitude):
            world = CompetitiveSubstrate(genome, height, width)
            initial = np.zeros((height, width), dtype=np.float64)
            initial[:, 0] = message
            initial[dead] = 0.0
            world.reset(initial)

            early: list[float] = []
            arrivals: list[tuple[float, np.ndarray, float]] = []
            total_steps = width - 1 + 2 * height
            for tick in range(total_steps):
                world.step(dead)
                destination_rows = world.state[:, -1].copy()
                destination = float(np.mean(destination_rows))
                if tick < width - 2:
                    early.append(abs(destination))
                else:
                    arrivals.append(
                        (
                            destination,
                            destination_rows,
                            float(np.mean(np.abs(world.state))),
                        )
                    )

            alignments = [
                math.exp(-5.0 * (destination - message) ** 2)
                for destination, _, _ in arrivals
            ]
            best_index = int(np.argmax(alignments))
            destination, destination_rows, energy = arrivals[best_index]
            row_agreement = float(np.mean(np.sign(message) * destination_rows > 0.0))
            early_leakage = max(early) if early else 0.0
            case_score = float(
                np.clip(
                    0.91 * alignments[best_index]
                    + 0.07 * row_agreement
                    + 0.02 * (1.0 - energy)
                    - 0.20 * early_leakage,
                    0.0,
                    1.0,
                )
            )
            case_scores.append(case_score)
    return CompetitiveEvaluation(
        score=float(min(case_scores)),
        case_scores=tuple(case_scores),
    )


def hand_flood_baseline() -> CompetitiveGenome:
    """Transparent upper control: approximate signed max-absolute flood fill."""
    return CompetitiveGenome(
        magnitude_gains=np.ones(5, dtype=np.float64),
        direction_bias=np.array([0.0, 0.0, 0.0, 0.10, -0.10]),
        temperature=18.0,
        signal_scale=float(np.arctanh(0.9) / 0.9),
        inertia=0.0,
    )


def _robust_score(values: list[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    tail = ordered[: max(1, int(np.ceil(len(ordered) * 0.40)))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


@dataclass(frozen=True)
class CompetitiveSearchConfig:
    population_size: int = 72
    elite_count: int = 12
    generations: int = 70
    seed: int = 41


@dataclass
class CompetitiveSearchResult:
    best_genome: CompetitiveGenome
    validation_score: float
    strict_hidden_score: float
    hand_baseline_score: float
    hidden_scores: dict[str, float]
    history: list[dict[str, float]]

    @property
    def internal_milestone(self) -> bool:
        return (
            self.validation_score >= 0.78
            and self.strict_hidden_score >= 0.72
            and self.strict_hidden_score >= 0.82 * self.hand_baseline_score
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "internal_milestone" if self.internal_milestone else "not_yet",
            "claim_scope": (
                "evolved competitive local routing approaches a transparent max-flood control "
                "across unseen damaged grids; this is not a new universal form of computation"
            ),
            "validation_score": self.validation_score,
            "strict_hidden_score": self.strict_hidden_score,
            "hand_baseline_score": self.hand_baseline_score,
            "hidden_scores": self.hidden_scores,
            "history": self.history,
            "best_genome": self.best_genome.to_dict(),
        }


def _scenario_batch(
    rng: np.random.Generator,
    generation: int,
) -> list[tuple[int, int, float, int]]:
    progress = generation / 69.0
    max_width = int(round(14 + 30 * progress))
    max_damage = 0.08 + 0.25 * progress
    scenarios: list[tuple[int, int, float, int]] = []
    for index in range(7):
        width = int(rng.integers(max(8, max_width // 2), max_width + 1))
        height = max(7, int(round(width * rng.uniform(0.42, 0.62))))
        damage = float(rng.uniform(max(0.0, max_damage - 0.16), max_damage))
        scenarios.append(
            (
                width,
                height,
                damage,
                200_000 + generation * 101 + index * 13,
            )
        )
    return scenarios


def _score_batch(
    genome: CompetitiveGenome,
    scenarios: list[tuple[int, int, float, int]],
) -> float:
    values = [
        evaluate_competitive_relay(
            genome,
            width,
            height,
            damage,
            seeds=(seed,),
        ).score
        for width, height, damage, seed in scenarios
    ]
    return _robust_score(values)


def search_competitive_routing(
    config: CompetitiveSearchConfig | None = None,
) -> CompetitiveSearchResult:
    config = config or CompetitiveSearchConfig()
    rng = np.random.default_rng(config.seed)
    population = [
        CompetitiveGenome.random(rng)
        for _ in range(config.population_size)
    ]
    history: list[dict[str, float]] = []
    champion: tuple[float, CompetitiveGenome] | None = None

    validation = [
        (18, 10, 0.12, 310_001),
        (24, 13, 0.20, 310_019),
        (32, 17, 0.27, 310_043),
        (40, 21, 0.31, 310_071),
    ]

    for generation in range(config.generations):
        batch_rng = np.random.default_rng(config.seed * 1_000_003 + generation)
        scenarios = _scenario_batch(batch_rng, generation)
        ranked = sorted(
            ((_score_batch(genome, scenarios), genome) for genome in population),
            key=lambda item: item[0],
            reverse=True,
        )

        shortlist = ranked[: max(18, config.elite_count * 2)]
        validated = sorted(
            ((_score_batch(genome, validation), genome) for _, genome in shortlist),
            key=lambda item: item[0],
            reverse=True,
        )
        if champion is None or validated[0][0] > champion[0]:
            champion = validated[0]

        if generation == 0 or generation == config.generations - 1 or generation % 7 == 0:
            history.append(
                {
                    "generation": float(generation),
                    "batch_best": float(ranked[0][0]),
                    "validation_best": float(validated[0][0]),
                    "batch_median": float(np.median([item[0] for item in ranked])),
                }
            )

        elites = [item[1] for item in validated[: config.elite_count]]
        next_population = list(elites)
        progress = generation / max(1, config.generations - 1)
        sigma = 0.16 * (1.0 - 0.78 * progress) + 0.012
        while len(next_population) < config.population_size:
            if rng.random() < 0.03:
                next_population.append(CompetitiveGenome.random(rng))
            else:
                parent = elites[int(rng.integers(0, len(elites)))]
                next_population.append(parent.mutate(rng, sigma=sigma, rate=0.38))
        population = next_population

    assert champion is not None
    best = champion[1]
    final_validation = _score_batch(best, validation)
    hidden_groups = {
        "48x25@0.33": [(48, 25, 0.33, seed) for seed in (410_003, 410_031, 410_067)],
        "60x31@0.35": [(60, 31, 0.35, seed) for seed in (420_007, 420_037, 420_073)],
        "72x37@0.37": [(72, 37, 0.37, seed) for seed in (430_009, 430_041, 430_079)],
    }

    hidden_scores = {
        name: min(
            evaluate_competitive_relay(
                best,
                width,
                height,
                damage,
                seeds=(seed,),
            ).score
            for width, height, damage, seed in scenarios
        )
        for name, scenarios in hidden_groups.items()
    }
    hand = hand_flood_baseline()
    hand_hidden = {
        name: min(
            evaluate_competitive_relay(
                hand,
                width,
                height,
                damage,
                seeds=(seed,),
            ).score
            for width, height, damage, seed in scenarios
        )
        for name, scenarios in hidden_groups.items()
    }
    return CompetitiveSearchResult(
        best_genome=best,
        validation_score=float(final_validation),
        strict_hidden_score=float(min(hidden_scores.values())),
        hand_baseline_score=float(min(hand_hidden.values())),
        hidden_scores=hidden_scores,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--population", type=int, default=72)
    parser.add_argument("--generations", type=int, default=70)
    args = parser.parse_args()

    result = search_competitive_routing(
        CompetitiveSearchConfig(
            population_size=args.population,
            elite_count=max(8, min(14, args.population // 6)),
            generations=args.generations,
            seed=args.seed,
        )
    )
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "validation_score": payload["validation_score"],
                "strict_hidden_score": payload["strict_hidden_score"],
                "hand_baseline_score": payload["hand_baseline_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
