from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .research_v2 import DirectionalGenome, DirectionalSubstrate


def genome_from_dict(data: dict[str, object]) -> DirectionalGenome:
    return DirectionalGenome(
        proposal_weights=np.asarray(data["proposal_weights"], dtype=np.float64),
        gate_weights=np.asarray(data["gate_weights"], dtype=np.float64),
        proposal_bias=np.asarray(data["proposal_bias"], dtype=np.float64),
        gate_bias=np.asarray(data["gate_bias"], dtype=np.float64),
    )


def make_obstacle_mask(
    seed: int,
    height: int,
    width: int,
    damage_fraction: float,
) -> np.ndarray:
    """Create random cell death plus a partial wall with a hidden gap."""
    if not 0.0 <= damage_fraction < 0.8:
        raise ValueError("damage_fraction must be in [0, 0.8)")
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=bool)
    if width > 2:
        mask[:, 1:-1] = (
            rng.random((height, width - 2)) < damage_fraction * 0.40
        )
    if damage_fraction > 0.0 and width >= 6 and height >= 5:
        low = max(2, width // 3)
        high = min(width - 2, (2 * width) // 3 + 1)
        wall_x = int(rng.integers(low, high))
        gap_size = max(2, int(round(height * (0.72 - damage_fraction))))
        gap_size = min(height - 2, gap_size)
        gap_y = int(rng.integers(0, height - gap_size + 1))
        mask[:, wall_x] = True
        mask[gap_y : gap_y + gap_size, wall_x] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask


@dataclass(frozen=True)
class ResilientRelayEvaluation:
    score: float
    case_scores: tuple[float, ...]
    base_damage_fraction: float


def evaluate_resilient_relay(
    genome: DirectionalGenome,
    width: int,
    height: int,
    damage_fraction: float,
    seeds: tuple[int, ...] = (0, 1),
    amplitude: float = 0.9,
) -> ResilientRelayEvaluation:
    """Route a bipolar signal around persistent dead cells and partial walls."""
    case_scores: list[float] = []
    for seed in seeds:
        dead = make_obstacle_mask(10_000 + seed, height, width, damage_fraction)
        for message in (-amplitude, amplitude):
            world = DirectionalSubstrate(genome, height, width)
            initial = np.zeros_like(world.state)
            initial[:, 0, 0] = message
            if genome.channels > 3:
                initial[:, :, 3] = -0.8
            initial[dead, :] = 0.0
            world.reset(initial)

            external = np.zeros_like(world.state)
            if genome.channels > 3:
                external[:, :, 3] = -0.8
            external[dead, :] = 0.0

            early_values: list[float] = []
            arrival_records: list[tuple[float, np.ndarray, float]] = []
            total_steps = width - 1 + 2 * height
            for tick in range(total_steps):
                world.step(external)
                world.state[dead, :] = 0.0
                destination_rows = world.state[:, -1, 0].copy()
                destination = float(np.mean(destination_rows))
                if tick < width - 2:
                    early_values.append(abs(destination))
                else:
                    arrival_records.append(
                        (
                            destination,
                            destination_rows,
                            float(np.mean(np.abs(world.state[:, :, 0]))),
                        )
                    )

            alignments = [
                math.exp(-4.0 * (destination - message) ** 2)
                for destination, _, _ in arrival_records
            ]
            best_index = int(np.argmax(alignments))
            destination, destination_rows, energy = arrival_records[best_index]
            alignment = alignments[best_index]
            row_agreement = float(
                np.mean(np.sign(message) * destination_rows > 0.0)
            )
            early_leakage = max(early_values) if early_values else 0.0
            score = float(
                np.clip(
                    0.90 * alignment
                    + 0.07 * row_agreement
                    + 0.03 * (1.0 - energy)
                    - 0.18 * early_leakage,
                    0.0,
                    1.0,
                )
            )
            case_scores.append(score)

    return ResilientRelayEvaluation(
        score=float(min(case_scores)),
        case_scores=tuple(case_scores),
        base_damage_fraction=damage_fraction,
    )


@dataclass(frozen=True)
class ResilienceConfig:
    population_size: int = 64
    elite_count: int = 10
    seed: int = 21
    scale: float = 0.65
    mutation_rate: float = 0.28


@dataclass
class ResilienceResult:
    best_genome: DirectionalGenome
    training_score: float
    hidden_scores: dict[str, float]
    base_hidden_scores: dict[str, float]
    history: list[dict[str, float]]

    @property
    def strict_hidden_score(self) -> float:
        return float(min(self.hidden_scores.values()))

    @property
    def base_strict_hidden_score(self) -> float:
        return float(min(self.base_hidden_scores.values()))

    @property
    def project_breakthrough(self) -> bool:
        return (
            self.training_score >= 0.80
            and self.strict_hidden_score >= 0.70
            and self.strict_hidden_score >= self.base_strict_hidden_score + 0.35
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "project_breakthrough" if self.project_breakthrough else "not_yet"
            ),
            "claim_scope": (
                "damage-tolerant local signal routing in controlled grids; "
                "not a world-level unconventional-computing breakthrough"
            ),
            "training_score": self.training_score,
            "strict_hidden_score": self.strict_hidden_score,
            "base_strict_hidden_score": self.base_strict_hidden_score,
            "hidden_scores": self.hidden_scores,
            "base_hidden_scores": self.base_hidden_scores,
            "history": self.history,
            "best_genome": self.best_genome.to_dict(),
        }


def adapt_for_resilience(
    base_genome: DirectionalGenome,
    config: ResilienceConfig | None = None,
) -> ResilienceResult:
    config = config or ResilienceConfig()
    rng = np.random.default_rng(config.seed)
    population: list[DirectionalGenome] = [base_genome]
    while len(population) < config.population_size:
        if len(population) < max(4, config.population_size // 4):
            population.append(
                base_genome.mutate(rng, sigma=0.05, rate=0.22)
            )
        else:
            population.append(
                DirectionalGenome.random(rng, base_genome.channels)
            )

    raw_stages = (
        (8, 7, 0.00, 12),
        (10, 7, 0.05, 16),
        (12, 8, 0.10, 20),
        (16, 9, 0.15, 28),
        (20, 11, 0.20, 36),
        (24, 13, 0.22, 42),
        (30, 15, 0.25, 54),
    )
    stages = [
        (
            width,
            height,
            damage,
            max(5, int(round(generations * config.scale))),
        )
        for width, height, damage, generations in raw_stages
    ]
    history: list[dict[str, float]] = []
    stage_best: tuple[float, DirectionalGenome] | None = None

    for stage_index, (width, height, damage, generations) in enumerate(stages):
        stage_seeds = (stage_index * 3, stage_index * 3 + 1)
        for generation in range(generations):
            ranked = sorted(
                (
                    (
                        evaluate_resilient_relay(
                            genome,
                            width,
                            height,
                            damage,
                            seeds=stage_seeds,
                        ).score,
                        genome,
                    )
                    for genome in population
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            stage_best = ranked[0]
            if (
                generation == 0
                or generation == generations - 1
                or generation % 8 == 0
            ):
                history.append(
                    {
                        "stage": float(stage_index),
                        "width": float(width),
                        "damage": float(damage),
                        "generation": float(generation),
                        "best": float(ranked[0][0]),
                        "median": float(
                            np.median([item[0] for item in ranked])
                        ),
                    }
                )
            elites = [item[1] for item in ranked[: config.elite_count]]
            next_population = list(elites)
            progress = generation / max(1, generations - 1)
            sigma = 0.11 * (1.0 - 0.78 * progress) + 0.008
            while len(next_population) < config.population_size:
                if rng.random() < 0.02:
                    next_population.append(
                        DirectionalGenome.random(rng, base_genome.channels)
                    )
                else:
                    parent = elites[int(rng.integers(0, len(elites)))]
                    next_population.append(
                        parent.mutate(
                            rng,
                            sigma=sigma,
                            rate=config.mutation_rate,
                        )
                    )
            population = next_population

    assert stage_best is not None
    final_width, final_height, final_damage, _ = stages[-1]
    final_ranked = sorted(
        (
            (
                evaluate_resilient_relay(
                    genome,
                    final_width,
                    final_height,
                    final_damage,
                    seeds=(18, 19, 20),
                ).score,
                genome,
            )
            for genome in population
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best = final_ranked[0][1]
    training_score = float(final_ranked[0][0])

    hidden_cases = (
        (34, 17, 0.27),
        (40, 19, 0.30),
        (48, 21, 0.32),
    )
    hidden_scores = {
        f"{width}x{height}@{damage:.2f}": evaluate_resilient_relay(
            best,
            width,
            height,
            damage,
            seeds=(101, 103, 107),
        ).score
        for width, height, damage in hidden_cases
    }
    base_hidden_scores = {
        f"{width}x{height}@{damage:.2f}": evaluate_resilient_relay(
            base_genome,
            width,
            height,
            damage,
            seeds=(101, 103, 107),
        ).score
        for width, height, damage in hidden_cases
    }
    return ResilienceResult(
        best_genome=best,
        training_score=training_score,
        hidden_scores=hidden_scores,
        base_hidden_scores=base_hidden_scores,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--scale", type=float, default=0.65)
    args = parser.parse_args()

    base_payload = json.loads(args.base.read_text(encoding="utf-8"))
    base_genome = genome_from_dict(base_payload["best_genome"])
    result = adapt_for_resilience(
        base_genome,
        ResilienceConfig(
            population_size=args.population,
            elite_count=max(6, min(12, args.population // 6)),
            seed=args.seed,
            scale=args.scale,
        ),
    )
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "training_score": payload["training_score"],
                "strict_hidden_score": payload["strict_hidden_score"],
                "base_strict_hidden_score": payload[
                    "base_strict_hidden_score"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
