from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np

from .research_v2 import DirectionalGenome
from .resilience_v2 import evaluate_resilient_relay, genome_from_dict


@dataclass(frozen=True)
class Scenario:
    width: int
    height: int
    damage: float
    obstacle_seed: int


@dataclass(frozen=True)
class RobustResilienceConfig:
    population_size: int = 56
    elite_count: int = 10
    seed: int = 31
    scale: float = 0.70
    mutation_rate: float = 0.30
    restart_rate: float = 0.03

    def validate(self) -> None:
        if self.population_size < 12:
            raise ValueError("population_size must be at least 12")
        if not 3 <= self.elite_count < self.population_size:
            raise ValueError("invalid elite_count")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")


def _robust_score(values: list[float]) -> float:
    """Reward broad competence and strongly punish one-layout collapse."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    tail_count = max(1, int(np.ceil(len(ordered) * 0.35)))
    lower_tail = float(np.mean(ordered[:tail_count]))
    return float(
        0.50 * ordered[0]
        + 0.30 * lower_tail
        + 0.20 * float(np.mean(ordered))
    )


def _sample_training_scenarios(
    rng: np.random.Generator,
    stage_width: int,
    stage_damage: float,
    generation: int,
    stage_index: int,
) -> list[Scenario]:
    """Generate a shared, changing batch so layouts cannot be memorised."""
    scenarios: list[Scenario] = []

    anchor_seed = 50_000 + stage_index * 10_000 + generation * 17
    scenarios.append(
        Scenario(
            width=stage_width,
            height=max(7, stage_width // 2 + 1),
            damage=stage_damage,
            obstacle_seed=anchor_seed,
        )
    )

    min_width = max(8, int(round(stage_width * 0.55)))
    for index in range(4):
        width = int(rng.integers(min_width, stage_width + 1))
        height = max(7, int(round(width * rng.uniform(0.42, 0.62))))
        low_damage = max(0.0, stage_damage - 0.14)
        damage = float(rng.uniform(low_damage, stage_damage + 1e-9))
        scenarios.append(
            Scenario(
                width=width,
                height=height,
                damage=damage,
                obstacle_seed=anchor_seed + 101 + index * 37,
            )
        )

    # Preserve clean transport while adding diversity at a different aspect ratio.
    scenarios.append(
        Scenario(
            width=max(8, stage_width - 3),
            height=max(8, stage_width // 3 + 3),
            damage=0.0,
            obstacle_seed=anchor_seed + 997,
        )
    )
    return scenarios


def _validation_scenarios(stage_width: int, stage_damage: float) -> list[Scenario]:
    widths = sorted(
        {
            max(8, int(round(stage_width * 0.65))),
            max(8, int(round(stage_width * 0.82))),
            stage_width,
        }
    )
    scenarios: list[Scenario] = []
    for width_index, width in enumerate(widths):
        height = max(8, width // 2 + 2)
        for damage_index, damage in enumerate(
            (max(0.0, stage_damage - 0.08), stage_damage)
        ):
            scenarios.append(
                Scenario(
                    width=width,
                    height=height,
                    damage=damage,
                    obstacle_seed=80_000 + width_index * 1_000 + damage_index * 101,
                )
            )
    return scenarios


def _score_scenarios(
    genome: DirectionalGenome,
    scenarios: list[Scenario],
) -> tuple[float, list[float]]:
    values = [
        evaluate_resilient_relay(
            genome,
            scenario.width,
            scenario.height,
            scenario.damage,
            seeds=(scenario.obstacle_seed,),
        ).score
        for scenario in scenarios
    ]
    return _robust_score(values), values


@dataclass
class RobustResilienceResult:
    best_genome: DirectionalGenome
    validation_score: float
    strict_hidden_score: float
    base_strict_hidden_score: float
    hidden_scores: dict[str, float]
    base_hidden_scores: dict[str, float]
    history: list[dict[str, float]]

    @property
    def internal_milestone(self) -> bool:
        return (
            self.validation_score >= 0.72
            and self.strict_hidden_score >= 0.60
            and self.strict_hidden_score >= self.base_strict_hidden_score + 0.30
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "internal_milestone" if self.internal_milestone else "not_yet"
            ),
            "claim_scope": (
                "robust local signal transmission across unseen sizes and persistent damage; "
                "not evidence of a new universal computational substrate"
            ),
            "validation_score": self.validation_score,
            "strict_hidden_score": self.strict_hidden_score,
            "base_strict_hidden_score": self.base_strict_hidden_score,
            "hidden_scores": self.hidden_scores,
            "base_hidden_scores": self.base_hidden_scores,
            "history": self.history,
            "best_genome": self.best_genome.to_dict(),
        }


def discover_robust_resilience(
    base_genome: DirectionalGenome,
    config: RobustResilienceConfig | None = None,
) -> RobustResilienceResult:
    config = config or RobustResilienceConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)

    population: list[DirectionalGenome] = [base_genome]
    while len(population) < config.population_size:
        if len(population) < config.population_size // 2:
            population.append(
                base_genome.mutate(
                    rng,
                    sigma=float(rng.uniform(0.025, 0.14)),
                    rate=float(rng.uniform(0.15, 0.38)),
                )
            )
        else:
            population.append(DirectionalGenome.random(rng, base_genome.channels))

    raw_stages = (
        (10, 0.06, 14),
        (14, 0.12, 18),
        (20, 0.19, 24),
        (28, 0.25, 32),
        (36, 0.31, 44),
    )
    stages = [
        (width, damage, max(6, int(round(generations * config.scale))))
        for width, damage, generations in raw_stages
    ]

    champion: tuple[float, DirectionalGenome] | None = None
    history: list[dict[str, float]] = []

    for stage_index, (stage_width, stage_damage, generations) in enumerate(stages):
        fixed_validation = _validation_scenarios(stage_width, stage_damage)
        for generation in range(generations):
            batch_rng = np.random.default_rng(
                config.seed * 1_000_003 + stage_index * 10_007 + generation
            )
            batch = _sample_training_scenarios(
                batch_rng,
                stage_width,
                stage_damage,
                generation,
                stage_index,
            )
            ranked: list[tuple[float, DirectionalGenome, list[float]]] = []
            for genome in population:
                score, values = _score_scenarios(genome, batch)
                ranked.append((score, genome, values))
            ranked.sort(key=lambda item: item[0], reverse=True)

            # Re-rank a broad shortlist on fixed validation layouts.
            shortlist = ranked[: max(config.elite_count * 2, 16)]
            validated = []
            for _, genome, _ in shortlist:
                validation_score, validation_values = _score_scenarios(
                    genome,
                    fixed_validation,
                )
                validated.append(
                    (validation_score, genome, validation_values)
                )
            validated.sort(key=lambda item: item[0], reverse=True)
            if champion is None or validated[0][0] > champion[0]:
                champion = (validated[0][0], validated[0][1])

            if (
                generation == 0
                or generation == generations - 1
                or generation % 6 == 0
            ):
                history.append(
                    {
                        "stage": float(stage_index),
                        "stage_width": float(stage_width),
                        "stage_damage": float(stage_damage),
                        "generation": float(generation),
                        "batch_best": float(ranked[0][0]),
                        "validation_best": float(validated[0][0]),
                        "validation_worst_case": float(min(validated[0][2])),
                        "batch_median": float(
                            np.median([item[0] for item in ranked])
                        ),
                    }
                )

            elites = [item[1] for item in validated[: config.elite_count]]
            next_population = list(elites)
            progress = generation / max(1, generations - 1)
            sigma = 0.10 * (1.0 - 0.72 * progress) + 0.01
            while len(next_population) < config.population_size:
                if rng.random() < config.restart_rate:
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

        # Carry the best validated rule and its local neighbourhood forward.
        assert champion is not None
        population[0] = champion[1]
        for index in range(1, min(8, len(population))):
            population[index] = champion[1].mutate(
                rng,
                sigma=0.035,
                rate=0.20,
            )

    assert champion is not None
    best = champion[1]
    final_validation, _ = _score_scenarios(
        best,
        _validation_scenarios(stages[-1][0], stages[-1][1]),
    )

    hidden_groups = {
        "40x21@0.30": [Scenario(40, 21, 0.30, seed) for seed in (120_001, 120_019, 120_041, 120_079)],
        "48x25@0.33": [Scenario(48, 25, 0.33, seed) for seed in (130_003, 130_027, 130_051, 130_087)],
        "60x31@0.35": [Scenario(60, 31, 0.35, seed) for seed in (140_009, 140_033, 140_063, 140_091)],
    }

    def group_score(genome: DirectionalGenome, scenarios: list[Scenario]) -> float:
        values = [
            evaluate_resilient_relay(
                genome,
                scenario.width,
                scenario.height,
                scenario.damage,
                seeds=(scenario.obstacle_seed,),
            ).score
            for scenario in scenarios
        ]
        return float(min(values))

    hidden_scores = {
        name: group_score(best, scenarios)
        for name, scenarios in hidden_groups.items()
    }
    base_hidden_scores = {
        name: group_score(base_genome, scenarios)
        for name, scenarios in hidden_groups.items()
    }
    return RobustResilienceResult(
        best_genome=best,
        validation_score=float(final_validation),
        strict_hidden_score=float(min(hidden_scores.values())),
        base_strict_hidden_score=float(min(base_hidden_scores.values())),
        hidden_scores=hidden_scores,
        base_hidden_scores=base_hidden_scores,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--population", type=int, default=56)
    parser.add_argument("--scale", type=float, default=0.70)
    args = parser.parse_args()

    base_payload = json.loads(args.base.read_text(encoding="utf-8"))
    base_genome = genome_from_dict(base_payload["best_genome"])
    result = discover_robust_resilience(
        base_genome,
        RobustResilienceConfig(
            population_size=args.population,
            elite_count=max(6, min(12, args.population // 6)),
            seed=args.seed,
            scale=args.scale,
        ),
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
                "base_strict_hidden_score": payload["base_strict_hidden_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
