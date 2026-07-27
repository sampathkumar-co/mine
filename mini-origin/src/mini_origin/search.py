from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .genome import Genome
from .tasks import Task, default_tasks


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 36
    generations: int = 25
    elite_count: int = 6
    channels: int = 3
    mutation_sigma: float = 0.14
    mutation_rate: float = 0.20
    evaluation_seeds: tuple[int, ...] = (11, 29, 47)
    complexity_penalty: float = 0.015
    seed: int = 7

    def validate(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if self.channels < 1:
            raise ValueError("channels must be positive")


@dataclass
class EvolutionResult:
    best_genome: Genome
    best_fitness: float
    task_scores: dict[str, float]
    history: list[dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "best_fitness": self.best_fitness,
            "task_scores": self.task_scores,
            "history": self.history,
            "best_genome": self.best_genome.to_dict(),
        }


def score_genome(genome: Genome, tasks: Iterable[Task], seeds: tuple[int, ...], complexity_penalty: float) -> tuple[float, dict[str, float]]:
    task_scores: dict[str, float] = {}
    for task in tasks:
        values = [task.evaluate(genome, seed) for seed in seeds]
        task_scores[task.name] = float(np.mean(values))
    mean_score = float(np.mean(list(task_scores.values())))
    fitness = mean_score - complexity_penalty * genome.complexity()
    return fitness, task_scores


def evolve(config: EvolutionConfig | None = None, tasks: list[Task] | None = None) -> EvolutionResult:
    config = config or EvolutionConfig()
    config.validate()
    tasks = tasks or default_tasks()
    if not tasks:
        raise ValueError("at least one task is required")

    rng = np.random.default_rng(config.seed)
    population = [Genome.random(rng, config.channels) for _ in range(config.population_size)]
    history: list[dict[str, float]] = []
    global_best: tuple[float, Genome, dict[str, float]] | None = None

    for generation in range(config.generations):
        ranked: list[tuple[float, Genome, dict[str, float]]] = []
        for genome in population:
            fitness, task_scores = score_genome(genome, tasks, config.evaluation_seeds, config.complexity_penalty)
            ranked.append((fitness, genome, task_scores))
        ranked.sort(key=lambda item: item[0], reverse=True)

        best = ranked[0]
        if global_best is None or best[0] > global_best[0]:
            global_best = best
        history.append(
            {
                "generation": float(generation),
                "best": float(best[0]),
                "median": float(np.median([item[0] for item in ranked])),
                "mean": float(np.mean([item[0] for item in ranked])),
            }
        )

        elites = [item[1] for item in ranked[: config.elite_count]]
        next_population = elites.copy()
        while len(next_population) < config.population_size:
            parent_a = elites[int(rng.integers(0, len(elites)))]
            parent_b = elites[int(rng.integers(0, len(elites)))]
            child = parent_a.crossover(parent_b, rng).mutate(rng, sigma=config.mutation_sigma, rate=config.mutation_rate)
            next_population.append(child)
        population = next_population

    assert global_best is not None
    return EvolutionResult(
        best_genome=global_best[1],
        best_fitness=float(global_best[0]),
        task_scores=global_best[2],
        history=history,
    )
