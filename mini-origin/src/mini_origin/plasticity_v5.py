from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _orthogonal_mapping(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _normalised_vectors(
    rng: np.random.Generator,
    count: int,
    dimension: int,
) -> np.ndarray:
    values = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _prediction_similarity(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = np.sum(prediction * target, axis=1) / np.maximum(
        pred_norm * target_norm,
        1e-12,
    )
    direction = np.clip(cosine, 0.0, 1.0)
    amplitude = np.exp(-2.0 * (pred_norm - target_norm) ** 2)
    return float(np.mean(direction * amplitude))


@dataclass(frozen=True)
class PlasticityGenome:
    """Dimension-agnostic coefficients for a local supervised plasticity law."""

    learning_rate: float
    error_coefficient: float
    hebb_coefficient: float
    prediction_coefficient: float
    decay: float
    memory_clip: float
    consensus_mix: float
    observation_dropout: float

    @classmethod
    def random(cls, rng: np.random.Generator) -> "PlasticityGenome":
        return cls(
            learning_rate=float(rng.uniform(0.015, 0.55)),
            error_coefficient=float(rng.uniform(-1.5, 2.5)),
            hebb_coefficient=float(rng.uniform(-1.5, 2.5)),
            prediction_coefficient=float(rng.uniform(-1.5, 1.5)),
            decay=float(rng.uniform(0.0, 0.08)),
            memory_clip=float(rng.uniform(0.5, 5.0)),
            consensus_mix=float(rng.uniform(0.0, 1.0)),
            observation_dropout=float(rng.uniform(0.0, 0.35)),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        sigma: float = 0.12,
        rate: float = 0.35,
    ) -> "PlasticityGenome":
        def scalar(value: float, low: float, high: float, scale: float = 1.0) -> float:
            if rng.random() >= rate:
                return value
            return float(np.clip(value + rng.normal(0.0, sigma * scale), low, high))

        return PlasticityGenome(
            learning_rate=scalar(self.learning_rate, 0.001, 0.9, 1.8),
            error_coefficient=scalar(self.error_coefficient, -3.0, 4.0, 3.0),
            hebb_coefficient=scalar(self.hebb_coefficient, -3.0, 4.0, 3.0),
            prediction_coefficient=scalar(self.prediction_coefficient, -3.0, 3.0, 2.5),
            decay=scalar(self.decay, 0.0, 0.25, 0.5),
            memory_clip=scalar(self.memory_clip, 0.25, 8.0, 8.0),
            consensus_mix=scalar(self.consensus_mix, 0.0, 1.0, 1.5),
            observation_dropout=scalar(self.observation_dropout, 0.0, 0.55, 1.0),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "error_coefficient": self.error_coefficient,
            "hebb_coefficient": self.hebb_coefficient,
            "prediction_coefficient": self.prediction_coefficient,
            "decay": self.decay,
            "memory_clip": self.memory_clip,
            "consensus_mix": self.consensus_mix,
            "observation_dropout": self.observation_dropout,
        }


class DistributedPlasticMemory:
    """Redundant cells that update only from their own observations and memory."""

    def __init__(
        self,
        genome: PlasticityGenome,
        cells: int,
        dimension: int,
        rng: np.random.Generator,
        initial_scale: float = 0.015,
    ):
        if cells < 4:
            raise ValueError("cells must be at least 4")
        if dimension < 2:
            raise ValueError("dimension must be at least 2")
        self.genome = genome
        self.cells = cells
        self.dimension = dimension
        self.memory = rng.normal(
            0.0,
            initial_scale,
            size=(cells, dimension, dimension),
        )
        self.alive = np.ones(cells, dtype=bool)

    def learn(
        self,
        key: np.ndarray,
        target: np.ndarray,
        rng: np.random.Generator,
        noise: float,
    ) -> None:
        if key.shape != (self.dimension,) or target.shape != (self.dimension,):
            raise ValueError("key and target must match memory dimension")
        visible = self.alive & (
            rng.random(self.cells) >= self.genome.observation_dropout
        )
        if not np.any(visible):
            return

        local_key = key + rng.normal(0.0, noise, (self.cells, self.dimension))
        local_target = target + rng.normal(0.0, noise, (self.cells, self.dimension))
        prediction = np.einsum("cij,cj->ci", self.memory, local_key)
        error = local_target - prediction

        error_term = np.einsum("ci,cj->cij", error, local_key)
        hebb_term = np.einsum("ci,cj->cij", local_target, local_key)
        prediction_term = np.einsum("ci,cj->cij", prediction, local_key)

        update = self.genome.learning_rate * (
            self.genome.error_coefficient * error_term
            + self.genome.hebb_coefficient * hebb_term
            + self.genome.prediction_coefficient * prediction_term
        )
        update -= self.genome.decay * self.memory
        self.memory[visible] = np.clip(
            self.memory[visible] + update[visible],
            -self.genome.memory_clip,
            self.genome.memory_clip,
        )

    def damage(self, fraction: float, rng: np.random.Generator) -> np.ndarray:
        if not 0.0 <= fraction < 1.0:
            raise ValueError("damage fraction must be in [0, 1)")
        candidates = np.flatnonzero(self.alive)
        kill_count = min(
            len(candidates) - 1,
            int(round(len(candidates) * fraction)),
        )
        killed = np.zeros(self.cells, dtype=bool)
        if kill_count > 0:
            selected = rng.choice(candidates, size=kill_count, replace=False)
            self.alive[selected] = False
            self.memory[selected] = 0.0
            killed[selected] = True
        return killed

    def predict(self, keys: np.ndarray) -> np.ndarray:
        if keys.ndim != 2 or keys.shape[1] != self.dimension:
            raise ValueError("keys must have shape [examples, dimension]")
        alive_memory = self.memory[self.alive]
        cell_predictions = np.einsum("cij,nj->cni", alive_memory, keys)
        mean_prediction = np.mean(cell_predictions, axis=0)
        median_prediction = np.median(cell_predictions, axis=0)
        mix = self.genome.consensus_mix
        return (1.0 - mix) * mean_prediction + mix * median_prediction


@dataclass(frozen=True)
class LearningScenario:
    dimension: int
    cells: int
    examples_per_dimension: int
    repetitions: int
    noise: float
    damage_fraction: float
    seed: int


@dataclass(frozen=True)
class LearningEvaluation:
    score: float
    pre_damage_score: float
    post_damage_score: float
    retention: float


def evaluate_learning(
    genome: PlasticityGenome,
    scenario: LearningScenario,
) -> LearningEvaluation:
    rng = np.random.default_rng(scenario.seed)
    mapping = _orthogonal_mapping(rng, scenario.dimension)
    train_count = scenario.dimension * scenario.examples_per_dimension
    train_keys = _normalised_vectors(rng, train_count, scenario.dimension)
    train_targets = train_keys @ mapping.T
    test_keys = _normalised_vectors(rng, scenario.dimension * 8, scenario.dimension)
    test_targets = test_keys @ mapping.T

    substrate = DistributedPlasticMemory(
        genome,
        cells=scenario.cells,
        dimension=scenario.dimension,
        rng=rng,
    )
    for _ in range(scenario.repetitions):
        order = rng.permutation(train_count)
        for index in order:
            substrate.learn(
                train_keys[index],
                train_targets[index],
                rng,
                noise=scenario.noise,
            )

    pre_prediction = substrate.predict(test_keys)
    pre_score = _prediction_similarity(pre_prediction, test_targets)
    substrate.damage(scenario.damage_fraction, rng)
    post_prediction = substrate.predict(test_keys)
    post_score = _prediction_similarity(post_prediction, test_targets)
    retention = post_score / max(pre_score, 1e-12)
    score = float(
        np.clip(
            0.78 * post_score
            + 0.17 * min(retention, 1.0)
            + 0.05 * pre_score,
            0.0,
            1.0,
        )
    )
    return LearningEvaluation(
        score=score,
        pre_damage_score=float(pre_score),
        post_damage_score=float(post_score),
        retention=float(retention),
    )


def no_learning_control(genome: PlasticityGenome) -> PlasticityGenome:
    return PlasticityGenome(
        learning_rate=0.0,
        error_coefficient=genome.error_coefficient,
        hebb_coefficient=genome.hebb_coefficient,
        prediction_coefficient=genome.prediction_coefficient,
        decay=genome.decay,
        memory_clip=genome.memory_clip,
        consensus_mix=genome.consensus_mix,
        observation_dropout=genome.observation_dropout,
    )


def no_error_control(genome: PlasticityGenome) -> PlasticityGenome:
    return PlasticityGenome(
        learning_rate=genome.learning_rate,
        error_coefficient=0.0,
        hebb_coefficient=genome.hebb_coefficient,
        prediction_coefficient=genome.prediction_coefficient,
        decay=genome.decay,
        memory_clip=genome.memory_clip,
        consensus_mix=genome.consensus_mix,
        observation_dropout=genome.observation_dropout,
    )


def hand_delta_control() -> PlasticityGenome:
    return PlasticityGenome(
        learning_rate=0.20,
        error_coefficient=1.0,
        hebb_coefficient=0.0,
        prediction_coefficient=0.0,
        decay=0.001,
        memory_clip=3.0,
        consensus_mix=0.0,
        observation_dropout=0.12,
    )


def _robust_score(values: list[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    tail = ordered[: max(1, int(math.ceil(len(ordered) * 0.4)))]
    return float(0.55 * ordered[0] + 0.25 * np.mean(tail) + 0.20 * np.mean(ordered))


@dataclass(frozen=True)
class PlasticitySearchConfig:
    population_size: int = 72
    elite_count: int = 12
    generations: int = 50
    seed: int = 51


@dataclass
class PlasticitySearchResult:
    best_genome: PlasticityGenome
    validation_score: float
    strict_hidden_score: float
    hidden_retention: float
    no_learning_score: float
    no_error_score: float
    hand_delta_score: float
    hidden_scores: dict[str, float]
    history: list[dict[str, float]]

    @property
    def internal_breakthrough(self) -> bool:
        return (
            self.validation_score >= 0.84
            and self.strict_hidden_score >= 0.78
            and self.hidden_retention >= 0.90
            and self.strict_hidden_score >= self.no_learning_score + 0.60
            and self.strict_hidden_score >= self.no_error_score + 0.18
            and self.strict_hidden_score >= 0.82 * self.hand_delta_score
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "within_lifetime_learning_breakthrough"
                if self.internal_breakthrough
                else "not_yet"
            ),
            "claim_scope": (
                "one dimension-agnostic local plasticity law learns unseen orthogonal mappings "
                "during its lifetime and retains them after distributed cell death; the update "
                "language remains human-designed and this is not a world-level learning breakthrough"
            ),
            "validation_score": self.validation_score,
            "strict_hidden_score": self.strict_hidden_score,
            "hidden_retention": self.hidden_retention,
            "no_learning_score": self.no_learning_score,
            "no_error_score": self.no_error_score,
            "hand_delta_score": self.hand_delta_score,
            "hidden_scores": self.hidden_scores,
            "history": self.history,
            "best_genome": self.best_genome.to_dict(),
        }


def _training_scenarios(generation: int, seed: int) -> list[LearningScenario]:
    rng = np.random.default_rng(seed * 1_000_003 + generation)
    scenarios: list[LearningScenario] = []
    for index in range(6):
        dimension = int(rng.choice((3, 4)))
        scenarios.append(
            LearningScenario(
                dimension=dimension,
                cells=int(rng.integers(32, 65)),
                examples_per_dimension=int(rng.integers(4, 7)),
                repetitions=int(rng.integers(2, 5)),
                noise=float(rng.uniform(0.005, 0.035)),
                damage_fraction=float(rng.uniform(0.10, 0.34)),
                seed=70_000 + generation * 101 + index * 17,
            )
        )
    return scenarios


def _validation_scenarios() -> list[LearningScenario]:
    return [
        LearningScenario(3, 48, 6, 4, 0.025, 0.30, 81_001),
        LearningScenario(4, 56, 6, 4, 0.030, 0.35, 81_019),
        LearningScenario(3, 40, 5, 5, 0.035, 0.38, 81_043),
        LearningScenario(4, 64, 5, 5, 0.035, 0.40, 81_071),
    ]


def _hidden_scenarios() -> dict[str, list[LearningScenario]]:
    return {
        "5d@45pct": [
            LearningScenario(5, 72, 7, 5, 0.045, 0.45, seed)
            for seed in (91_003, 91_031, 91_067)
        ],
        "6d@55pct": [
            LearningScenario(6, 80, 8, 5, 0.060, 0.55, seed)
            for seed in (92_007, 92_037, 92_073)
        ],
        "8d@65pct": [
            LearningScenario(8, 96, 9, 6, 0.080, 0.65, seed)
            for seed in (93_009, 93_041, 93_079)
        ],
    }


def _score_scenarios(
    genome: PlasticityGenome,
    scenarios: list[LearningScenario],
) -> tuple[float, list[LearningEvaluation]]:
    evaluations = [evaluate_learning(genome, scenario) for scenario in scenarios]
    return _robust_score([value.score for value in evaluations]), evaluations


def search_plasticity(
    config: PlasticitySearchConfig | None = None,
) -> PlasticitySearchResult:
    config = config or PlasticitySearchConfig()
    rng = np.random.default_rng(config.seed)
    population = [PlasticityGenome.random(rng) for _ in range(config.population_size)]
    validation = _validation_scenarios()
    history: list[dict[str, float]] = []
    champion: tuple[float, PlasticityGenome] | None = None

    for generation in range(config.generations):
        training = _training_scenarios(generation, config.seed)
        ranked = sorted(
            ((_score_scenarios(genome, training)[0], genome) for genome in population),
            key=lambda item: item[0],
            reverse=True,
        )
        shortlist = ranked[: max(18, config.elite_count * 2)]
        validated = sorted(
            ((_score_scenarios(genome, validation)[0], genome) for _, genome in shortlist),
            key=lambda item: item[0],
            reverse=True,
        )
        if champion is None or validated[0][0] > champion[0]:
            champion = validated[0]

        if generation == 0 or generation == config.generations - 1 or generation % 5 == 0:
            history.append(
                {
                    "generation": float(generation),
                    "training_best": float(ranked[0][0]),
                    "validation_best": float(validated[0][0]),
                    "training_median": float(np.median([item[0] for item in ranked])),
                }
            )

        elites = [item[1] for item in validated[: config.elite_count]]
        next_population = list(elites)
        progress = generation / max(1, config.generations - 1)
        sigma = 0.15 * (1.0 - 0.78 * progress) + 0.012
        while len(next_population) < config.population_size:
            if rng.random() < 0.035:
                next_population.append(PlasticityGenome.random(rng))
            else:
                parent = elites[int(rng.integers(0, len(elites)))]
                next_population.append(parent.mutate(rng, sigma=sigma, rate=0.38))
        population = next_population

    assert champion is not None
    best = champion[1]
    validation_score, _ = _score_scenarios(best, validation)
    hidden_groups = _hidden_scenarios()

    hidden_scores: dict[str, float] = {}
    hidden_retentions: list[float] = []
    for name, scenarios in hidden_groups.items():
        evaluations = [evaluate_learning(best, scenario) for scenario in scenarios]
        hidden_scores[name] = float(min(value.score for value in evaluations))
        hidden_retentions.extend(value.retention for value in evaluations)

    all_hidden = [scenario for group in hidden_groups.values() for scenario in group]
    no_learning_score, _ = _score_scenarios(no_learning_control(best), all_hidden)
    no_error_score, _ = _score_scenarios(no_error_control(best), all_hidden)
    hand_delta_score, _ = _score_scenarios(hand_delta_control(), all_hidden)

    return PlasticitySearchResult(
        best_genome=best,
        validation_score=float(validation_score),
        strict_hidden_score=float(min(hidden_scores.values())),
        hidden_retention=float(min(hidden_retentions)),
        no_learning_score=float(no_learning_score),
        no_error_score=float(no_error_score),
        hand_delta_score=float(hand_delta_score),
        hidden_scores=hidden_scores,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--population", type=int, default=72)
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()

    result = search_plasticity(
        PlasticitySearchConfig(
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
                "hidden_retention": payload["hidden_retention"],
                "no_learning_score": payload["no_learning_score"],
                "no_error_score": payload["no_error_score"],
                "hand_delta_score": payload["hand_delta_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
