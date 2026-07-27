from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _shift_fixed(state: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift without wraparound. output[y+dy,x+dx] receives input[y,x]."""
    out = np.zeros_like(state)
    h, w = state.shape[:2]
    dst_y = slice(max(0, dy), min(h, h + dy))
    dst_x = slice(max(0, dx), min(w, w + dx))
    src_y = slice(max(0, -dy), min(h, h - dy))
    src_x = slice(max(0, -dx), min(w, w - dx))
    out[dst_y, dst_x] = state[src_y, src_x]
    return out


@dataclass(frozen=True)
class DirectionalGenome:
    """Shared gated local rule with directional perception."""

    proposal_weights: np.ndarray
    gate_weights: np.ndarray
    proposal_bias: np.ndarray
    gate_bias: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.proposal_bias.size)

    @classmethod
    def random(cls, rng: np.random.Generator, channels: int = 4) -> "DirectionalGenome":
        return cls(
            proposal_weights=rng.normal(0.0, 0.38, (6, channels, channels)),
            gate_weights=rng.normal(0.0, 0.22, (6, channels, channels)),
            proposal_bias=rng.normal(0.0, 0.06, channels),
            gate_bias=rng.normal(-0.9, 0.18, channels),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        sigma: float = 0.10,
        rate: float = 0.28,
    ) -> "DirectionalGenome":
        def m(value: np.ndarray) -> np.ndarray:
            mask = rng.random(value.shape) < rate
            changed = value + mask * rng.normal(0.0, sigma, value.shape)
            return np.clip(changed, -4.0, 4.0)

        return DirectionalGenome(
            proposal_weights=m(self.proposal_weights),
            gate_weights=m(self.gate_weights),
            proposal_bias=m(self.proposal_bias),
            gate_bias=m(self.gate_bias),
        )

    def complexity(self) -> float:
        values = np.concatenate(
            [
                self.proposal_weights.ravel(),
                self.gate_weights.ravel(),
                self.proposal_bias.ravel(),
                self.gate_bias.ravel(),
            ]
        )
        return float(np.mean(np.abs(values) > 0.10))

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_weights": self.proposal_weights.tolist(),
            "gate_weights": self.gate_weights.tolist(),
            "proposal_bias": self.proposal_bias.tolist(),
            "gate_bias": self.gate_bias.tolist(),
            "channels": self.channels,
            "complexity": self.complexity(),
        }


class DirectionalSubstrate:
    def __init__(self, genome: DirectionalGenome, height: int, width: int):
        if height < 2 or width < 2:
            raise ValueError("height and width must be at least 2")
        self.genome = genome
        self.state = np.zeros((height, width, genome.channels), dtype=np.float64)

    def reset(self, state: np.ndarray | None = None) -> None:
        if state is None:
            self.state.fill(0.0)
            return
        if state.shape != self.state.shape:
            raise ValueError(f"state shape must be {self.state.shape}")
        self.state = np.asarray(state, dtype=np.float64).copy()

    def step(self, external_input: np.ndarray | None = None) -> np.ndarray:
        if external_input is None:
            external_input = np.zeros_like(self.state)
        if external_input.shape != self.state.shape:
            raise ValueError("external input must match state")
        features = (
            self.state,
            _shift_fixed(self.state, 1, 0),
            _shift_fixed(self.state, -1, 0),
            _shift_fixed(self.state, 0, 1),
            _shift_fixed(self.state, 0, -1),
            external_input,
        )
        proposal_pre = np.zeros_like(self.state) + self.genome.proposal_bias
        gate_pre = np.zeros_like(self.state) + self.genome.gate_bias
        for feature, proposal_w, gate_w in zip(
            features, self.genome.proposal_weights, self.genome.gate_weights
        ):
            proposal_pre += feature @ proposal_w.T
            gate_pre += feature @ gate_w.T
        proposal = np.tanh(proposal_pre)
        gate = _sigmoid(gate_pre)
        self.state = np.clip(self.state + gate * (proposal - self.state), -1.0, 1.0)
        return self.state


@dataclass(frozen=True)
class RelayEvaluation:
    score: float
    negative_destination: float
    positive_destination: float
    early_leakage: float


def evaluate_relay(
    genome: DirectionalGenome,
    width: int,
    height: int = 6,
    amplitude: float = 0.9,
) -> RelayEvaluation:
    """Balanced bounded relay benchmark using both signs and no wraparound."""
    if width < 2:
        raise ValueError("width must be at least 2")
    destinations: list[float] = []
    sign_scores: list[float] = []
    early_leakages: list[float] = []
    for message in (-amplitude, amplitude):
        world = DirectionalSubstrate(genome, height, width)
        initial = np.zeros_like(world.state)
        initial[:, 0, 0] = message
        if genome.channels > 3:
            initial[:, :, 3] = -0.8
        world.reset(initial)
        external = np.zeros_like(world.state)
        if genome.channels > 3:
            external[:, :, 3] = -0.8

        for _ in range(max(0, width - 2)):
            world.step(external)
        early = abs(float(np.mean(world.state[:, -1, 0])))
        world.step(external)
        destination = float(np.mean(world.state[:, -1, 0]))

        error = (destination - message) ** 2
        alignment = math.exp(-4.0 * error)
        energy = float(np.mean(np.abs(world.state[:, :, 0])))
        score = float(
            np.clip(
                0.97 * alignment + 0.03 * (1.0 - energy) - 0.20 * early,
                0.0,
                1.0,
            )
        )
        destinations.append(destination)
        sign_scores.append(score)
        early_leakages.append(early)
    return RelayEvaluation(
        score=float(min(sign_scores)),
        negative_destination=destinations[0],
        positive_destination=destinations[1],
        early_leakage=float(max(early_leakages)),
    )


@dataclass(frozen=True)
class RelayCurriculumConfig:
    widths: tuple[int, ...] = (2, 3, 4, 6, 8, 12, 16)
    generations_per_stage: tuple[int, ...] = (20, 20, 24, 28, 32, 38, 46)
    population_size: int = 96
    elite_count: int = 16
    channels: int = 4
    mutation_sigma: float = 0.16
    mutation_rate: float = 0.35
    random_restart_rate: float = 0.03
    seed: int = 7

    def validate(self) -> None:
        if len(self.widths) != len(self.generations_per_stage):
            raise ValueError("width and generation schedules must match")
        if self.population_size < 8:
            raise ValueError("population must be at least 8")
        if not 2 <= self.elite_count < self.population_size:
            raise ValueError("invalid elite count")


@dataclass
class RelayDiscoveryResult:
    best_genome: DirectionalGenome
    training_score: float
    hidden_scores: dict[str, float]
    hidden_details: dict[str, dict[str, float]]
    stage_history: list[dict[str, float]]
    random_baseline: float
    identity_baseline: float

    @property
    def strict_transfer_score(self) -> float:
        return float(min(self.hidden_scores.values()))

    @property
    def project_breakthrough(self) -> bool:
        return (
            self.training_score >= 0.97
            and self.strict_transfer_score >= 0.95
            and self.strict_transfer_score >= self.random_baseline + 0.40
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "project_breakthrough" if self.project_breakthrough else "not_yet",
            "claim_scope": "strict directional relay primitive; not a world-level computing breakthrough",
            "training_score": self.training_score,
            "strict_transfer_score": self.strict_transfer_score,
            "hidden_scores": self.hidden_scores,
            "hidden_details": self.hidden_details,
            "random_baseline": self.random_baseline,
            "identity_baseline": self.identity_baseline,
            "stage_history": self.stage_history,
            "best_genome": self.best_genome.to_dict(),
        }


def _identity_genome(channels: int = 4) -> DirectionalGenome:
    proposal = np.zeros((6, channels, channels), dtype=np.float64)
    proposal[0] = np.eye(channels)
    gate = np.zeros_like(proposal)
    return DirectionalGenome(
        proposal_weights=proposal,
        gate_weights=gate,
        proposal_bias=np.zeros(channels),
        gate_bias=np.full(channels, 2.0),
    )


def discover_relay(config: RelayCurriculumConfig | None = None) -> RelayDiscoveryResult:
    config = config or RelayCurriculumConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    population = [
        DirectionalGenome.random(rng, config.channels)
        for _ in range(config.population_size)
    ]
    global_best: tuple[float, DirectionalGenome] | None = None
    history: list[dict[str, float]] = []

    for stage_index, (width, generations) in enumerate(
        zip(config.widths, config.generations_per_stage)
    ):
        for generation in range(generations):
            ranked = sorted(
                (
                    (evaluate_relay(genome, width).score, genome)
                    for genome in population
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            if global_best is None or ranked[0][0] > global_best[0]:
                global_best = ranked[0]
            if generation == 0 or generation == generations - 1 or generation % 10 == 0:
                history.append(
                    {
                        "stage": float(stage_index),
                        "width": float(width),
                        "generation": float(generation),
                        "best": float(ranked[0][0]),
                        "median": float(np.median([item[0] for item in ranked])),
                    }
                )
            elites = [item[1] for item in ranked[: config.elite_count]]
            next_population = list(elites)
            progress = generation / max(1, generations - 1)
            sigma = config.mutation_sigma * (1.0 - 0.75 * progress) + 0.01
            while len(next_population) < config.population_size:
                if rng.random() < config.random_restart_rate:
                    next_population.append(
                        DirectionalGenome.random(rng, config.channels)
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

        ranked = sorted(
            (
                (evaluate_relay(genome, width).score, genome)
                for genome in population
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        global_best = ranked[0]

    assert global_best is not None
    best = global_best[1]
    training_score = evaluate_relay(best, config.widths[-1]).score
    hidden_widths = (18, 22, 30)
    hidden_eval = {
        str(width): evaluate_relay(best, width)
        for width in hidden_widths
    }

    baseline_rng = np.random.default_rng(config.seed + 10_000)
    random_candidates = [
        DirectionalGenome.random(baseline_rng, config.channels)
        for _ in range(128)
    ]
    random_baseline = max(
        min(evaluate_relay(candidate, width).score for width in hidden_widths)
        for candidate in random_candidates
    )
    identity = _identity_genome(config.channels)
    identity_baseline = min(
        evaluate_relay(identity, width).score
        for width in hidden_widths
    )

    return RelayDiscoveryResult(
        best_genome=best,
        training_score=float(training_score),
        hidden_scores={key: value.score for key, value in hidden_eval.items()},
        hidden_details={
            key: {
                "negative_destination": value.negative_destination,
                "positive_destination": value.positive_destination,
                "early_leakage": value.early_leakage,
            }
            for key, value in hidden_eval.items()
        },
        stage_history=history,
        random_baseline=float(random_baseline),
        identity_baseline=float(identity_baseline),
    )


def run_relay_to_json(
    output: str,
    seed: int = 7,
    population: int = 96,
    scale: float = 1.0,
) -> dict[str, object]:
    base_generations = np.asarray((20, 20, 24, 28, 32, 38, 46), dtype=int)
    generations = tuple(
        max(3, int(round(value * scale)))
        for value in base_generations
    )
    result = discover_relay(
        RelayCurriculumConfig(
            population_size=population,
            elite_count=max(4, min(16, population // 6)),
            generations_per_stage=generations,
            seed=seed,
        )
    )
    payload = result.to_dict()
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="relay-v2.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    payload = run_relay_to_json(
        args.output,
        args.seed,
        args.population,
        args.scale,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "training_score": payload["training_score"],
                "strict_transfer_score": payload["strict_transfer_score"],
                "random_baseline": payload["random_baseline"],
                "identity_baseline": payload["identity_baseline"],
            },
            indent=2,
        )
    )
