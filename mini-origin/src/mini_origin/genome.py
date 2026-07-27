from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Genome:
    """Local update law shared by every cell in a substrate."""

    self_weights: np.ndarray
    mean_weights: np.ndarray
    max_weights: np.ndarray
    input_weights: np.ndarray
    bias: np.ndarray
    leak: float

    @property
    def channels(self) -> int:
        return int(self.bias.shape[0])

    @classmethod
    def random(cls, rng: np.random.Generator, channels: int = 3) -> "Genome":
        scale = 0.65
        return cls(
            self_weights=rng.normal(0.0, scale, (channels, channels)),
            mean_weights=rng.normal(0.0, scale, (channels, channels)),
            max_weights=rng.normal(0.0, scale, (channels, channels)),
            input_weights=rng.normal(0.0, scale, (channels, channels)),
            bias=rng.normal(0.0, 0.15, channels),
            leak=float(rng.uniform(0.15, 0.95)),
        )

    def mutate(self, rng: np.random.Generator, sigma: float = 0.12, rate: float = 0.18) -> "Genome":
        def mutate_array(value: np.ndarray) -> np.ndarray:
            mask = rng.random(value.shape) < rate
            return np.clip(value + mask * rng.normal(0.0, sigma, value.shape), -3.0, 3.0)

        leak = float(np.clip(self.leak + (rng.normal(0.0, sigma) if rng.random() < rate else 0.0), 0.02, 1.0))
        return Genome(
            self_weights=mutate_array(self.self_weights),
            mean_weights=mutate_array(self.mean_weights),
            max_weights=mutate_array(self.max_weights),
            input_weights=mutate_array(self.input_weights),
            bias=mutate_array(self.bias),
            leak=leak,
        )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        if self.channels != other.channels:
            raise ValueError("genomes must have the same channel count")

        def mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            mask = rng.random(a.shape) < 0.5
            return np.where(mask, a, b)

        return Genome(
            self_weights=mix(self.self_weights, other.self_weights),
            mean_weights=mix(self.mean_weights, other.mean_weights),
            max_weights=mix(self.max_weights, other.max_weights),
            input_weights=mix(self.input_weights, other.input_weights),
            bias=mix(self.bias, other.bias),
            leak=self.leak if rng.random() < 0.5 else other.leak,
        )

    def complexity(self) -> float:
        values = np.concatenate(
            [
                self.self_weights.ravel(),
                self.mean_weights.ravel(),
                self.max_weights.ravel(),
                self.input_weights.ravel(),
                self.bias.ravel(),
            ]
        )
        return float(np.mean(np.abs(values) > 0.10))

    def to_dict(self) -> dict[str, object]:
        return {
            "self_weights": self.self_weights.tolist(),
            "mean_weights": self.mean_weights.tolist(),
            "max_weights": self.max_weights.tolist(),
            "input_weights": self.input_weights.tolist(),
            "bias": self.bias.tolist(),
            "leak": self.leak,
            "channels": self.channels,
            "complexity": self.complexity(),
        }
