from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome import Genome


@dataclass
class CellularSubstrate:
    """Toroidal 2-D universe executing one local rule at every cell."""

    genome: Genome
    height: int = 16
    width: int = 16

    def __post_init__(self) -> None:
        if self.height < 3 or self.width < 3:
            raise ValueError("height and width must be at least 3")
        self.state = np.zeros((self.height, self.width, self.genome.channels), dtype=np.float64)

    def reset(self, state: np.ndarray | None = None) -> None:
        if state is None:
            self.state.fill(0.0)
            return
        expected = (self.height, self.width, self.genome.channels)
        if state.shape != expected:
            raise ValueError(f"state shape must be {expected}, got {state.shape}")
        self.state = np.asarray(state, dtype=np.float64).copy()

    def _neighbour_features(self) -> tuple[np.ndarray, np.ndarray]:
        neighbours = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbours.append(np.roll(np.roll(self.state, dy, axis=0), dx, axis=1))
        stacked = np.stack(neighbours, axis=0)
        return stacked.mean(axis=0), stacked.max(axis=0)

    def step(self, external_input: np.ndarray | None = None) -> np.ndarray:
        if external_input is None:
            external_input = np.zeros_like(self.state)
        if external_input.shape != self.state.shape:
            raise ValueError("external_input must match state shape")

        neighbour_mean, neighbour_max = self._neighbour_features()
        proposal = (
            self.state @ self.genome.self_weights.T
            + neighbour_mean @ self.genome.mean_weights.T
            + neighbour_max @ self.genome.max_weights.T
            + external_input @ self.genome.input_weights.T
            + self.genome.bias
        )
        proposal = np.tanh(proposal)
        self.state = np.clip(
            (1.0 - self.genome.leak) * self.state + self.genome.leak * proposal,
            -1.0,
            1.0,
        )
        return self.state

    def run(self, steps: int, input_schedule: dict[int, np.ndarray] | None = None) -> np.ndarray:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        schedule = input_schedule or {}
        for tick in range(steps):
            self.step(schedule.get(tick))
        return self.state

    def damage(self, fraction: float, rng: np.random.Generator) -> np.ndarray:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        mask = rng.random((self.height, self.width)) < fraction
        self.state[mask, :] = 0.0
        return mask

    def inject_noise(self, sigma: float, rng: np.random.Generator) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.state = np.clip(self.state + rng.normal(0.0, sigma, self.state.shape), -1.0, 1.0)
