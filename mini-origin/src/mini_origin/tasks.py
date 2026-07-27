from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .genome import Genome
from .substrate import CellularSubstrate


class Task(Protocol):
    name: str

    def evaluate(self, genome: Genome, seed: int) -> float: ...


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    error = float(np.mean((a - b) ** 2))
    return float(np.exp(-2.5 * error))


@dataclass(frozen=True)
class MemoryTask:
    name: str = "memory"
    size: int = 14
    steps: int = 10

    def evaluate(self, genome: Genome, seed: int) -> float:
        rng = np.random.default_rng(seed)
        world = CellularSubstrate(genome, self.size, self.size)
        target = np.zeros_like(world.state)
        pattern = rng.choice((-0.8, 0.8), size=(self.size, self.size))
        target[:, :, 0] = pattern
        world.reset(target)
        for _ in range(self.steps):
            world.inject_noise(0.025, rng)
            world.step()
        return _similarity(world.state[:, :, 0], pattern)


@dataclass(frozen=True)
class RelayTask:
    name: str = "relay"
    height: int = 10
    width: int = 16
    steps: int = 14

    def evaluate(self, genome: Genome, seed: int) -> float:
        rng = np.random.default_rng(seed)
        world = CellularSubstrate(genome, self.height, self.width)
        message = float(rng.choice((-0.9, 0.9)))
        initial = np.zeros_like(world.state)
        initial[:, 0, 0] = message
        world.reset(initial)
        world.run(self.steps)
        destination = float(np.mean(world.state[:, -1, 0]))
        alignment = 1.0 - min(abs(destination - message) / 2.0, 1.0)
        energy = float(np.mean(np.abs(world.state[:, :, 0])))
        return float(np.clip(0.88 * alignment + 0.12 * (1.0 - energy), 0.0, 1.0))


@dataclass(frozen=True)
class RepairTask:
    name: str = "repair"
    size: int = 14
    settle_steps: int = 5
    recovery_steps: int = 7
    damage_fraction: float = 0.22

    def evaluate(self, genome: Genome, seed: int) -> float:
        rng = np.random.default_rng(seed)
        initial = np.zeros((self.size, self.size, genome.channels), dtype=np.float64)
        yy, xx = np.mgrid[: self.size, : self.size]
        cy = (self.size - 1) / 2.0
        cx = (self.size - 1) / 2.0
        radius = self.size * 0.27
        initial[:, :, 0] = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2, 0.85, -0.45)
        if genome.channels > 1:
            initial[:, :, 1] = np.roll(initial[:, :, 0], 1, axis=0)

        reference = CellularSubstrate(genome, self.size, self.size)
        reference.reset(initial)
        reference.run(self.settle_steps + self.recovery_steps)
        target = reference.state.copy()

        damaged = CellularSubstrate(genome, self.size, self.size)
        damaged.reset(initial)
        damaged.run(self.settle_steps)
        mask = damaged.damage(self.damage_fraction, rng)
        post_damage_error = float(np.mean((damaged.state - target) ** 2))
        damaged.run(self.recovery_steps)
        recovered_error = float(np.mean((damaged.state - target) ** 2))

        if post_damage_error <= 1e-12:
            recovery_gain = 1.0
        else:
            recovery_gain = np.clip(1.0 - recovered_error / post_damage_error, 0.0, 1.0)
        damaged_region_similarity = _similarity(damaged.state[mask], target[mask]) if np.any(mask) else 1.0
        return float(0.55 * damaged_region_similarity + 0.45 * recovery_gain)


def default_tasks() -> list[Task]:
    return [MemoryTask(), RelayTask(), RepairTask()]
