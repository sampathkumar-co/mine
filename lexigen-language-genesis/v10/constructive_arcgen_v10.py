from __future__ import annotations

import random
from typing import Any

Point = tuple[int, int]
Vector = tuple[int, int]


def _inside(point: Point, size: int) -> bool:
    return 0 <= point[0] < size and 0 <= point[1] < size


def _simulate(size: int, obstacles: set[Point]) -> tuple[bool, int]:
    row, col = 1, 2
    rdir, cdir = 0, 1
    bounces = 0
    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(size * size * 8):
        if not _inside((row, col), size):
            return 3 <= bounces <= 6, bounces
        state = row, col, rdir, cdir
        if state in seen:
            return False, bounces
        seen.add(state)
        if (row + rdir, col + cdir) in obstacles:
            bounces += 1
            if bounces > 10:
                return False, bounces
            if abs(rdir):
                increase = (row, col - 1) in obstacles
                decrease = (row, col + 1) in obstacles
            else:
                increase = (row - 1, col) in obstacles
                decrease = (row + 1, col) in obstacles
            if increase == decrease:
                return False, bounces
            if decrease:
                rdir, cdir = (0 if abs(rdir) else -1), (0 if abs(cdir) else -1)
            if increase:
                rdir, cdir = (0 if abs(rdir) else 1), (0 if abs(cdir) else 1)
        row, col = row + rdir, col + cdir
    return False, bounces


def parameters(seed: int) -> dict[str, Any]:
    for attempt in range(20_000):
        rng = random.Random((seed << 16) + attempt)
        size = rng.randint(7, 10)
        probability = rng.uniform(0.14, 0.32)
        obstacles = {
            (row, col)
            for row in range(size)
            for col in range(size)
            if rng.random() < probability
        }
        obstacles.difference_update({(1, 0), (1, 1), (1, 2)})
        valid, _ = _simulate(size, obstacles)
        if not valid:
            continue
        colours = [int((row, col) in obstacles) for row in range(size) for col in range(size)]
        return {
            "size": size,
            "flip": rng.randrange(2),
            "flop": rng.randrange(2),
            "xpose": rng.randrange(2),
            "colors": colours,
        }
    raise RuntimeError(f"could not construct v10 ARC-GEN parameters for seed {seed}")


def generate(task_module: Any, seed: int) -> dict[str, Any]:
    item = task_module.generate(**parameters(seed))
    if item.get("input") is None or item.get("output") is None:
        raise RuntimeError(f"official renderer rejected v10 constructive seed {seed}")
    return item
