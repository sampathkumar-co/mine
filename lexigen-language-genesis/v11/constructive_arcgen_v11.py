from __future__ import annotations

import random
from typing import Any


def _valid(width: int, height: int, colours: list[int]) -> bool:
    for row in range(height):
        if all(colours[row * width + col] == 0 for col in range(width)):
            return False
    for col in range(width):
        if all(colours[row * width + col] == 0 for row in range(height)):
            return False
    for row in range(1, height):
        for col in range(1, width):
            square = {
                colours[row * width + col],
                colours[row * width + col - 1],
                colours[(row - 1) * width + col],
                colours[(row - 1) * width + col - 1],
            }
            if len(square) == 1:
                return False
    return True


def parameters(seed: int) -> dict[str, Any]:
    for attempt in range(256):
        rng = random.Random((seed << 8) + attempt)
        width = rng.randint(1, 4)
        height = rng.randint(1, 4)
        palette_size = rng.randint(3, 6)
        palette = rng.sample(range(1, 10), palette_size)
        colours = [rng.choice(palette + [0]) for _ in range(width * height)]
        if _valid(width, height, colours):
            return {"width": width, "height": height, "colors": colours}
    raise RuntimeError(f"could not construct v11 task parameters for seed {seed}")


def generate(task_module: Any, seed: int) -> dict[str, Any]:
    item = task_module.generate(**parameters(seed))
    if item.get("input") is None or item.get("output") is None:
        raise RuntimeError(f"official renderer rejected v11 seed {seed}")
    return item
