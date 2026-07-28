from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

Point = tuple[int, int]


@dataclass(frozen=True)
class Placement:
    row: int
    col: int
    wide: int
    tall: int
    angle: int
    source: frozenset[Point]
    destination: frozenset[Point]


def _delta(angle: int) -> tuple[int, int]:
    return ((-1, -1), (-1, 1), (1, -1), (1, 1))[angle]


def _options(size: int, wide: int, tall: int, angle: int) -> list[Placement]:
    dr, dc = _delta(angle)
    result = []
    for row in range(size - tall + 1):
        for col in range(size - wide + 1):
            source = frozenset(
                (r, c)
                for r in range(row, row + tall)
                for c in range(col, col + wide)
            )
            destination = frozenset((r + dr, c + dc) for r, c in source)
            if any(not (0 <= r < size and 0 <= c < size) for r, c in destination):
                continue
            result.append(Placement(row, col, wide, tall, angle, source, destination))
    return result


def parameters(seed: int) -> dict[str, Any]:
    for attempt in range(128):
        rng = random.Random((seed << 8) + attempt)
        size = rng.choice((10, 11))
        count = rng.randint(2, 4)
        wides = [rng.randint(2, 5) for _ in range(count)]
        talls = [rng.randint(2, 5) for _ in range(count)]
        angles = [rng.randrange(4) for _ in range(count)]
        colours = rng.sample([1, 2, 3, 4, 5, 6, 9], count)
        option_lists = []
        for wide, tall, angle in zip(wides, talls, angles):
            options = _options(size, wide, tall, angle)
            rng.shuffle(options)
            option_lists.append(options)
        order = sorted(range(count), key=lambda index: len(option_lists[index]))
        selected: dict[int, Placement] = {}

        def search(position: int, sources: set[Point], destinations: set[Point]) -> bool:
            if position == len(order):
                return True
            index = order[position]
            for placement in option_lists[index]:
                if placement.source & sources or placement.destination & destinations:
                    continue
                selected[index] = placement
                if search(
                    position + 1,
                    sources | set(placement.source),
                    destinations | set(placement.destination),
                ):
                    return True
                selected.pop(index, None)
            return False

        if not search(0, set(), set()):
            continue
        ordered = [selected[index] for index in range(count)]
        return {
            "size": size,
            "wides": wides,
            "talls": talls,
            "brows": [placement.row for placement in ordered],
            "bcols": [placement.col for placement in ordered],
            "colors": colours,
            "angles": angles,
        }
    raise RuntimeError(f"could not construct v9 ARC-GEN parameters for seed {seed}")


def generate(task_module: Any, seed: int) -> dict[str, Any]:
    item = task_module.generate(**parameters(seed))
    if item.get("input") is None or item.get("output") is None:
        raise RuntimeError(f"official renderer rejected v9 constructive seed {seed}")
    return item
