from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

Point = tuple[int, int]


@dataclass(frozen=True)
class LinePlacement:
    row: int
    col: int
    length: int
    angle: int
    points: frozenset[Point]
    forbidden: frozenset[Point]


def _placement(row: int, col: int, length: int, angle: int) -> LinePlacement:
    points = frozenset((row + index, col + index * angle) for index in range(length))
    forbidden: set[Point] = {
        (row - 1, col - angle),
        (row + length, col + angle * length),
    }
    for point_row, point_col in points:
        for delta_row, delta_col in (
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (0, 2),
            (0, -2),
            (2, 0),
            (-2, 0),
        ):
            forbidden.add((point_row + delta_row, point_col + delta_col))
    return LinePlacement(row, col, length, angle, points, frozenset(forbidden))


def _options(length: int, angle: int) -> list[LinePlacement]:
    placements = []
    for row in range(17 - length):
        if angle == 1:
            columns = range(17 - length)
        else:
            columns = range(length - 1, 16)
        for col in columns:
            placements.append(_placement(row, col, length, angle))
    return placements


def _compatible(candidate: LinePlacement, chosen: list[LinePlacement]) -> bool:
    for prior in chosen:
        if candidate.points & prior.points:
            return False
        if candidate.points & prior.forbidden:
            return False
        if prior.points & candidate.forbidden:
            return False
    return True


def _balanced_angles(count: int, rng: random.Random) -> list[int]:
    positive = count // 2
    negative = count // 2
    if count % 2:
        if rng.randrange(2):
            positive += 1
        else:
            negative += 1
    values = [1] * positive + [-1] * negative
    rng.shuffle(values)
    return values


def parameters(seed: int) -> dict[str, list[int]]:
    for attempt in range(128):
        rng = random.Random((seed << 8) + attempt)
        count = rng.randint(2, 7)
        lengths = rng.sample(range(2, 11), count)
        angles = _balanced_angles(count, rng)
        positive_total = sum(length for length, angle in zip(lengths, angles) if angle == 1)
        negative_total = sum(length for length, angle in zip(lengths, angles) if angle == -1)
        if positive_total == negative_total:
            continue

        requests = sorted(
            enumerate(zip(lengths, angles)),
            key=lambda item: (-item[1][0], item[0]),
        )
        option_lists: dict[int, list[LinePlacement]] = {}
        for index, (length, angle) in requests:
            values = _options(length, angle)
            rng.shuffle(values)
            option_lists[index] = values

        selected: dict[int, LinePlacement] = {}

        def search(position: int, chosen: list[LinePlacement]) -> bool:
            if position == len(requests):
                return True
            index, _ = requests[position]
            for candidate in option_lists[index]:
                if not _compatible(candidate, chosen):
                    continue
                selected[index] = candidate
                if search(position + 1, chosen + [candidate]):
                    return True
                selected.pop(index, None)
            return False

        if not search(0, []):
            continue
        ordered = [selected[index] for index in range(count)]
        return {
            "rows": [item.row for item in ordered],
            "cols": [item.col for item in ordered],
            "lengths": lengths,
            "angles": angles,
        }
    raise RuntimeError(f"could not construct a valid ARC-GEN instance for seed {seed}")


def generate(task_module: Any, seed: int) -> dict[str, Any]:
    values = parameters(seed)
    item = task_module.generate(**values)
    if item.get("input") is None or item.get("output") is None:
        raise RuntimeError(f"official renderer rejected constructive seed {seed}")
    return item
