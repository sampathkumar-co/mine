from __future__ import annotations

import heapq
import random

from common import Incidence


def greedy_cover(incidence: Incidence, seed: int, randomized: bool, step_limit: int) -> list[int]:
    rng = random.Random(seed)
    uncovered = bytearray(b"\x01") * len(incidence.tsets)
    remaining = len(incidence.tsets)
    gains = [len(covered) for covered in incidence.cover_by_block]
    heap = [(-gain, rng.random(), index) for index, gain in enumerate(gains)]
    heapq.heapify(heap)
    selected: list[int] = []
    selected_set: set[int] = set()

    while remaining and len(selected) < step_limit:
        width = 10 if randomized else 1
        valid: list[tuple[int, float, int]] = []
        while heap and len(valid) < width:
            neg_gain, tie, block_index = heapq.heappop(heap)
            if block_index not in selected_set and -neg_gain == gains[block_index]:
                valid.append((-neg_gain, tie, block_index))
        if not valid or valid[0][0] <= 0:
            break
        if randomized:
            weights = [max(1, item[0]) ** 3 for item in valid]
            draw = rng.randrange(sum(weights))
            chosen = valid[-1][2]
            for weight, item in zip(weights, valid):
                if draw < weight:
                    chosen = item[2]
                    break
                draw -= weight
        else:
            chosen = valid[0][2]
        for gain, tie, block_index in valid:
            if block_index != chosen:
                heapq.heappush(heap, (-gain, tie, block_index))
        selected.append(chosen)
        selected_set.add(chosen)
        newly_covered: list[int] = []
        for subset_index in incidence.cover_by_block[chosen]:
            if uncovered[subset_index]:
                uncovered[subset_index] = 0
                remaining -= 1
                newly_covered.append(subset_index)
        for subset_index in newly_covered:
            for block_index in incidence.blocks_by_t[subset_index]:
                if block_index not in selected_set and gains[block_index] > 0:
                    gains[block_index] -= 1
                    heapq.heappush(heap, (-gains[block_index], rng.random(), block_index))
    return selected if remaining == 0 else []


def coverage_counts(incidence: Incidence, selected: list[int]) -> list[int]:
    counts = [0] * len(incidence.tsets)
    for block_index in selected:
        for subset_index in incidence.cover_by_block[block_index]:
            counts[subset_index] += 1
    return counts


def prune_redundant(incidence: Incidence, selected: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    selected = list(dict.fromkeys(selected))
    counts = coverage_counts(incidence, selected)
    changed = True
    while changed:
        changed = False
        order = list(selected)
        rng.shuffle(order)
        order.sort(key=lambda b: sum(counts[s] == 1 for s in incidence.cover_by_block[b]))
        for block_index in order:
            if block_index not in selected:
                continue
            if all(counts[s] >= 2 for s in incidence.cover_by_block[block_index]):
                selected.remove(block_index)
                for subset_index in incidence.cover_by_block[block_index]:
                    counts[subset_index] -= 1
                changed = True
    return selected


def normalize_to_budget(
    incidence: Incidence, selected: list[int], budget: int, rng: random.Random
) -> list[int]:
    selected = list(dict.fromkeys(selected))
    if len(selected) > budget:
        counts = coverage_counts(incidence, selected)
        while len(selected) > budget:
            removable = min(
                selected,
                key=lambda b: (
                    sum(counts[s] == 1 for s in incidence.cover_by_block[b]),
                    rng.random(),
                ),
            )
            selected.remove(removable)
            for subset_index in incidence.cover_by_block[removable]:
                counts[subset_index] -= 1
    selected_set = set(selected)
    while len(selected) < budget:
        candidate = rng.randrange(len(incidence.blocks))
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
    return selected
