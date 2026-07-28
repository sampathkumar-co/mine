from __future__ import annotations

import heapq
import random

from common import Incidence, coverage_counts


def greedy_cover(incidence: Incidence, seed: int, randomized: bool, limit: int) -> list[int]:
    rng = random.Random(seed)
    uncovered = bytearray(b"\x01") * len(incidence.tsets)
    remaining = len(incidence.tsets)
    gains = [len(row) for row in incidence.cover_by_block]
    heap = [(-gain, rng.random(), i) for i, gain in enumerate(gains)]
    heapq.heapify(heap)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    while remaining and len(chosen) < limit:
        width = 12 if randomized else 1
        valid = []
        while heap and len(valid) < width:
            neg, tie, index = heapq.heappop(heap)
            if index not in chosen_set and -neg == gains[index]:
                valid.append((-neg, tie, index))
        if not valid or valid[0][0] <= 0:
            break
        if randomized:
            weights = [max(1, item[0]) ** 4 for item in valid]
            draw = rng.randrange(sum(weights))
            pick = valid[-1][2]
            for weight, item in zip(weights, valid):
                if draw < weight:
                    pick = item[2]
                    break
                draw -= weight
        else:
            pick = valid[0][2]
        for gain, tie, index in valid:
            if index != pick:
                heapq.heappush(heap, (-gain, tie, index))
        chosen.append(pick)
        chosen_set.add(pick)
        newly = []
        for subset in incidence.cover_by_block[pick]:
            if uncovered[subset]:
                uncovered[subset] = 0
                remaining -= 1
                newly.append(subset)
        for subset in newly:
            for index in incidence.blocks_by_t[subset]:
                if index not in chosen_set and gains[index] > 0:
                    gains[index] -= 1
                    heapq.heappush(heap, (-gains[index], rng.random(), index))
    return chosen if remaining == 0 else []


def prune(incidence: Incidence, selected: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    selected = list(dict.fromkeys(selected))
    counts = coverage_counts(incidence, selected)
    changed = True
    while changed:
        changed = False
        order = list(selected)
        rng.shuffle(order)
        order.sort(key=lambda b: sum(counts[s] == 1 for s in incidence.cover_by_block[b]))
        for block in order:
            if block not in selected:
                continue
            if all(counts[s] >= 2 for s in incidence.cover_by_block[block]):
                selected.remove(block)
                for s in incidence.cover_by_block[block]:
                    counts[s] -= 1
                changed = True
    return selected


def normalize_budget(
    incidence: Incidence, selected: list[int], budget: int, rng: random.Random
) -> list[int]:
    selected = list(dict.fromkeys(selected))
    counts = coverage_counts(incidence, selected)
    while len(selected) > budget:
        remove = min(
            selected,
            key=lambda b: (
                sum(counts[s] == 1 for s in incidence.cover_by_block[b]),
                rng.random(),
            ),
        )
        selected.remove(remove)
        for s in incidence.cover_by_block[remove]:
            counts[s] -= 1
    selected_set = set(selected)
    while len(selected) < budget:
        candidate = rng.randrange(len(incidence.blocks))
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
    return selected
