from __future__ import annotations

import math
import random
import time

from common import Incidence, coverage_counts
from greedy import normalize_budget


def fixed_budget_repair(
    incidence: Incidence,
    initial: list[int],
    budget: int,
    seed: int,
    deadline: float,
) -> tuple[list[int], dict[str, object]]:
    rng = random.Random(seed)
    selected = normalize_budget(incidence, initial, budget, rng)
    selected_set = set(selected)
    counts = coverage_counts(incidence, selected)
    uncovered = sum(x == 0 for x in counts)
    best = list(selected)
    best_uncovered = uncovered
    iterations = accepted = 0
    stagnation = 0

    while time.monotonic() < deadline and best_uncovered:
        iterations += 1
        uncovered_indices = [i for i, c in enumerate(counts) if c == 0]
        if not uncovered_indices:
            best = list(selected)
            best_uncovered = 0
            break
        anchor = min(
            rng.sample(uncovered_indices, min(8, len(uncovered_indices))),
            key=lambda s: len(incidence.blocks_by_t[s]),
        )
        candidate_pool = list(incidence.blocks_by_t[anchor])
        rng.shuffle(candidate_pool)
        candidate_pool = candidate_pool[:64]
        candidate_pool.extend(rng.randrange(len(incidence.blocks)) for _ in range(12))

        remove_pool = sorted(
            selected,
            key=lambda b: (
                sum(counts[s] == 1 for s in incidence.cover_by_block[b]),
                -sum(counts[s] > 1 for s in incidence.cover_by_block[b]),
                rng.random(),
            ),
        )[: min(24, len(selected))]

        best_move = None
        for add in candidate_pool:
            if add in selected_set:
                continue
            add_set = set(incidence.cover_by_block[add])
            for remove in remove_pool:
                remove_set = set(incidence.cover_by_block[remove])
                score = uncovered
                for s in remove_set:
                    if counts[s] == 1 and s not in add_set:
                        score += 1
                for s in add_set:
                    if counts[s] == 0:
                        score -= 1
                critical_after = sum(
                    counts[s]
                    + (1 if s in add_set else 0)
                    - (1 if s in remove_set else 0)
                    == 1
                    for s in add_set
                )
                objective = (score, critical_after, rng.random())
                if best_move is None or objective < best_move[0]:
                    best_move = (objective, remove, add)
        if best_move is None:
            continue

        (score, _, _), remove, add = best_move
        temperature = max(0.03, 2.0 * (1.0 - min(iterations, 250_000) / 250_000.0))
        accept = score <= uncovered or rng.random() < math.exp(-(score - uncovered) / temperature)
        if accept:
            pos = selected.index(remove)
            selected[pos] = add
            selected_set.remove(remove)
            selected_set.add(add)
            for s in incidence.cover_by_block[remove]:
                counts[s] -= 1
            for s in incidence.cover_by_block[add]:
                counts[s] += 1
            uncovered = score
            accepted += 1
            if uncovered < best_uncovered:
                best = list(selected)
                best_uncovered = uncovered
                stagnation = 0
            else:
                stagnation += 1

        if stagnation > 10_000:
            for _ in range(min(4, len(selected))):
                remove = rng.choice(selected)
                add = rng.randrange(len(incidence.blocks))
                if add in selected_set:
                    continue
                pos = selected.index(remove)
                selected[pos] = add
                selected_set.remove(remove)
                selected_set.add(add)
                for s in incidence.cover_by_block[remove]:
                    counts[s] -= 1
                for s in incidence.cover_by_block[add]:
                    counts[s] += 1
            uncovered = sum(c == 0 for c in counts)
            stagnation = 0

    return best, {
        "iterations": iterations,
        "accepted_moves": accepted,
        "best_uncovered": best_uncovered,
    }
