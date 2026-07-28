from __future__ import annotations

import math
import random
import time

from common import Incidence
from greedy import coverage_counts, normalize_to_budget


def replacement_score(
    incidence: Incidence,
    counts: list[int],
    uncovered_count: int,
    remove_block: int,
    add_block: int,
) -> int:
    add_covered = set(incidence.cover_by_block[add_block])
    score = uncovered_count
    for subset_index in incidence.cover_by_block[remove_block]:
        if counts[subset_index] == 1 and subset_index not in add_covered:
            score += 1
    for subset_index in incidence.cover_by_block[add_block]:
        if counts[subset_index] == 0:
            score -= 1
    return score


def apply_replacement(
    incidence: Incidence,
    selected: list[int],
    selected_set: set[int],
    counts: list[int],
    remove_block: int,
    add_block: int,
) -> None:
    position = selected.index(remove_block)
    selected[position] = add_block
    selected_set.remove(remove_block)
    selected_set.add(add_block)
    for subset_index in incidence.cover_by_block[remove_block]:
        counts[subset_index] -= 1
    for subset_index in incidence.cover_by_block[add_block]:
        counts[subset_index] += 1


def stochastic_fixed_budget(
    incidence: Incidence,
    initial: list[int],
    budget: int,
    seed: int,
    deadline: float,
) -> tuple[list[int], dict[str, object]]:
    rng = random.Random(seed)
    selected = normalize_to_budget(incidence, initial, budget, rng)
    selected_set = set(selected)
    counts = coverage_counts(incidence, selected)
    uncovered_count = sum(count == 0 for count in counts)
    best = list(selected)
    best_uncovered = uncovered_count
    iterations = 0
    accepted = 0
    last_improvement = 0

    while time.monotonic() < deadline and best_uncovered > 0:
        iterations += 1
        uncovered = [index for index, count in enumerate(counts) if count == 0]
        if not uncovered:
            best = list(selected)
            best_uncovered = 0
            break
        anchor = rng.choice(uncovered)
        candidate_pool = list(incidence.blocks_by_t[anchor])
        rng.shuffle(candidate_pool)
        candidate_pool = candidate_pool[:48]
        candidate_pool.extend(rng.randrange(len(incidence.blocks)) for _ in range(8))

        remove_order = sorted(
            selected,
            key=lambda block: (
                sum(counts[s] == 1 for s in incidence.cover_by_block[block]),
                rng.random(),
            ),
        )[: min(16, len(selected))]

        best_move: tuple[int, int, int] | None = None
        for add_block in candidate_pool:
            if add_block in selected_set:
                continue
            for remove_block in remove_order:
                score = replacement_score(
                    incidence, counts, uncovered_count, remove_block, add_block
                )
                if best_move is None or score < best_move[0]:
                    best_move = (score, remove_block, add_block)
                    if score == 0:
                        break
            if best_move is not None and best_move[0] == 0:
                break
        if best_move is None:
            continue

        score, remove_block, add_block = best_move
        temperature = max(0.05, 2.5 * (1.0 - min(iterations, 200_000) / 200_000.0))
        accept = score <= uncovered_count
        if not accept:
            accept = rng.random() < math.exp(-(score - uncovered_count) / temperature)
        if accept:
            apply_replacement(
                incidence, selected, selected_set, counts, remove_block, add_block
            )
            uncovered_count = score
            accepted += 1
            if uncovered_count < best_uncovered:
                best_uncovered = uncovered_count
                best = list(selected)
                last_improvement = iterations

        if iterations - last_improvement > 12_000:
            for _ in range(min(3, len(selected))):
                remove_block = rng.choice(selected)
                add_block = rng.randrange(len(incidence.blocks))
                if add_block not in selected_set:
                    apply_replacement(
                        incidence, selected, selected_set, counts, remove_block, add_block
                    )
            uncovered_count = sum(count == 0 for count in counts)
            last_improvement = iterations

    return best, {
        "iterations": iterations,
        "accepted_moves": accepted,
        "best_uncovered": best_uncovered,
    }
