from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path


@dataclass(frozen=True, order=True)
class Metric:
    diagnosed_mass: int
    expected_cost: int
    worst_cost: int


def canonical_partition(values: tuple[int, ...]) -> tuple[int, ...]:
    remap: dict[int, int] = {}
    output = []
    for value in values:
        if value not in remap:
            remap[value] = len(remap)
        output.append(remap[value])
    return tuple(output)


def all_partitions(size: int) -> tuple[tuple[int, ...], ...]:
    rows = {
        canonical_partition(values)
        for values in itertools.product(range(size), repeat=size)
    }
    return tuple(sorted(row for row in rows if len(set(row)) > 1))


def children(partition: tuple[int, ...], allowed: int) -> tuple[int, ...]:
    cells: dict[int, int] = {}
    for index, response in enumerate(partition):
        if allowed & (1 << index):
            cells[response] = cells.get(response, 0) | (1 << index)
    return tuple(sorted(cells.values()))


def pure(labels: tuple[int, ...], allowed: int) -> bool:
    return len({label for index, label in enumerate(labels) if allowed & (1 << index)}) <= 1


def subset_mass(masses: tuple[int, ...], allowed: int) -> int:
    return sum(mass for index, mass in enumerate(masses) if allowed & (1 << index))


def dominates(left: Metric, right: Metric) -> bool:
    return (
        left.diagnosed_mass >= right.diagnosed_mass
        and left.expected_cost <= right.expected_cost
        and left.worst_cost <= right.worst_cost
        and left != right
    )


def frontier(metrics: list[Metric]) -> tuple[Metric, ...]:
    unique = sorted(set(metrics))
    return tuple(
        metric for metric in unique
        if not any(dominates(other, metric) for other in unique)
    )


def partition_refines(fine: tuple[int, ...], coarse: tuple[int, ...]) -> bool:
    if fine == coarse:
        return False
    return all(any((cell & parent) == cell for parent in coarse) for cell in fine)


def pointwise_cost_dominates(
    left: tuple[int, ...], right: tuple[int, ...], allowed: int, strict: bool
) -> bool:
    pairs = [
        (left[index], right[index])
        for index in range(len(left)) if allowed & (1 << index)
    ]
    return all(a <= b for a, b in pairs) and (not strict or any(a < b for a, b in pairs))


def query_dominates(
    partitions: tuple[tuple[int, ...], ...],
    costs: tuple[tuple[int, ...], ...],
    allowed: int,
    left: int,
    right: int,
) -> tuple[bool, str | None]:
    left_partition = children(partitions[left], allowed)
    right_partition = children(partitions[right], allowed)
    if len(left_partition) <= 1 or len(right_partition) <= 1:
        return False, None
    if left_partition == right_partition:
        weak = pointwise_cost_dominates(costs[left], costs[right], allowed, False)
        strict = pointwise_cost_dominates(costs[left], costs[right], allowed, True)
        if strict or (weak and left < right):
            return True, "equivalent"
        return False, None
    if partition_refines(left_partition, right_partition) and pointwise_cost_dominates(
        costs[left], costs[right], allowed, True
    ):
        return True, "strict-refinement"
    return False, None


def retained_queries(
    partitions: tuple[tuple[int, ...], ...],
    costs: tuple[tuple[int, ...], ...],
    allowed: int,
    remaining: tuple[int, ...],
) -> tuple[tuple[int, ...], int, int]:
    informative = tuple(
        query for query in remaining if len(children(partitions[query], allowed)) > 1
    )
    kept = []
    equivalent_removed = 0
    refinement_removed = 0
    for query in informative:
        reasons = [
            reason
            for other in informative if other != query
            for dominated, reason in [query_dominates(partitions, costs, allowed, other, query)]
            if dominated
        ]
        if reasons:
            if "strict-refinement" in reasons:
                refinement_removed += 1
            else:
                equivalent_removed += 1
        else:
            kept.append(query)
    return tuple(kept), equivalent_removed, refinement_removed


def enumerate_frontier(
    partitions: tuple[tuple[int, ...], ...],
    labels: tuple[int, ...],
    masses: tuple[int, ...],
    costs: tuple[tuple[int, ...], ...],
    use_dominance: bool,
) -> tuple[Metric, ...]:
    full = (1 << len(labels)) - 1

    @lru_cache(maxsize=None)
    def solve(allowed: int, remaining: tuple[int, ...]) -> tuple[Metric, ...]:
        if pure(labels, allowed):
            return (Metric(subset_mass(masses, allowed), 0, 0),)
        active = (
            retained_queries(partitions, costs, allowed, remaining)[0]
            if use_dominance else tuple(
                query for query in remaining
                if len(children(partitions[query], allowed)) > 1
            )
        )
        candidates: list[Metric] = []
        for query in active:
            query_children = children(partitions[query], allowed)
            next_remaining = tuple(other for other in active if other != query)
            child_frontiers = [solve(child, next_remaining) for child in query_children]
            immediate = sum(
                masses[index] * costs[query][index]
                for index in range(len(labels)) if allowed & (1 << index)
            )
            for combination in itertools.product(*child_frontiers):
                candidates.append(Metric(
                    diagnosed_mass=sum(item.diagnosed_mass for item in combination),
                    expected_cost=immediate + sum(item.expected_cost for item in combination),
                    worst_cost=max(
                        costs[query][(child & -child).bit_length() - 1] + item.worst_cost
                        for child, item in zip(query_children, combination)
                    ),
                ))
        return frontier(candidates) if candidates else (Metric(0, 0, 0),)

    return solve(full, tuple(range(len(partitions))))


def canonical_binary_labels(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        values for values in itertools.product((0, 1), repeat=size)
        if values[0] == 0 and len(set(values)) > 1
    )


def profile(
    partitions: tuple[tuple[int, ...], ...], profile_index: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    size = len(partitions[0])
    masses = tuple(1 + ((index + profile_index) % 2) for index in range(size))
    rows = []
    max_cells = max(len(set(partition)) for partition in partitions)
    for query, partition in enumerate(partitions):
        if profile_index == 0:
            row = tuple(1 for _ in range(size))
        elif profile_index == 1:
            base = 1 + max_cells - len(set(partition))
            row = tuple(base for _ in range(size))
        elif profile_index == 2:
            base = 1 + max_cells - len(set(partition))
            row = tuple(base + ((query + response) % 2) for response in partition)
        else:
            row = tuple(
                1 + hashlib.sha256(
                    f"v65:{query}:{index}:{partition}".encode("utf-8")
                ).digest()[0] % 3
                for index in range(size)
            )
        rows.append(row)
    return masses, tuple(rows)


def counterexamples() -> dict[str, object]:
    fine = (0, 1, 2)
    coarse = (0, 0, 1)
    equal_costs = ((1, 1, 1), (1, 1, 1))
    unsafe_costs = ((2, 2, 2), (1, 1, 1))
    allowed = 0b111
    strict_required = not query_dominates((coarse, fine), equal_costs, allowed, 1, 0)[0]
    cost_required = not query_dominates((coarse, fine), unsafe_costs, allowed, 1, 0)[0]
    positive = query_dominates(
        (coarse, fine), ((2, 2, 2), (1, 1, 1)), allowed, 1, 0
    ) == (True, "strict-refinement")
    return {
        "passed": strict_required and cost_required and positive,
        "strict_cost_improvement_required": strict_required,
        "pointwise_no_worse_cost_required": cost_required,
        "positive_refinement_example": positive,
        "meaning": (
            "Strict refinement is pruned only when its immediate cost is pointwise "
            "no worse and strictly better somewhere; equal-cost strict refinement is "
            "retained to preserve the planner's deterministic root-query tie-break."
        ),
    }


def run() -> dict[str, object]:
    case_count = 0
    match_count = 0
    mismatches = []
    cases_with_refinement = 0
    refinement_removed = 0
    equivalent_removed = 0
    size_summaries = []
    for size, query_count in ((3, 4), (4, 3)):
        signatures = all_partitions(size)
        labels_set = canonical_binary_labels(size)
        local_cases = 0
        local_matches = 0
        for chosen in itertools.combinations_with_replacement(signatures, query_count):
            partitions = tuple(chosen)
            for labels in labels_set:
                for profile_index in range(4):
                    masses, costs = profile(partitions, profile_index)
                    plain = enumerate_frontier(partitions, labels, masses, costs, False)
                    reduced = enumerate_frontier(partitions, labels, masses, costs, True)
                    case_count += 1
                    local_cases += 1
                    if plain == reduced:
                        match_count += 1
                        local_matches += 1
                    elif len(mismatches) < 20:
                        mismatches.append({
                            "size": size,
                            "partitions": partitions,
                            "labels": labels,
                            "profile": profile_index,
                            "plain": [item.__dict__ for item in plain],
                            "reduced": [item.__dict__ for item in reduced],
                        })
                    _, equivalent, refinement = retained_queries(
                        partitions, costs, (1 << size) - 1, tuple(range(query_count))
                    )
                    equivalent_removed += equivalent
                    refinement_removed += refinement
                    cases_with_refinement += int(refinement > 0)
        size_summaries.append({
            "hypotheses": size,
            "queries": query_count,
            "partition_signatures": len(signatures),
            "labelings": len(labels_set),
            "cases": local_cases,
            "frontier_matches": local_matches,
        })
    examples = counterexamples()
    gate = (
        examples["passed"]
        and case_count >= 15_000
        and match_count == case_count
        and not mismatches
        and cases_with_refinement >= 500
        and refinement_removed >= 500
    )
    protocol = {
        "finite_models": [[3, 4], [4, 3]],
        "oracle": "direct complete-tree nondominated-frontier enumeration",
        "dominance_rule": (
            "same-partition response-cost Pareto dominance plus strict local partition "
            "refinement with pointwise no-worse cost and strict improvement somewhere"
        ),
        "mass_profiles": 4,
        "cost_values": [1, 2, 3],
    }
    return {
        "status": "refinement_dominance_certificate_pass" if gate else "rejected",
        "development_gate": gate,
        "claim_scope": (
            "Exhaustive finite-model evidence for strict descendant-local refinement "
            "dominance under positive masses and static response-dependent costs. This "
            "is internal machine evidence, not a proof-assistant theorem or external review."
        ),
        "protocol": protocol,
        "protocol_digest": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "case_count": case_count,
        "frontier_match_count": match_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "cases_with_root_refinement_reduction": cases_with_refinement,
        "root_refinement_queries_removed": refinement_removed,
        "root_equivalent_queries_removed": equivalent_removed,
        "counterexamples": examples,
        "size_summaries": size_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cases": report["case_count"],
        "matches": report["frontier_match_count"],
        "refinement_cases": report["cases_with_root_refinement_reduction"],
        "refinement_removed": report["root_refinement_queries_removed"],
    }, indent=2))
    if not report["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
