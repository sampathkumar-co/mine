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
    next_value = 0
    output = []
    for value in values:
        if value not in remap:
            remap[value] = next_value
            next_value += 1
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


def subset_mass(masses: tuple[int, ...], allowed: int) -> int:
    return sum(mass for index, mass in enumerate(masses) if allowed & (1 << index))


def pure(labels: tuple[int, ...], allowed: int) -> bool:
    values = {label for index, label in enumerate(labels) if allowed & (1 << index)}
    return len(values) <= 1


def response_cost(
    costs: tuple[tuple[int, ...], ...], query: int, child: int
) -> int:
    index = (child & -child).bit_length() - 1
    return costs[query][index]


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


def vector_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def retained_queries(
    partitions: tuple[tuple[int, ...], ...],
    costs: tuple[tuple[int, ...], ...],
    allowed: int,
    remaining: tuple[int, ...],
) -> tuple[int, ...]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for query in remaining:
        signature = children(partitions[query], allowed)
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    kept: list[int] = []
    for signature, queries in groups.items():
        vectors = {
            query: tuple(response_cost(costs, query, child) for child in signature)
            for query in queries
        }
        for query in queries:
            dominated = False
            for other in queries:
                if other == query:
                    continue
                if vector_dominates(vectors[other], vectors[query]):
                    dominated = True
                    break
                if vectors[other] == vectors[query] and other < query:
                    dominated = True
                    break
            if not dominated:
                kept.append(query)
    return tuple(sorted(kept))


def enumerate_frontier(
    partitions: tuple[tuple[int, ...], ...],
    labels: tuple[int, ...],
    masses: tuple[int, ...],
    costs: tuple[tuple[int, ...], ...],
    use_quotient: bool,
) -> tuple[Metric, ...]:
    full = (1 << len(labels)) - 1

    @lru_cache(maxsize=None)
    def solve(allowed: int, remaining: tuple[int, ...]) -> tuple[Metric, ...]:
        if pure(labels, allowed):
            return (Metric(subset_mass(masses, allowed), 0, 0),)
        active = (
            retained_queries(partitions, costs, allowed, remaining)
            if use_quotient else tuple(
                query for query in remaining
                if len(children(partitions[query], allowed)) > 1
            )
        )
        candidates: list[Metric] = []
        for query in active:
            query_children = children(partitions[query], allowed)
            next_remaining = tuple(q for q in active if q != query)
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
                        response_cost(costs, query, child) + item.worst_cost
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
    if profile_index == 0:
        masses = tuple(1 for _ in range(size))
    elif profile_index == 1:
        masses = tuple(1 + (index % 2) for index in range(size))
    elif profile_index == 2:
        masses = tuple(2 - (index % 2) for index in range(size))
    else:
        masses = tuple(1 + ((index * 3 + 1) % 2) for index in range(size))
    rows = []
    for query, partition in enumerate(partitions):
        response_values = sorted(set(partition))
        by_response = {}
        for response in response_values:
            if profile_index == 0:
                value = 1
            elif profile_index == 1:
                value = 1 + ((query + response) % 3)
            elif profile_index == 2:
                value = 1 + ((2 * query - response) % 3)
            else:
                digest = hashlib.sha256(
                    f"v62:{query}:{response}:{partition}".encode("utf-8")
                ).digest()
                value = 1 + digest[0] % 3
            by_response[response] = value
        rows.append(tuple(by_response[value] for value in partition))
    return masses, tuple(rows)


def scalar_collapse_counterexample() -> dict[str, object]:
    partitions = ((0, 1), (0, 1))
    costs = ((1, 9), (9, 1))
    labels = (0, 1)
    first = enumerate_frontier(partitions, labels, (9, 1), costs, False)
    second = enumerate_frontier(partitions, labels, (1, 9), costs, False)
    return {
        "passed": first != second and first[0].expected_cost == 18 and second[0].expected_cost == 18,
        "first_frontier": [item.__dict__ for item in first],
        "second_frontier": [item.__dict__ for item in second],
        "meaning": "Incomparable equivalent tests cannot be replaced by one scalar-cost representative independently of the prior.",
    }


def run() -> dict[str, object]:
    case_count = 0
    frontier_match_count = 0
    cases_with_reduction = 0
    removed_total = 0
    mismatches: list[dict[str, object]] = []
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
                    plain = enumerate_frontier(
                        partitions, labels, masses, costs, False
                    )
                    quotient = enumerate_frontier(
                        partitions, labels, masses, costs, True
                    )
                    case_count += 1
                    local_cases += 1
                    if plain == quotient:
                        frontier_match_count += 1
                        local_matches += 1
                    elif len(mismatches) < 20:
                        mismatches.append({
                            "size": size,
                            "partitions": partitions,
                            "labels": labels,
                            "profile": profile_index,
                            "plain": [item.__dict__ for item in plain],
                            "quotient": [item.__dict__ for item in quotient],
                        })
                    full = (1 << size) - 1
                    retained = retained_queries(
                        partitions, costs, full, tuple(range(query_count))
                    )
                    removed = query_count - len(retained)
                    removed_total += removed
                    cases_with_reduction += int(removed > 0)
        size_summaries.append({
            "hypotheses": size,
            "queries": query_count,
            "partition_signatures": len(signatures),
            "labelings": len(labels_set),
            "cases": local_cases,
            "frontier_matches": local_matches,
        })
    counterexample = scalar_collapse_counterexample()
    gate = (
        counterexample["passed"]
        and case_count >= 15_000
        and frontier_match_count == case_count
        and not mismatches
        and cases_with_reduction >= 1_000
        and removed_total >= 1_000
    )
    protocol = {
        "finite_models": [[3, 4], [4, 3]],
        "partition_enumeration": "all nontrivial set partitions, combinations with replacement",
        "binary_labels": "all nonconstant labels modulo global complement",
        "mass_profiles": 4,
        "response_cost_values": [1, 2, 3],
        "oracle": "direct complete-tree enumeration with nondominated feasible-metric frontier",
    }
    digest = hashlib.sha256(json.dumps({
        "protocol": protocol,
        "size_summaries": size_summaries,
        "case_count": case_count,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": "exhaustive_finite_tree_certificate_pass" if gate else "rejected",
        "development_gate": gate,
        "claim_scope": "Exhaustive finite-model equality between direct complete-tree enumeration before and after descendant-local response-cost Pareto quotienting. This is a machine-checked small-instance certificate, not a proof assistant theorem or independent peer review.",
        "protocol": protocol,
        "protocol_digest": digest,
        "case_count": case_count,
        "frontier_match_count": frontier_match_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "cases_with_root_reduction": cases_with_reduction,
        "root_queries_removed": removed_total,
        "scalar_collapse_counterexample": counterexample,
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
        "reductions": report["cases_with_root_reduction"],
    }, indent=2))
    if not report["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
