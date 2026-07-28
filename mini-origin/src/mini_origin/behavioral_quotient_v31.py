from __future__ import annotations

from dataclasses import dataclass
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import cross_domain_compiler_v30 as base
from . import cross_domain_runner_v30 as runner

Query = base.Query
Task = base.Task
Policy = base.Policy
Evaluation = base.Evaluation

QUERY_RULES = ("first", "minimax_bucket", "gini", "max_outcomes")
DECODER_RULES = ("argmin", "argmax", "first_below")
COMPLEXITY = {"minimax_bucket": 1, "gini": 2, "max_outcomes": 3, "first": 4}


@dataclass(frozen=True)
class BehaviourClass:
    members: tuple[str, ...]
    canonical: str
    digest: str
    states: int


def _partition_signature(task: Task, allowed: frozenset[int], index: int) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(part)) for part in base.partitions(task.queries[index], allowed).values()))


def _reachable(task: Task, rule: str) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    rows = set()
    for target in range(task.candidate_count):
        allowed = frozenset(range(task.candidate_count))
        remaining = frozenset(range(len(task.queries)))
        while len(allowed) > 1:
            state = (tuple(sorted(allowed)), tuple(sorted(remaining)))
            if state in rows:
                break
            rows.add(state)
            try:
                index = base.select_query(task, allowed, remaining, rule)
            except RuntimeError:
                break
            query = task.queries[index]
            nxt = base.partitions(query, allowed).get(query.outcomes[target], frozenset())
            if not nxt or nxt == allowed:
                break
            allowed = nxt
            remaining = frozenset(value for value in remaining if value != index)
    return rows


def behaviour_quotient(tasks: list[Task], rules: Iterable[str] = QUERY_RULES) -> tuple[BehaviourClass, ...]:
    rules = tuple(rules)
    state_union = {}
    for ti, task in enumerate(tasks):
        state_union[ti] = set().union(*(_reachable(task, rule) for rule in rules))
    groups: dict[str, list[str]] = {}
    counts = {}
    for rule in rules:
        signature = []
        for ti, task in enumerate(tasks):
            for allowed_values, remaining_values in sorted(state_union[ti]):
                allowed = frozenset(allowed_values)
                remaining = frozenset(remaining_values)
                try:
                    index = base.select_query(task, allowed, remaining, rule)
                    partition = _partition_signature(task, allowed, index)
                except RuntimeError:
                    partition = ((),)
                signature.append((ti, allowed_values, remaining_values, partition))
        digest = hashlib.sha256(repr(tuple(signature)).encode()).hexdigest()
        groups.setdefault(digest, []).append(rule)
        counts[digest] = len(signature)
    classes = []
    for digest, members in groups.items():
        ordered = tuple(sorted(members, key=lambda rule: (COMPLEXITY[rule], rule)))
        classes.append(BehaviourClass(ordered, ordered[0], digest, counts[digest]))
    return tuple(sorted(classes, key=lambda row: (COMPLEXITY[row.canonical], row.members)))


def binary_equivalence_certificate(maximum_total: int = 255) -> dict[str, object]:
    violations = []
    comparisons = 0
    for total in range(2, maximum_total + 1):
        parts = sorted({tuple(sorted((left, total - left), reverse=True)) for left in range(1, total)})
        for first in parts:
            for second in parts:
                minimax = (max(first), sum(x * x for x in first)) < (max(second), sum(x * x for x in second))
                gini = (sum(x * x for x in first), max(first)) < (sum(x * x for x in second), max(second))
                comparisons += 1
                if minimax != gini:
                    violations.append((total, first, second))
    return {
        "maximum_total": maximum_total,
        "comparisons": comparisons,
        "violations": violations,
        "passed": not violations,
        "identity": "For binary buckets (m,n-m), the sum of squares is monotone in the larger bucket m.",
    }


@functools.lru_cache(maxsize=None)
def _parts(total: int, slots: int, limit: int) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    if slots == 0:
        return ()
    rows = []
    for first in range(min(total, limit), 0, -1):
        rows.extend((first,) + rest for rest in _parts(total - first, slots - 1, first))
    return tuple(rows)


def integer_partitions(total: int, maximum_parts: int = 5) -> tuple[tuple[int, ...], ...]:
    return tuple(row for row in _parts(total, maximum_parts, total) if len(row) >= 2)


def reversal_pair(total: int, maximum_parts: int = 5) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    rows = [row for row in integer_partitions(total, maximum_parts) if len(row) >= 3]
    candidates = []
    for robust in rows:
        for lure in rows:
            robust_sq = sum(x * x for x in robust)
            lure_sq = sum(x * x for x in lure)
            if max(robust) < max(lure) and robust_sq > lure_sq:
                candidates.append((robust_sq - lure_sq, max(lure) - max(robust), robust, lure))
    return max(candidates)[-2:] if candidates else None


def multioutcome_divergence_certificate(maximum_total: int = 64) -> dict[str, object]:
    witnesses = []
    for total in range(3, maximum_total + 1):
        pair = reversal_pair(total)
        if pair:
            robust, lure = pair
            witnesses.append({
                "total": total,
                "minimax_partition": robust,
                "gini_partition": lure,
                "minimax_max": max(robust),
                "gini_max": max(lure),
                "minimax_sum_squares": sum(x * x for x in robust),
                "gini_sum_squares": sum(x * x for x in lure),
            })
    return {"maximum_total": maximum_total, "witness_count": len(witnesses), "first_witness": witnesses[0] if witnesses else None, "passed": bool(witnesses)}


def _balanced(total: int, outcomes: int = 3) -> tuple[int, ...]:
    outcomes = min(total, outcomes)
    quotient, remainder = divmod(total, outcomes)
    return tuple(quotient + int(index < remainder) for index in range(outcomes))


def _split(subset: tuple[int, ...], sizes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    groups = []
    offset = 0
    for size in sizes:
        groups.append(subset[offset:offset + size])
        offset += size
    return tuple(groups)


def divergent_task(candidate_count: int, seed: int, domain: str, maximum_parts: int) -> Task:
    rng = np.random.default_rng(seed)
    root = tuple(int(value) for value in rng.permutation(candidate_count))
    queries = []
    seen = set()
    visited = set()

    def add(groups: tuple[tuple[int, ...], ...], name: str) -> None:
        outcomes = [-1] * candidate_count
        for outcome, group in enumerate(groups):
            for candidate in group:
                outcomes[candidate] = outcome
        row = tuple(outcomes)
        if row not in seen:
            seen.add(row)
            queries.append(Query(name, row))

    def recurse(subset: tuple[int, ...], depth: int) -> None:
        key = frozenset(subset)
        if len(subset) <= 1 or key in visited:
            return
        visited.add(key)
        pair = reversal_pair(len(subset), maximum_parts)
        robust_sizes, lure_sizes = pair if pair else (_balanced(len(subset)), _balanced(len(subset)))
        robust_groups = _split(subset, robust_sizes)
        add(robust_groups, f"robust-{depth}-{len(queries)}")
        if lure_sizes != robust_sizes:
            add(_split(subset, lure_sizes), f"lure-{depth}-{len(queries)}")
        for group in robust_groups:
            recurse(group, depth + 1)

    recurse(root, 0)
    for candidate in root:
        queries.append(Query(f"confirm-{candidate}", tuple(0 if value == candidate else 1 for value in range(candidate_count))))
    return Task(domain, candidate_count, tuple(queries))


def hidden_tasks(seed: int) -> list[Task]:
    return [
        divergent_task(24, seed + 1, "multichannel_fault", 4),
        divergent_task(36, seed + 2, "categorical_diagnosis", 5),
        divergent_task(48, seed + 3, "assay_panel", 4),
        divergent_task(60, seed + 4, "failure_signature", 5),
    ]


def run_trial(seed: int, task: Task, target: int, policy: Policy, budget: int) -> tuple[bool, int, bool]:
    allowed = frozenset(range(task.candidate_count))
    remaining = frozenset(range(len(task.queries)))
    count = 0
    while len(allowed) > 1 and count < budget:
        try:
            index = base.select_query(task, allowed, remaining, policy.query_rule)
        except RuntimeError:
            return False, count, True
        query = task.queries[index]
        outcomes, values = base.observe(seed + count * 7919, query, target, replicates=384)
        observed = base.decode(policy.decoder_rule, outcomes, values)
        parts = base.partitions(query, allowed)
        count += 1
        if observed not in parts:
            return False, count, True
        allowed = parts[observed]
        remaining = frozenset(value for value in remaining if value != index)
        if target not in allowed:
            return False, count, True
    return len(allowed) == 1 and next(iter(allowed), -1) == target, count, False


def evaluate(seed: int, tasks: list[Task], trials_per_task: int, policy: Policy) -> Evaluation:
    rng = np.random.default_rng(seed)
    rows = []
    sizes = []
    for task_index, task in enumerate(tasks):
        budget = math.ceil(math.log2(task.candidate_count)) + 2
        for trial, target in enumerate(rng.integers(0, task.candidate_count, size=trials_per_task)):
            rows.append(run_trial(seed + task_index * 1_000_003 + trial * 104_729, task, int(target), policy, budget))
            sizes.append(task.candidate_count)
    counts = [row[1] for row in rows]
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean(counts)),
        maximum_queries=max(counts),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        mean_log2_ratio=float(np.mean([count / max(1, math.ceil(math.log2(size))) for count, size in zip(counts, sizes)])),
    )


def exact_divergence_certificate(seed: int, canonical: Policy, alternative: Policy) -> dict[str, object]:
    tasks = [
        divergent_task(8, seed + 1, "certificate_four", 4),
        divergent_task(9, seed + 2, "certificate_five", 5),
    ]
    canonical_proof = base.exact_certificate(tasks, canonical)
    rows = []
    for task in tasks:
        allowed = frozenset(range(task.candidate_count))
        remaining = frozenset(range(len(task.queries)))
        canonical_index = base.select_query(task, allowed, remaining, canonical.query_rule)
        alternative_index = base.select_query(task, allowed, remaining, alternative.query_rule)
        canonical_depth = max(base.noiseless_depth(task, canonical, target) for target in range(task.candidate_count))
        alternative_depth = max(base.noiseless_depth(task, alternative, target) for target in range(task.candidate_count))
        rows.append({
            "domain": task.domain,
            "canonical_root_query": task.queries[canonical_index].name,
            "alternative_root_query": task.queries[alternative_index].name,
            "canonical_worst_depth": canonical_depth,
            "alternative_worst_depth": alternative_depth,
        })
    passed = canonical_proof["passed"] and all(row["canonical_worst_depth"] <= row["alternative_worst_depth"] for row in rows) and any(row["canonical_worst_depth"] < row["alternative_worst_depth"] for row in rows)
    return {"canonical_proof": canonical_proof, "rows": rows, "root_divergence": all(row["canonical_root_query"] != row["alternative_root_query"] for row in rows), "passed": passed}


def class_digest(value: BehaviourClass, decoder: str) -> str:
    return hashlib.sha256(f"{value.digest}:{','.join(value.members)}:{value.canonical}:{decoder}".encode()).hexdigest()


def run(seed: int = 1601) -> dict[str, object]:
    train = base.training_tasks(seed * 10_000 + 107)
    development = base.development_tasks(seed * 10_000 + 1_000_109)
    quotient = behaviour_quotient(train + development)
    rows = []
    for value in quotient:
        for decoder in DECODER_RULES:
            policy = Policy(value.canonical, decoder)
            train_score = runner.evaluate(seed * 10_000 + 2_000_003, train, 24, policy)
            development_score = runner.evaluate(seed * 10_000 + 3_000_007, development, 32, policy)
            score = (development_score.accuracy, -development_score.invalid_rate, -development_score.mean_queries, len(value.members), -COMPLEXITY[value.canonical], int(decoder == "argmin"))
            rows.append((score, value, decoder, train_score, development_score))
    _, selected, decoder, train_score, development_score = max(rows, key=lambda row: row[0])
    policy = Policy(selected.canonical, decoder)
    frozen_digest = class_digest(selected, decoder)
    binary_certificate = binary_equivalence_certificate()
    divergence_certificate = multioutcome_divergence_certificate()

    # Hidden multi-outcome domains are generated only after the quotient class,
    # canonical representative, decoder and digest are frozen.
    hidden = hidden_tasks(seed * 10_000 + 12_000_001)
    candidate = evaluate(seed * 10_000 + 13_000_003, hidden, 96, policy)
    gini = evaluate(seed * 10_000 + 13_000_005, hidden, 96, Policy("gini", decoder))
    first = evaluate(seed * 10_000 + 13_000_007, hidden, 96, Policy("first", "argmin"))
    outcomes = evaluate(seed * 10_000 + 13_000_009, hidden, 96, Policy("max_outcomes", "argmin"))
    proof = exact_divergence_certificate(seed * 10_000 + 13_000_011, policy, Policy("gini", decoder))
    gini_gain = gini.mean_queries - candidate.mean_queries
    first_gap = candidate.accuracy - first.accuracy
    outcomes_gap = candidate.accuracy - outcomes.accuracy
    members = set(selected.members)
    gate = (
        {"minimax_bucket", "gini"}.issubset(members)
        and selected.canonical == "minimax_bucket"
        and decoder == "argmin"
        and binary_certificate["passed"]
        and divergence_certificate["passed"]
        and development_score.accuracy >= 0.985
        and candidate.accuracy >= 0.985
        and candidate.invalid_rate <= 0.01
        and candidate.mean_log2_ratio <= 1.15
        and gini_gain >= 0.20
        and first_gap >= 0.20
        and outcomes_gap >= 0.10
        and proof["passed"]
    )
    return {
        "status": "behavioral_quotient_transfer_candidate" if gate else "not_yet",
        "claim_scope": "Query objectives are quotiented by exact reachable-state behaviour, a canonical representative is frozen before hidden data, and the class is tested on multi-outcome diagnosis tasks where binary-equivalent objectives diverge; this remains generalized binary search and active diagnosis, not a world breakthrough.",
        "seed": seed,
        "candidate_gate": gate,
        "quotient_classes": [{"members": list(value.members), "canonical": value.canonical, "digest": value.digest, "states": value.states} for value in quotient],
        "selected_class": {"members": list(selected.members), "canonical": selected.canonical, "decoder": decoder},
        "training_score": train_score.__dict__,
        "development_score": development_score.__dict__,
        "frozen_class_digest": frozen_digest,
        "binary_equivalence_certificate": binary_certificate,
        "multioutcome_divergence_certificate": divergence_certificate,
        "hidden_domains": [task.domain for task in hidden],
        "candidate": candidate.__dict__,
        "gini_control": gini.__dict__,
        "first_control": first.__dict__,
        "max_outcomes_control": outcomes.__dict__,
        "gini_query_gain": gini_gain,
        "first_accuracy_gap": first_gap,
        "max_outcomes_accuracy_gap": outcomes_gap,
        "exact_divergence_certificate": proof,
        "classes_evaluated": len(quotient),
        "policies_evaluated": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_class": report["selected_class"],
        "accuracy": report["candidate"]["accuracy"],
        "mean_queries": report["candidate"]["mean_queries"],
        "gini_query_gain": report["gini_query_gain"],
        "first_gap": report["first_accuracy_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
