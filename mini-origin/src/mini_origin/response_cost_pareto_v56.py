from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from . import average_odt_frontier_v44 as frontier
from . import cost_prior_quotient_v48 as weighted
from . import exact_quotient_certificate_v42 as corpus
from . import state_policy_v34 as state


BUDGET = 500_000
BUDGET_LADDER = (10_000, 50_000, 250_000, 500_000)
PROFILE_SEEDS = (5601, 5602, 5603)
RANDOM_THEOREM_TASKS = 48


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ResponseCostProfile:
    hypothesis_mass: tuple[int, ...]
    hypothesis_cost_by_query: tuple[tuple[int, ...], ...]
    seed: int


@dataclass(frozen=True)
class Plan:
    diagnosed_mass: int
    expected_cost_numerator: int
    worst_cost: int
    query: int | None


@dataclass(frozen=True)
class SolverStats:
    calls: int
    memo_entries: int
    query_expansions: int
    memo_hits: int
    raw_queries_considered: int
    representative_queries_considered: int
    dominated_queries_removed: int


@dataclass(frozen=True)
class SolveResult:
    plan: Plan
    stats: SolverStats


def plan_score(plan: Plan) -> tuple[int, int, int, int]:
    return (
        plan.diagnosed_mass,
        -plan.expected_cost_numerator,
        -plan.worst_cost,
        -(plan.query if plan.query is not None else 10**9),
    )


def plan_metrics(plan: Plan) -> tuple[int, int, int]:
    return (
        plan.diagnosed_mass,
        plan.expected_cost_numerator,
        plan.worst_cost,
    )


def partition(task: object, allowed: int, query: int) -> tuple[int, ...]:
    children = []
    for response_mask in task.masks_for(query).values():
        child = allowed & response_mask
        if child:
            children.append(child)
    return tuple(sorted(children))


def subset_mass(profile: ResponseCostProfile, allowed: int) -> int:
    total = 0
    pending = allowed
    while pending:
        bit = pending & -pending
        index = bit.bit_length() - 1
        pending ^= bit
        total += profile.hypothesis_mass[index]
    return total


def cell_cost(
    profile: ResponseCostProfile,
    query: int,
    child: int,
) -> int:
    index = (child & -child).bit_length() - 1
    return profile.hypothesis_cost_by_query[query][index]


def cost_vector(
    task: object,
    profile: ResponseCostProfile,
    allowed: int,
    query: int,
) -> tuple[int, ...]:
    return tuple(
        cell_cost(profile, query, child)
        for child in partition(task, allowed, query)
    )


def vector_dominates(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> bool:
    return (
        len(left) == len(right)
        and all(a <= b for a, b in zip(left, right))
        and any(a < b for a, b in zip(left, right))
    )


def representative_dominates(
    left_query: int,
    left_vector: tuple[int, ...],
    right_query: int,
    right_vector: tuple[int, ...],
) -> bool:
    if vector_dominates(left_vector, right_vector):
        return True
    return left_vector == right_vector and left_query < right_query


def pareto_representatives(
    task: object,
    profile: ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = partition(task, allowed, query)
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    result: dict[tuple[int, ...], tuple[int, ...]] = {}
    for signature, queries in groups.items():
        vectors = {
            query: tuple(
                cell_cost(profile, query, child)
                for child in signature
            )
            for query in queries
        }
        keep = []
        for query in queries:
            if any(
                other != query
                and representative_dominates(
                    other,
                    vectors[other],
                    query,
                    vectors[query],
                )
                for other in queries
            ):
                continue
            keep.append(query)
        result[signature] = tuple(sorted(keep))
    return result


def pareto_mask(
    task: object,
    profile: ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> int:
    mask = 0
    for queries in pareto_representatives(
        task, profile, allowed, remaining
    ).values():
        for query in queries:
            mask |= 1 << query
    return mask


def pareto_certificate(
    task: object,
    profile: ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> dict[str, int | bool]:
    groups: dict[tuple[int, ...], list[int]] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = partition(task, allowed, query)
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    selected = pareto_representatives(
        task, profile, allowed, remaining
    )
    passed = True
    dominated = 0
    incomparable_classes = 0
    duplicate_classes = 0
    for signature, queries in groups.items():
        selected_set = set(selected[signature])
        if len(queries) > 1:
            duplicate_classes += 1
        if len(selected_set) > 1:
            incomparable_classes += 1
        vectors = {
            query: tuple(
                cell_cost(profile, query, child)
                for child in signature
            )
            for query in queries
        }
        for query in queries:
            if query in selected_set:
                if any(
                    other != query
                    and representative_dominates(
                        other,
                        vectors[other],
                        query,
                        vectors[query],
                    )
                    for other in queries
                ):
                    passed = False
            else:
                dominated += 1
                if not any(
                    representative_dominates(
                        other,
                        vectors[other],
                        query,
                        vectors[query],
                    )
                    for other in selected_set
                ):
                    passed = False
    return {
        "passed": passed,
        "separating_partition_classes": len(groups),
        "duplicate_partition_classes": duplicate_classes,
        "incomparable_pareto_classes": incomparable_classes,
        "dominated_queries_removed": dominated,
    }


def stable_value(
    seed: int,
    token: str,
    low: int,
    high: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed}:{token}".encode("utf-8")
    ).digest()
    return low + int.from_bytes(digest[:8], "big") % (
        high - low + 1
    )


def profile_for_task(task: object, seed: int) -> ResponseCostProfile:
    masses = weighted.profile_for_task(
        task, seed
    ).hypothesis_mass
    rows = []
    for query in range(task.query_count):
        response_values = sorted({
            task.rows[index][query]
            for index in range(task.candidate_count)
        })
        base = stable_value(
            seed,
            f"{task.name}:query-base:{query}",
            1,
            15,
        )
        response_cost = {
            value: base + stable_value(
                seed,
                f"{task.name}:query:{query}:response:{value}",
                0,
                4,
            )
            for value in response_values
        }
        rows.append(tuple(
            response_cost[task.rows[index][query]]
            for index in range(task.candidate_count)
        ))
    return ResponseCostProfile(
        hypothesis_mass=tuple(masses),
        hypothesis_cost_by_query=tuple(rows),
        seed=seed,
    )


def profile_digest(
    task: object,
    profile: ResponseCostProfile,
) -> str:
    return hashlib.sha256(json.dumps({
        "task": task.name,
        "seed": profile.seed,
        "hypothesis_mass": profile.hypothesis_mass,
        "hypothesis_cost_by_query": (
            profile.hypothesis_cost_by_query
        ),
    }, sort_keys=True).encode("utf-8")).hexdigest()


def immediate_expected_cost(
    profile: ResponseCostProfile,
    allowed: int,
    query: int,
) -> int:
    total = 0
    pending = allowed
    while pending:
        bit = pending & -pending
        index = bit.bit_length() - 1
        pending ^= bit
        total += (
            profile.hypothesis_mass[index]
            * profile.hypothesis_cost_by_query[query][index]
        )
    return total


class PlainPlanner:
    def __init__(
        self,
        task: object,
        profile: ResponseCostProfile,
        budget: int,
    ) -> None:
        self.task = task
        self.profile = profile
        self.budget = budget
        self.memo: dict[tuple[int, int], Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0
        self.raw_queries_considered = 0

    def solve(self, allowed: int, remaining: int) -> Plan:
        self.calls += 1
        key = allowed, remaining
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        mass = subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = Plan(mass, 0, 0, None)
            self.memo[key] = answer
            return answer
        candidates = []
        pending = remaining
        while pending:
            bit = pending & -pending
            query = bit.bit_length() - 1
            pending ^= bit
            self.raw_queries_considered += 1
            children = partition(self.task, allowed, query)
            if len(children) <= 1:
                continue
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded(
                    "plain response-cost budget exceeded"
                )
            child_plans = [
                self.solve(
                    child,
                    remaining & ~(1 << query),
                )
                for child in children
            ]
            candidates.append(Plan(
                diagnosed_mass=sum(
                    child.diagnosed_mass
                    for child in child_plans
                ),
                expected_cost_numerator=(
                    immediate_expected_cost(
                        self.profile, allowed, query
                    )
                    + sum(
                        child.expected_cost_numerator
                        for child in child_plans
                    )
                ),
                worst_cost=max(
                    cell_cost(self.profile, query, child_mask)
                    + child_plan.worst_cost
                    for child_mask, child_plan
                    in zip(children, child_plans)
                ),
                query=query,
            ))
        answer = (
            max(candidates, key=plan_score)
            if candidates else Plan(0, 0, 0, None)
        )
        self.memo[key] = answer
        return answer

    def result(self, allowed: int, remaining: int) -> SolveResult:
        answer = self.solve(allowed, remaining)
        return SolveResult(answer, SolverStats(
            calls=self.calls,
            memo_entries=len(self.memo),
            query_expansions=self.query_expansions,
            memo_hits=self.memo_hits,
            raw_queries_considered=(
                self.raw_queries_considered
            ),
            representative_queries_considered=(
                self.raw_queries_considered
            ),
            dominated_queries_removed=0,
        ))


class ParetoPlanner:
    def __init__(
        self,
        task: object,
        profile: ResponseCostProfile,
        budget: int,
    ) -> None:
        self.task = task
        self.profile = profile
        self.budget = budget
        self.memo: dict[tuple[int, int], Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0
        self.raw_queries_considered = 0
        self.representative_queries_considered = 0
        self.dominated_queries_removed = 0

    def canonical(self, allowed: int, remaining: int) -> int:
        raw = remaining.bit_count()
        canonical = pareto_mask(
            self.task,
            self.profile,
            allowed,
            remaining,
        )
        kept = canonical.bit_count()
        self.raw_queries_considered += raw
        self.representative_queries_considered += kept
        self.dominated_queries_removed += max(0, raw - kept)
        return canonical

    def solve(self, allowed: int, remaining: int) -> Plan:
        self.calls += 1
        canonical = self.canonical(allowed, remaining)
        key = allowed, canonical
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        mass = subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = Plan(mass, 0, 0, None)
            self.memo[key] = answer
            return answer
        candidates = []
        pending = canonical
        while pending:
            bit = pending & -pending
            query = bit.bit_length() - 1
            pending ^= bit
            children = partition(self.task, allowed, query)
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded(
                    "Pareto response-cost budget exceeded"
                )
            child_plans = [
                self.solve(
                    child,
                    canonical & ~(1 << query),
                )
                for child in children
            ]
            candidates.append(Plan(
                diagnosed_mass=sum(
                    child.diagnosed_mass
                    for child in child_plans
                ),
                expected_cost_numerator=(
                    immediate_expected_cost(
                        self.profile, allowed, query
                    )
                    + sum(
                        child.expected_cost_numerator
                        for child in child_plans
                    )
                ),
                worst_cost=max(
                    cell_cost(self.profile, query, child_mask)
                    + child_plan.worst_cost
                    for child_mask, child_plan
                    in zip(children, child_plans)
                ),
                query=query,
            ))
        answer = (
            max(candidates, key=plan_score)
            if candidates else Plan(0, 0, 0, None)
        )
        self.memo[key] = answer
        return answer

    def result(self, allowed: int, remaining: int) -> SolveResult:
        answer = self.solve(allowed, remaining)
        return SolveResult(answer, SolverStats(
            calls=self.calls,
            memo_entries=len(self.memo),
            query_expansions=self.query_expansions,
            memo_hits=self.memo_hits,
            raw_queries_considered=(
                self.raw_queries_considered
            ),
            representative_queries_considered=(
                self.representative_queries_considered
            ),
            dominated_queries_removed=(
                self.dominated_queries_removed
            ),
        ))


def descendants(mask: int):
    subset = mask
    while subset:
        yield subset
        subset = (subset - 1) & mask


def dominance_map(
    task: object,
    profile: ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> dict[int, int]:
    groups: dict[tuple[int, ...], list[int]] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = partition(task, allowed, query)
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    removed: dict[int, int] = {}
    for signature, queries in groups.items():
        vectors = {
            query: tuple(
                cell_cost(profile, query, child)
                for child in signature
            )
            for query in queries
        }
        for query in queries:
            dominators = [
                other for other in queries
                if other != query
                and representative_dominates(
                    other,
                    vectors[other],
                    query,
                    vectors[query],
                )
            ]
            if dominators:
                removed[query] = min(dominators)
    return removed


def hereditary_pareto_theorem(
    task: object,
    profile: ResponseCostProfile,
) -> dict[str, object]:
    comparisons = 0
    descendant_checks = 0
    violations = []
    remaining = (1 << task.query_count) - 1
    for allowed in descendants(task.full_mask):
        removed = dominance_map(
            task, profile, allowed, remaining
        )
        for query, dominator in removed.items():
            comparisons += 1
            for child in descendants(allowed):
                left_signature = partition(
                    task, child, dominator
                )
                right_signature = partition(
                    task, child, query
                )
                descendant_checks += 1
                if left_signature != right_signature:
                    violations.append({
                        "kind": "partition",
                        "allowed": allowed,
                        "child": child,
                        "dominator": dominator,
                        "query": query,
                    })
                    break
                left_vector = tuple(
                    cell_cost(
                        profile, dominator, cell
                    )
                    for cell in left_signature
                )
                right_vector = tuple(
                    cell_cost(profile, query, cell)
                    for cell in right_signature
                )
                if not (
                    vector_dominates(
                        left_vector, right_vector
                    )
                    or (
                        left_vector == right_vector
                        and dominator < query
                    )
                ):
                    violations.append({
                        "kind": "cost-dominance",
                        "allowed": allowed,
                        "child": child,
                        "dominator": dominator,
                        "query": query,
                        "left": left_vector,
                        "right": right_vector,
                    })
                    break
    return {
        "comparisons": comparisons,
        "descendant_checks": descendant_checks,
        "violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations,
    }


def random_task_and_profile(seed: int):
    rng = random.Random(seed)
    hypotheses = rng.randint(5, 8)
    base_queries = rng.randint(3, 6)
    columns = []
    for _ in range(base_queries):
        outcomes = rng.randint(2, 4)
        columns.append([
            str(rng.randrange(outcomes))
            for _ in range(hypotheses)
        ])
    duplicate_source = rng.randrange(base_queries)
    source_values = columns[duplicate_source]
    remap = {
        value: f"r{index}"
        for index, value in enumerate(
            sorted(set(source_values), reverse=True)
        )
    }
    columns.append([
        remap[value] for value in source_values
    ])
    rows = [
        tuple(column[row] for column in columns)
        for row in range(hypotheses)
    ]
    labels = tuple(
        str(rng.randrange(rng.randint(2, 4)))
        for _ in rows
    )
    task = state.base.make_task(
        f"response-cost-theorem-{seed}",
        tuple(f"q{index}" for index in range(len(columns))),
        rows,
        labels,
    )
    masses = tuple(rng.randint(1, 19) for _ in rows)
    costs = []
    for query, column in enumerate(columns):
        by_response = {
            value: rng.randint(1, 13)
            for value in sorted(set(column))
        }
        costs.append(tuple(
            by_response[column[index]]
            for index in range(hypotheses)
        ))
    if seed % 2 == 0:
        source = duplicate_source
        duplicate = len(columns) - 1
        source_signature = partition(
            task, task.full_mask, source
        )
        duplicate_signature = partition(
            task, task.full_mask, duplicate
        )
        source_vector = [rng.randint(5, 10) for _ in source_signature]
        duplicate_vector = [max(1, value - rng.randint(0, 4)) for value in source_vector]
        for query, signature, vector in (
            (source, source_signature, source_vector),
            (duplicate, duplicate_signature, duplicate_vector),
        ):
            query_cost = list(costs[query])
            for child, value in zip(signature, vector):
                pending = child
                while pending:
                    bit = pending & -pending
                    index = bit.bit_length() - 1
                    pending ^= bit
                    query_cost[index] = value
            costs[query] = tuple(query_cost)
    profile = ResponseCostProfile(
        hypothesis_mass=masses,
        hypothesis_cost_by_query=tuple(costs),
        seed=seed,
    )
    return task, profile


def incomparable_counterexample() -> dict[str, object]:
    task = state.base.make_task(
        "incomparable-response-cost-counterexample",
        ("left-cheap", "right-cheap"),
        (("0", "x"), ("1", "y")),
        ("a", "b"),
    )
    first = ResponseCostProfile(
        hypothesis_mass=(9, 1),
        hypothesis_cost_by_query=(
            (1, 9),
            (9, 1),
        ),
        seed=1,
    )
    second = ResponseCostProfile(
        hypothesis_mass=(1, 9),
        hypothesis_cost_by_query=first.hypothesis_cost_by_query,
        seed=2,
    )
    remaining = (1 << task.query_count) - 1
    first_plan = PlainPlanner(
        task, first, 100
    ).result(task.full_mask, remaining).plan
    second_plan = PlainPlanner(
        task, second, 100
    ).result(task.full_mask, remaining).plan
    vectors = [
        cost_vector(task, first, task.full_mask, query)
        for query in range(task.query_count)
    ]
    passed = (
        partition(task, task.full_mask, 0)
        == partition(task, task.full_mask, 1)
        and not vector_dominates(vectors[0], vectors[1])
        and not vector_dominates(vectors[1], vectors[0])
        and first_plan.query == 0
        and second_plan.query == 1
    )
    return {
        "passed": passed,
        "cost_vectors": vectors,
        "first_prior": first.hypothesis_mass,
        "first_optimal_query": first_plan.query,
        "second_prior": second.hypothesis_mass,
        "second_optimal_query": second_plan.query,
        "meaning": (
            "Equivalent tests with incomparable response-cost vectors "
            "cannot be collapsed to one universal representative."
        ),
    }


def random_theorem_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    for seed in range(56001, 56001 + RANDOM_THEOREM_TASKS):
        task, profile = random_task_and_profile(seed)
        theorem = hereditary_pareto_theorem(task, profile)
        remaining = (1 << task.query_count) - 1
        plain = PlainPlanner(
            task, profile, BUDGET
        ).result(task.full_mask, remaining)
        pareto = ParetoPlanner(
            task, profile, BUDGET
        ).result(task.full_mask, remaining)
        matched = plan_metrics(plain.plan) == plan_metrics(
            pareto.plan
        )
        if not matched:
            mismatches.append({
                "seed": seed,
                "plain": plan_metrics(plain.plan),
                "pareto": plan_metrics(pareto.plan),
            })
        rows.append({
            "seed": seed,
            "hypotheses": task.candidate_count,
            "queries": task.query_count,
            "theorem_passed": theorem["passed"],
            "plan_matched": matched,
            "plain_expansions": plain.stats.query_expansions,
            "pareto_expansions": pareto.stats.query_expansions,
            "dominated_queries_removed": (
                pareto.stats.dominated_queries_removed
            ),
        })
    return {
        "task_count": len(rows),
        "theorem_pass_count": sum(
            int(row["theorem_passed"]) for row in rows
        ),
        "plan_match_count": sum(
            int(row["plan_matched"]) for row in rows
        ),
        "tasks_with_dominance_reduction": sum(
            int(row["dominated_queries_removed"] > 0)
            for row in rows
        ),
        "mismatches": mismatches,
        "passed": (
            len(rows) == RANDOM_THEOREM_TASKS
            and not mismatches
            and all(row["theorem_passed"] for row in rows)
        ),
        "rows": rows,
    }


def evaluate_state(
    task: object,
    profile: ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    pareto_result = None
    plain_result = None
    pareto_solved = False
    plain_solved = False
    try:
        pareto_result = ParetoPlanner(
            task, profile, BUDGET
        ).result(allowed, remaining)
        pareto_solved = True
    except BudgetExceeded:
        pass
    if pareto_solved:
        try:
            plain_result = PlainPlanner(
                task, profile, BUDGET
            ).result(allowed, remaining)
            plain_solved = True
        except BudgetExceeded:
            pass
    matched = (
        pareto_solved
        and plain_solved
        and plan_metrics(pareto_result.plan)
        == plan_metrics(plain_result.plan)
    )
    if pareto_solved:
        plain_lower = (
            plain_result.stats.query_expansions
            if plain_solved else BUDGET + 1
        )
        ratio = plain_lower / max(
            1, pareto_result.stats.query_expansions
        )
    else:
        ratio = None
    ladder = {}
    for budget in BUDGET_LADDER:
        ladder[str(budget)] = {
            "pareto_solved": (
                pareto_solved
                and pareto_result.stats.query_expansions
                <= budget
            ),
            "plain_solved": (
                plain_solved
                and plain_result.stats.query_expansions
                <= budget
            ),
        }
    certificate = pareto_certificate(
        task, profile, allowed, remaining
    )
    return {
        "profile_seed": profile.seed,
        "profile_digest": profile_digest(task, profile),
        "candidate_count": allowed.bit_count(),
        "candidate_mass": subset_mass(profile, allowed),
        "raw_remaining_queries": remaining.bit_count(),
        "root_pareto_certificate": certificate,
        "pareto_solved": pareto_solved,
        "plain_solved": plain_solved,
        "matched_if_both": matched,
        "pareto_plan": (
            plan_metrics(pareto_result.plan)
            if pareto_result else None
        ),
        "plain_plan": (
            plan_metrics(plain_result.plan)
            if plain_result else None
        ),
        "pareto_stats": (
            pareto_result.stats.__dict__
            if pareto_result else None
        ),
        "plain_stats": (
            plain_result.stats.__dict__
            if plain_result else None
        ),
        "expansion_ratio_lower_bound": ratio,
        "budget_ladder": ladder,
    }


def run() -> dict[str, object]:
    counterexample = incomparable_counterexample()
    theorem = random_theorem_certificate()
    tasks, verification = corpus.load_all_opened_tasks()
    rows = []
    base_state_digests = set()
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            base_digest = hashlib.sha256(
                f"{task.name}:{allowed}:{remaining}".encode(
                    "utf-8"
                )
            ).hexdigest()
            base_state_digests.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = profile_for_task(task, seed)
                row = evaluate_state(
                    task,
                    profile,
                    allowed,
                    remaining,
                )
                row["task"] = task.name
                row["base_state_digest"] = base_digest
                row["response_cost_state_digest"] = (
                    hashlib.sha256(
                        f"{base_digest}:{seed}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )
                rows.append(row)
    solved = [row for row in rows if row["pareto_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    pareto_only = [
        row for row in solved if not row["plain_solved"]
    ]
    ratios = [
        float(row["expansion_ratio_lower_bound"])
        for row in solved
    ]
    ladder = {
        str(budget): {
            key: sum(
                int(row["budget_ladder"][str(budget)][key])
                for row in rows
            )
            for key in ("pareto_solved", "plain_solved")
        }
        for budget in BUDGET_LADDER
    }
    dominated_removed = sum(
        int(row["pareto_stats"]["dominated_queries_removed"])
        for row in solved
    )
    root_incomparable_classes = sum(
        int(row["root_pareto_certificate"][
            "incomparable_pareto_classes"
        ])
        for row in rows
    )
    gate = (
        counterexample["passed"]
        and theorem["passed"]
        and theorem["tasks_with_dominance_reduction"] >= 20
        and verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(base_state_digests) == 65
        and len(rows) == 65 * len(PROFILE_SEEDS)
        and len(solved) >= 180
        and len(both) >= 120
        and len(pareto_only) >= 25
        and all(row["matched_if_both"] for row in both)
        and all(
            row["root_pareto_certificate"]["passed"]
            for row in rows
        )
        and dominated_removed >= 1000
        and root_incomparable_classes > 0
        and float(np.median(ratios)) >= 3.0
        and float(np.quantile(ratios, 0.9)) >= 10.0
        and ladder["50000"]["pareto_solved"] >= (
            ladder["50000"]["plain_solved"] + 20
        )
    )
    frozen_digest = hashlib.sha256(json.dumps({
        "budget": BUDGET,
        "budget_ladder": BUDGET_LADDER,
        "profile_seeds": PROFILE_SEEDS,
        "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        "objective": (
            "max_diagnosed_prior_mass_min_expected_response_cost_"
            "min_worst_response_cost"
        ),
        "quotient_rule": (
            "same_unlabelled_partition_retain_componentwise_pareto_"
            "cost_vectors_equal_vector_tie_by_query_index"
        ),
        "scope": (
            "static_positive_hypothesis_mass_static_positive_"
            "response_dependent_test_cost_inherited_remaining_mask"
        ),
        "base_state_digests": sorted(base_state_digests),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": (
            "response_cost_pareto_candidate"
            if gate else "not_yet"
        ),
        "development_gate": gate,
        "claim_scope": (
            "For static positive hypothesis masses, response-dependent "
            "positive test costs and inherited test availability, locally "
            "equivalent tests are quotiented by removing only tests whose "
            "cell-aligned cost vector is componentwise dominated. "
            "Incomparable equivalent tests remain in the search. The study "
            "certifies hereditary dominance and exact optimum preservation. "
            "State-dependent future costs and exogenous availability remain "
            "outside the theorem."
        ),
        "protocol": {
            "budget": BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
            "profile_seeds": list(PROFILE_SEEDS),
            "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        },
        "incomparable_counterexample": counterexample,
        "theorem_certificate": theorem,
        "base_state_count": len(base_state_digests),
        "response_cost_state_count": len(rows),
        "pareto_solved_count": len(solved),
        "plain_solved_count": len(both),
        "pareto_only_solved_count": len(pareto_only),
        "both_plan_match_count": sum(
            int(row["matched_if_both"]) for row in both
        ),
        "dominated_queries_removed": dominated_removed,
        "root_incomparable_pareto_classes": (
            root_incomparable_classes
        ),
        "expansion_ratio_lower_bound_median": (
            float(np.median(ratios)) if ratios else None
        ),
        "expansion_ratio_lower_bound_p90": (
            float(np.quantile(ratios, 0.9))
            if ratios else None
        ),
        "budget_ladder_results": ladder,
        "archive_verification": verification,
        "frozen_response_cost_digest": frozen_digest,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "counterexample_passed": report[
            "incomparable_counterexample"
        ]["passed"],
        "theorem_passed": report[
            "theorem_certificate"
        ]["passed"],
        "states": report["response_cost_state_count"],
        "pareto_solved": report["pareto_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "pareto_only": report["pareto_only_solved_count"],
        "dominated_removed": report[
            "dominated_queries_removed"
        ],
        "median_ratio": report[
            "expansion_ratio_lower_bound_median"
        ],
        "p90_ratio": report[
            "expansion_ratio_lower_bound_p90"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
