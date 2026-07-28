from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import state_policy_v34 as state


BUDGET = 250_000
BUDGET_LADDER = (10_000, 50_000, 250_000)
PROFILE_SEEDS = (4801, 4802, 4803)
RANDOM_THEOREM_TASKS = 48


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightedProfile:
    hypothesis_mass: tuple[int, ...]
    query_cost: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class WeightedPlan:
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


@dataclass(frozen=True)
class SolveResult:
    plan: WeightedPlan
    stats: SolverStats


def plan_score(plan: WeightedPlan) -> tuple[int, int, int, int]:
    return (
        plan.diagnosed_mass,
        -plan.expected_cost_numerator,
        -plan.worst_cost,
        -(plan.query if plan.query is not None else 10**9),
    )


def plan_metrics(plan: WeightedPlan) -> tuple[int, int, int]:
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


def subset_mass(profile: WeightedProfile, allowed: int) -> int:
    total = 0
    pending = allowed
    while pending:
        bit = pending & -pending
        index = bit.bit_length() - 1
        pending ^= bit
        total += profile.hypothesis_mass[index]
    return total


def stable_positive(seed: int, token: str, low: int, high: int) -> int:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return low + int.from_bytes(digest[:8], "big") % (high - low + 1)


def profile_for_task(task: object, seed: int) -> WeightedProfile:
    masses = tuple(
        stable_positive(seed, f"{task.name}:hypothesis:{index}", 1, 31)
        for index in range(task.candidate_count)
    )
    costs = tuple(
        stable_positive(seed, f"{task.name}:query:{index}", 1, 13)
        for index in range(task.query_count)
    )
    return WeightedProfile(masses, costs, seed)


def profile_digest(task: object, profile: WeightedProfile) -> str:
    return hashlib.sha256(json.dumps({
        "task": task.name,
        "seed": profile.seed,
        "hypothesis_mass": profile.hypothesis_mass,
        "query_cost": profile.query_cost,
    }, sort_keys=True).encode("utf-8")).hexdigest()


def representative_map(
    task: object,
    profile: WeightedProfile,
    allowed: int,
    remaining: int,
) -> dict[tuple[int, ...], int]:
    representatives: dict[tuple[int, ...], int] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = partition(task, allowed, query)
        if len(signature) <= 1:
            continue
        previous = representatives.get(signature)
        if previous is None or (
            profile.query_cost[query], query
        ) < (
            profile.query_cost[previous], previous
        ):
            representatives[signature] = query
    return representatives


def representative_mask(
    task: object,
    profile: WeightedProfile,
    allowed: int,
    remaining: int,
) -> int:
    mask = 0
    for query in representative_map(
        task, profile, allowed, remaining
    ).values():
        mask |= 1 << query
    return mask


def representative_certificate(
    task: object,
    profile: WeightedProfile,
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
    selected = representative_map(task, profile, allowed, remaining)
    passed = True
    cost_sensitive_changes = 0
    duplicate_classes = 0
    for signature, queries in groups.items():
        if len(queries) > 1:
            duplicate_classes += 1
        optimum = min(queries, key=lambda query: (
            profile.query_cost[query], query
        ))
        if selected[signature] != optimum:
            passed = False
        if len(queries) > 1 and optimum != min(queries):
            cost_sensitive_changes += 1
    return {
        "passed": passed,
        "separating_classes": len(groups),
        "duplicate_classes": duplicate_classes,
        "cost_sensitive_representative_changes": cost_sensitive_changes,
    }


class PlainWeightedPlanner:
    def __init__(
        self,
        task: object,
        profile: WeightedProfile,
        budget: int,
    ) -> None:
        self.task = task
        self.profile = profile
        self.budget = budget
        self.memo: dict[tuple[int, int], WeightedPlan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0
        self.raw_queries_considered = 0

    def solve(self, allowed: int, remaining: int) -> WeightedPlan:
        self.calls += 1
        key = allowed, remaining
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        mass = subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = WeightedPlan(mass, 0, 0, None)
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
                raise BudgetExceeded("plain weighted budget exceeded")
            child_plans = [
                self.solve(child, remaining & ~(1 << query))
                for child in children
            ]
            cost = self.profile.query_cost[query]
            candidates.append(WeightedPlan(
                diagnosed_mass=sum(
                    row.diagnosed_mass for row in child_plans
                ),
                expected_cost_numerator=(
                    cost * mass
                    + sum(
                        row.expected_cost_numerator
                        for row in child_plans
                    )
                ),
                worst_cost=cost + max(
                    row.worst_cost for row in child_plans
                ),
                query=query,
            ))
        answer = (
            max(candidates, key=plan_score)
            if candidates else WeightedPlan(0, 0, 0, None)
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
            raw_queries_considered=self.raw_queries_considered,
            representative_queries_considered=(
                self.raw_queries_considered
            ),
        ))


class GlobalWeightedPlanner:
    def __init__(
        self,
        task: object,
        profile: WeightedProfile,
        root_allowed: int,
        root_remaining: int,
        budget: int,
    ) -> None:
        self.task = task
        self.profile = profile
        self.budget = budget
        self.root_mask = representative_mask(
            task, profile, root_allowed, root_remaining
        )
        self.memo: dict[tuple[int, int], WeightedPlan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0
        self.raw_queries_considered = 0

    def solve(self, allowed: int, remaining: int) -> WeightedPlan:
        self.calls += 1
        available = remaining & self.root_mask
        key = allowed, available
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        mass = subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = WeightedPlan(mass, 0, 0, None)
            self.memo[key] = answer
            return answer
        candidates = []
        pending = available
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
                raise BudgetExceeded("global weighted budget exceeded")
            child_plans = [
                self.solve(child, available & ~(1 << query))
                for child in children
            ]
            cost = self.profile.query_cost[query]
            candidates.append(WeightedPlan(
                diagnosed_mass=sum(
                    row.diagnosed_mass for row in child_plans
                ),
                expected_cost_numerator=(
                    cost * mass
                    + sum(
                        row.expected_cost_numerator
                        for row in child_plans
                    )
                ),
                worst_cost=cost + max(
                    row.worst_cost for row in child_plans
                ),
                query=query,
            ))
        answer = (
            max(candidates, key=plan_score)
            if candidates else WeightedPlan(0, 0, 0, None)
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
            raw_queries_considered=self.raw_queries_considered,
            representative_queries_considered=(
                self.raw_queries_considered
            ),
        ))


class LocalWeightedPlanner:
    def __init__(
        self,
        task: object,
        profile: WeightedProfile,
        budget: int,
    ) -> None:
        self.task = task
        self.profile = profile
        self.budget = budget
        self.memo: dict[tuple[int, int], WeightedPlan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0
        self.raw_queries_considered = 0
        self.representative_queries_considered = 0

    def canonical(self, allowed: int, remaining: int) -> int:
        self.raw_queries_considered += remaining.bit_count()
        mask = representative_mask(
            self.task, self.profile, allowed, remaining
        )
        self.representative_queries_considered += mask.bit_count()
        return mask

    def solve(self, allowed: int, remaining: int) -> WeightedPlan:
        self.calls += 1
        canonical = self.canonical(allowed, remaining)
        key = allowed, canonical
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        mass = subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = WeightedPlan(mass, 0, 0, None)
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
                raise BudgetExceeded("local weighted budget exceeded")
            child_plans = [
                self.solve(child, canonical & ~(1 << query))
                for child in children
            ]
            cost = self.profile.query_cost[query]
            candidates.append(WeightedPlan(
                diagnosed_mass=sum(
                    row.diagnosed_mass for row in child_plans
                ),
                expected_cost_numerator=(
                    cost * mass
                    + sum(
                        row.expected_cost_numerator
                        for row in child_plans
                    )
                ),
                worst_cost=cost + max(
                    row.worst_cost for row in child_plans
                ),
                query=query,
            ))
        answer = (
            max(candidates, key=plan_score)
            if candidates else WeightedPlan(0, 0, 0, None)
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
            raw_queries_considered=self.raw_queries_considered,
            representative_queries_considered=(
                self.representative_queries_considered
            ),
        ))


def expected_elimination_numerator(
    task: object,
    profile: WeightedProfile,
    allowed: int,
    query: int,
) -> int:
    mass = subset_mass(profile, allowed)
    return mass * mass - sum(
        subset_mass(profile, child) ** 2
        for child in partition(task, allowed, query)
    )


def select_weighted_greedy_query(
    task: object,
    profile: WeightedProfile,
    allowed: int,
    remaining: int,
) -> int:
    candidates = []
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        if len(partition(task, allowed, query)) > 1:
            candidates.append(query)
    if not candidates:
        raise RuntimeError("no separating query")
    return max(candidates, key=lambda query: (
        Fraction(
            expected_elimination_numerator(
                task, profile, allowed, query
            ),
            profile.query_cost[query],
        ),
        -profile.query_cost[query],
        -query,
    ))


class WeightedGreedy:
    def __init__(self, task: object, profile: WeightedProfile) -> None:
        self.task = task
        self.profile = profile
        self.memo: dict[tuple[int, int], WeightedPlan] = {}

    def solve(self, allowed: int, remaining: int) -> WeightedPlan:
        key = allowed, remaining
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        mass = subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = WeightedPlan(mass, 0, 0, None)
            self.memo[key] = answer
            return answer
        try:
            query = select_weighted_greedy_query(
                self.task, self.profile, allowed, remaining
            )
        except RuntimeError:
            answer = WeightedPlan(0, 0, 0, None)
            self.memo[key] = answer
            return answer
        children = partition(self.task, allowed, query)
        child_plans = [
            self.solve(child, remaining & ~(1 << query))
            for child in children
        ]
        cost = self.profile.query_cost[query]
        answer = WeightedPlan(
            diagnosed_mass=sum(
                row.diagnosed_mass for row in child_plans
            ),
            expected_cost_numerator=(
                cost * mass
                + sum(
                    row.expected_cost_numerator
                    for row in child_plans
                )
            ),
            worst_cost=cost + max(
                row.worst_cost for row in child_plans
            ),
            query=query,
        )
        self.memo[key] = answer
        return answer


def descendants(mask: int):
    subset = mask
    while subset:
        yield subset
        subset = (subset - 1) & mask


def hereditary_cost_theorem(
    task: object,
    profile: WeightedProfile,
) -> dict[str, object]:
    comparisons = 0
    descendant_checks = 0
    violations = []
    for allowed in descendants(task.full_mask):
        groups: dict[tuple[int, ...], list[int]] = {}
        for query in range(task.query_count):
            groups.setdefault(
                partition(task, allowed, query), []
            ).append(query)
        for signature, queries in groups.items():
            if len(signature) <= 1 or len(queries) <= 1:
                continue
            cheapest = min(queries, key=lambda query: (
                profile.query_cost[query], query
            ))
            for query in queries:
                comparisons += 1
                if profile.query_cost[cheapest] > profile.query_cost[query]:
                    violations.append({
                        "kind": "cheapest-order",
                        "allowed": allowed,
                        "cheapest": cheapest,
                        "query": query,
                    })
                for child in descendants(allowed):
                    descendant_checks += 1
                    if partition(task, child, cheapest) != partition(
                        task, child, query
                    ):
                        violations.append({
                            "kind": "hereditary-partition",
                            "allowed": allowed,
                            "child": child,
                            "cheapest": cheapest,
                            "query": query,
                        })
                        break
    return {
        "comparisons": comparisons,
        "descendant_checks": descendant_checks,
        "violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations,
    }


def random_weighted_task(seed: int):
    rng = random.Random(seed)
    hypotheses = rng.randint(5, 8)
    base_queries = rng.randint(3, 6)
    base_columns = [
        [str(rng.randrange(rng.randint(2, 4))) for _ in range(hypotheses)]
        for _ in range(base_queries)
    ]
    duplicate_source = rng.randrange(base_queries)
    columns = base_columns + [list(base_columns[duplicate_source])]
    rows = [
        tuple(columns[query][row] for query in range(len(columns)))
        for row in range(hypotheses)
    ]
    labels = tuple(str(rng.randrange(rng.randint(2, 4))) for _ in rows)
    task = state.base.make_task(
        f"weighted-theorem-{seed}",
        tuple(f"q{index}" for index in range(len(columns))),
        rows,
        labels,
    )
    masses = tuple(rng.randint(1, 19) for _ in rows)
    costs = [rng.randint(1, 11) for _ in columns]
    costs[-1] = max(1, costs[duplicate_source] - rng.randint(0, 3))
    profile = WeightedProfile(tuple(masses), tuple(costs), seed)
    return task, profile


def random_theorem_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    for seed in range(48001, 48001 + RANDOM_THEOREM_TASKS):
        task, profile = random_weighted_task(seed)
        remaining = (1 << task.query_count) - 1
        theorem = hereditary_cost_theorem(task, profile)
        plain = PlainWeightedPlanner(task, profile, 500_000).result(
            task.full_mask, remaining
        )
        local = LocalWeightedPlanner(task, profile, 500_000).result(
            task.full_mask, remaining
        )
        matched = plan_metrics(plain.plan) == plan_metrics(local.plan)
        if not matched:
            mismatches.append({
                "seed": seed,
                "plain": plan_metrics(plain.plan),
                "local": plan_metrics(local.plan),
            })
        rows.append({
            "seed": seed,
            "hypotheses": task.candidate_count,
            "queries": task.query_count,
            "theorem_passed": theorem["passed"],
            "plan_matched": matched,
            "plain_expansions": plain.stats.query_expansions,
            "local_expansions": local.stats.query_expansions,
        })
    return {
        "task_count": len(rows),
        "theorem_pass_count": sum(
            int(row["theorem_passed"]) for row in rows
        ),
        "plan_match_count": sum(
            int(row["plan_matched"]) for row in rows
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
    profile: WeightedProfile,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    local_result = None
    global_result = None
    plain_result = None
    local_solved = False
    global_solved = False
    plain_solved = False
    try:
        local_result = LocalWeightedPlanner(
            task, profile, BUDGET
        ).result(allowed, remaining)
        local_solved = True
    except BudgetExceeded:
        pass
    try:
        global_result = GlobalWeightedPlanner(
            task, profile, allowed, remaining, BUDGET
        ).result(allowed, remaining)
        global_solved = True
    except BudgetExceeded:
        pass
    try:
        plain_result = PlainWeightedPlanner(
            task, profile, BUDGET
        ).result(allowed, remaining)
        plain_solved = True
    except BudgetExceeded:
        pass
    greedy = WeightedGreedy(task, profile).solve(allowed, remaining)
    local_metrics = (
        plan_metrics(local_result.plan) if local_result else None
    )
    global_match = (
        local_solved and global_solved
        and local_metrics == plan_metrics(global_result.plan)
    )
    plain_match = (
        local_solved and plain_solved
        and local_metrics == plan_metrics(plain_result.plan)
    )
    exact_dominates_greedy = (
        local_solved
        and plan_score(local_result.plan) >= plan_score(greedy)
    )
    strict_exact_gain = (
        local_solved
        and plan_score(local_result.plan) > plan_score(greedy)
    )
    equal_diagnosis = (
        local_solved
        and local_result.plan.diagnosed_mass == greedy.diagnosed_mass
    )
    expected_cost_saving = (
        greedy.expected_cost_numerator
        - local_result.plan.expected_cost_numerator
        if equal_diagnosis else None
    )
    if local_solved:
        plain_lower = (
            plain_result.stats.query_expansions
            if plain_solved else BUDGET + 1
        )
        global_lower = (
            global_result.stats.query_expansions
            if global_solved else BUDGET + 1
        )
        plain_ratio = plain_lower / max(
            1, local_result.stats.query_expansions
        )
        global_ratio = global_lower / max(
            1, local_result.stats.query_expansions
        )
    else:
        plain_ratio = None
        global_ratio = None
    ladder = {}
    for budget in BUDGET_LADDER:
        ladder[str(budget)] = {
            "local_solved": (
                local_solved
                and local_result.stats.query_expansions <= budget
            ),
            "global_solved": (
                global_solved
                and global_result.stats.query_expansions <= budget
            ),
            "plain_solved": (
                plain_solved
                and plain_result.stats.query_expansions <= budget
            ),
        }
    certificate = representative_certificate(
        task, profile, allowed, remaining
    )
    return {
        "profile_seed": profile.seed,
        "profile_digest": profile_digest(task, profile),
        "candidate_count": allowed.bit_count(),
        "candidate_mass": subset_mass(profile, allowed),
        "raw_remaining_queries": remaining.bit_count(),
        "root_representative_certificate": certificate,
        "local_solved": local_solved,
        "global_solved": global_solved,
        "plain_solved": plain_solved,
        "local_global_match": global_match,
        "local_plain_match": plain_match,
        "local_plan": local_metrics,
        "global_plan": (
            plan_metrics(global_result.plan) if global_result else None
        ),
        "plain_plan": (
            plan_metrics(plain_result.plan) if plain_result else None
        ),
        "greedy_plan": plan_metrics(greedy),
        "exact_dominates_greedy": exact_dominates_greedy,
        "strict_exact_gain": strict_exact_gain,
        "equal_diagnosis": equal_diagnosis,
        "expected_cost_saving": expected_cost_saving,
        "local_stats": (
            local_result.stats.__dict__ if local_result else None
        ),
        "global_stats": (
            global_result.stats.__dict__ if global_result else None
        ),
        "plain_stats": (
            plain_result.stats.__dict__ if plain_result else None
        ),
        "plain_expansion_ratio_lower_bound": plain_ratio,
        "global_expansion_ratio_lower_bound": global_ratio,
        "budget_ladder": ladder,
    }


def run() -> dict[str, object]:
    theorem = random_theorem_certificate()
    tasks, verification = corpus.load_all_opened_tasks()
    rows = []
    base_state_digests = set()
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            base_digest = hashlib.sha256(
                f"{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_state_digests.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = profile_for_task(task, seed)
                row = evaluate_state(
                    task, profile, allowed, remaining
                )
                row["task"] = task.name
                row["base_state_digest"] = base_digest
                row["weighted_state_digest"] = hashlib.sha256(
                    f"{base_digest}:{seed}".encode("utf-8")
                ).hexdigest()
                rows.append(row)
    local = [row for row in rows if row["local_solved"]]
    local_plain = [row for row in local if row["plain_solved"]]
    local_global = [row for row in local if row["global_solved"]]
    local_only_plain = [
        row for row in local if not row["plain_solved"]
    ]
    local_only_global = [
        row for row in local if not row["global_solved"]
    ]
    plain_ratios = [
        float(row["plain_expansion_ratio_lower_bound"])
        for row in local
    ]
    global_ratios = [
        float(row["global_expansion_ratio_lower_bound"])
        for row in local
    ]
    savings = [
        int(row["expected_cost_saving"])
        for row in local
        if row["expected_cost_saving"] is not None
    ]
    ladder = {
        str(budget): {
            key: sum(
                int(row["budget_ladder"][str(budget)][key])
                for row in rows
            )
            for key in (
                "local_solved", "global_solved", "plain_solved"
            )
        }
        for budget in BUDGET_LADDER
    }
    cost_sensitive_changes = sum(
        int(row["root_representative_certificate"][
            "cost_sensitive_representative_changes"
        ])
        for row in rows
    )
    gate = (
        theorem["passed"]
        and verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(base_state_digests) == 65
        and len(rows) == 65 * len(PROFILE_SEEDS)
        and len(local) == len(rows)
        and len(local_plain) >= 120
        and len(local_only_plain) >= 30
        and len(local_global) >= 150
        and len(local_only_global) >= 15
        and all(row["local_plain_match"] for row in local_plain)
        and all(row["local_global_match"] for row in local_global)
        and all(row["exact_dominates_greedy"] for row in local)
        and sum(int(row["strict_exact_gain"]) for row in local) >= 100
        and sum(savings) >= 100
        and all(
            row["root_representative_certificate"]["passed"]
            for row in rows
        )
        and cost_sensitive_changes > 0
        and float(np.median(plain_ratios)) >= 5.0
        and float(np.quantile(plain_ratios, 0.9)) >= 20.0
        and float(np.median(global_ratios)) >= 2.0
        and float(np.quantile(global_ratios, 0.9)) >= 5.0
        and ladder["50000"]["local_solved"] >= (
            ladder["50000"]["plain_solved"] + 30
        )
        and ladder["50000"]["local_solved"] >= (
            ladder["50000"]["global_solved"] + 20
        )
    )
    frozen_digest = hashlib.sha256(json.dumps({
        "budget": BUDGET,
        "budget_ladder": BUDGET_LADDER,
        "profile_seeds": PROFILE_SEEDS,
        "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        "objective": (
            "max_diagnosed_prior_mass_min_expected_test_cost_"
            "min_worst_test_cost"
        ),
        "equivalence_rule": (
            "same_unlabelled_response_partition_keep_min_cost_then_index"
        ),
        "scope": (
            "static_positive_integer_hypothesis_mass_static_positive_"
            "integer_test_cost_inherited_remaining_test_mask"
        ),
        "base_state_digests": sorted(base_state_digests),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": (
            "cost_prior_quotient_candidate" if gate else "not_yet"
        ),
        "development_gate": gate,
        "claim_scope": (
            "For static positive test costs, positive hypothesis masses and an "
            "inherited remaining-test mask, tests inducing the same response "
            "partition on the current candidate state are quotiented by retaining "
            "the cheapest representative. The study verifies hereditary "
            "equivalence, exact optimum preservation, weighted greedy dominance "
            "and exact-search frontier extension. State-dependent future costs, "
            "response-specific costs and exogenous descendant availability are "
            "outside this theorem."
        ),
        "protocol": {
            "budget": BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
            "profile_seeds": list(PROFILE_SEEDS),
            "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        },
        "theorem_certificate": theorem,
        "base_state_count": len(base_state_digests),
        "weighted_state_count": len(rows),
        "local_solved_count": len(local),
        "plain_solved_count": len(local_plain),
        "local_only_vs_plain_count": len(local_only_plain),
        "global_solved_count": len(local_global),
        "local_only_vs_global_count": len(local_only_global),
        "local_plain_match_count": sum(
            int(row["local_plain_match"]) for row in local_plain
        ),
        "local_global_match_count": sum(
            int(row["local_global_match"]) for row in local_global
        ),
        "exact_dominates_greedy_count": sum(
            int(row["exact_dominates_greedy"]) for row in local
        ),
        "strict_exact_gain_count": sum(
            int(row["strict_exact_gain"]) for row in local
        ),
        "aggregate_expected_cost_saving_vs_greedy": sum(savings),
        "cost_sensitive_representative_changes": cost_sensitive_changes,
        "plain_expansion_ratio_lower_bound_median": float(
            np.median(plain_ratios)
        ),
        "plain_expansion_ratio_lower_bound_p90": float(
            np.quantile(plain_ratios, 0.9)
        ),
        "global_expansion_ratio_lower_bound_median": float(
            np.median(global_ratios)
        ),
        "global_expansion_ratio_lower_bound_p90": float(
            np.quantile(global_ratios, 0.9)
        ),
        "budget_ladder_results": ladder,
        "archive_verification": verification,
        "frozen_weighted_digest": frozen_digest,
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
        "theorem_passed": report["theorem_certificate"]["passed"],
        "weighted_states": report["weighted_state_count"],
        "local_solved": report["local_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "global_solved": report["global_solved_count"],
        "strict_gains": report["strict_exact_gain_count"],
        "expected_cost_saving": report[
            "aggregate_expected_cost_saving_vs_greedy"
        ],
        "plain_median_ratio": report[
            "plain_expansion_ratio_lower_bound_median"
        ],
        "global_median_ratio": report[
            "global_expansion_ratio_lower_bound_median"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
