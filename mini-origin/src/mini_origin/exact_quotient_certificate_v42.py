from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Iterable

import numpy as np

from . import exact_tail_v36 as v36
from . import external_local_quotient_v41 as v41
from . import local_quotient_v40 as v40
from . import state_policy_v34 as v34


MATCHED_BUDGET = 250_000
CHALLENGE_BUDGET = 50_000
MAX_MATCHED_PER_TASK = 50
MAX_CHALLENGE_PER_TASK = 12
RANDOM_THEOREM_TASKS = 40
SYNTHETIC_CHALLENGES = 32


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class SolverStats:
    calls: int
    cache_states: int
    query_expansions: int
    cache_hits: int
    raw_queries_considered: int
    representative_queries_considered: int


@dataclass(frozen=True)
class SolveResult:
    plan: v36.Plan
    stats: SolverStats


class CountingPlainPlanner:
    """Matched exact DP without state-local query quotienting."""

    def __init__(self, task: object, budget: int) -> None:
        self.task = task
        self.budget = budget
        self.cache: dict[tuple[int, int], v36.Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.cache_hits = 0
        self.raw_queries_considered = 0

    def solve(self, allowed: int, remaining: int) -> v36.Plan:
        self.calls += 1
        key = (allowed, remaining)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = v36.Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        candidates = []
        query_bits = remaining
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            self.raw_queries_considered += 1
            children = tuple(sorted(
                child
                for mask in self.task.masks_for(query).values()
                if (child := allowed & mask)
            ))
            if len(children) <= 1:
                continue
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded("plain exact-search budget exceeded")
            next_remaining = remaining & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in children
            ]
            candidates.append(v36.Plan(
                diagnosed=sum(row.diagnosed for row in child_plans),
                worst_queries=1 + max(
                    row.worst_queries for row in child_plans
                ),
                total_queries=size + sum(
                    row.total_queries for row in child_plans
                ),
                query=query,
            ))
        plan = (
            max(candidates, key=lambda row: row.score())
            if candidates
            else v36.Plan(0, 0, 0, None)
        )
        self.cache[key] = plan
        return plan

    def result(self, allowed: int, remaining: int) -> SolveResult:
        plan = self.solve(allowed, remaining)
        return SolveResult(
            plan,
            SolverStats(
                calls=self.calls,
                cache_states=len(self.cache),
                query_expansions=self.query_expansions,
                cache_hits=self.cache_hits,
                raw_queries_considered=self.raw_queries_considered,
                representative_queries_considered=(
                    self.raw_queries_considered
                ),
            ),
        )


class CountingQuotientPlanner:
    """Exact DP after canonicalising state-local response partitions."""

    def __init__(self, task: object, budget: int) -> None:
        self.task = task
        self.budget = budget
        self.cache: dict[tuple[int, int], v36.Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.cache_hits = 0
        self.raw_queries_considered = 0
        self.representative_queries_considered = 0

    def signature(self, allowed: int, query: int) -> tuple[int, ...]:
        return tuple(sorted(
            child
            for mask in self.task.masks_for(query).values()
            if (child := allowed & mask)
        ))

    def canonical_remaining(self, allowed: int, remaining: int) -> int:
        representatives: dict[tuple[int, ...], int] = {}
        query_bits = remaining
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            self.raw_queries_considered += 1
            signature = self.signature(allowed, query)
            if len(signature) <= 1:
                continue
            previous = representatives.get(signature)
            if previous is None or query < previous:
                representatives[signature] = query
        self.representative_queries_considered += len(representatives)
        canonical = 0
        for query in representatives.values():
            canonical |= 1 << query
        return canonical

    def solve(self, allowed: int, remaining: int) -> v36.Plan:
        self.calls += 1
        canonical = self.canonical_remaining(allowed, remaining)
        key = (allowed, canonical)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = v36.Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        candidates = []
        query_bits = canonical
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            children = self.signature(allowed, query)
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded("quotient exact-search budget exceeded")
            next_remaining = canonical & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in children
            ]
            candidates.append(v36.Plan(
                diagnosed=sum(row.diagnosed for row in child_plans),
                worst_queries=1 + max(
                    row.worst_queries for row in child_plans
                ),
                total_queries=size + sum(
                    row.total_queries for row in child_plans
                ),
                query=query,
            ))
        plan = (
            max(candidates, key=lambda row: row.score())
            if candidates
            else v36.Plan(0, 0, 0, None)
        )
        self.cache[key] = plan
        return plan

    def result(self, allowed: int, remaining: int) -> SolveResult:
        plan = self.solve(allowed, remaining)
        return SolveResult(
            plan,
            SolverStats(
                calls=self.calls,
                cache_states=len(self.cache),
                query_expansions=self.query_expansions,
                cache_hits=self.cache_hits,
                raw_queries_considered=self.raw_queries_considered,
                representative_queries_considered=(
                    self.representative_queries_considered
                ),
            ),
        )


def plan_metrics(plan: v36.Plan) -> tuple[int, int, int]:
    return plan.diagnosed, plan.worst_queries, plan.total_queries


def partition_signature(
    task: object,
    allowed: int,
    query: int,
) -> tuple[int, ...]:
    return tuple(sorted(
        child
        for mask in task.masks_for(query).values()
        if (child := allowed & mask)
    ))


def quotient_representative_count(
    task: object,
    allowed: int,
    remaining: int,
) -> int:
    signatures = set()
    query_bits = remaining
    while query_bits:
        bit = query_bits & -query_bits
        query = bit.bit_length() - 1
        query_bits ^= bit
        signature = partition_signature(task, allowed, query)
        if len(signature) > 1:
            signatures.add(signature)
    return len(signatures)


def descendants(mask: int) -> Iterable[int]:
    subset = mask
    while subset:
        yield subset
        subset = (subset - 1) & mask


def local_equivalence_theorem(task: object) -> dict[str, object]:
    """Exhaustively verify equivalence is hereditary on candidate subsets."""
    comparisons = 0
    descendants_checked = 0
    violations = []
    full = task.full_mask
    for allowed in descendants(full):
        signatures: dict[tuple[int, ...], list[int]] = {}
        for query in range(task.query_count):
            signature = partition_signature(task, allowed, query)
            signatures.setdefault(signature, []).append(query)
        for queries in signatures.values():
            if len(queries) <= 1:
                continue
            for left_index in range(len(queries)):
                for right_index in range(left_index + 1, len(queries)):
                    left = queries[left_index]
                    right = queries[right_index]
                    comparisons += 1
                    for child in descendants(allowed):
                        descendants_checked += 1
                        if (
                            partition_signature(task, child, left)
                            != partition_signature(task, child, right)
                        ):
                            violations.append({
                                "allowed": allowed,
                                "child": child,
                                "left": left,
                                "right": right,
                            })
                            break
    return {
        "comparisons": comparisons,
        "descendants_checked": descendants_checked,
        "violations": violations,
        "passed": not violations,
    }


def random_task(seed: int):
    rng = np.random.default_rng(seed)
    hypotheses = int(rng.integers(5, 9))
    queries = int(rng.integers(4, 8))
    rows = []
    for _ in range(hypotheses):
        rows.append(tuple(
            str(int(rng.integers(0, int(rng.integers(2, 5)))))
            for _ in range(queries)
        ))
    labels = tuple(
        str(int(value))
        for value in rng.integers(0, int(rng.integers(2, 4)), size=hypotheses)
    )
    return v34.base.make_task(
        f"random-theorem-{seed}",
        tuple(f"q{index}" for index in range(queries)),
        rows,
        labels,
    )


def random_theorem_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    for seed in range(4201, 4201 + RANDOM_THEOREM_TASKS):
        task = random_task(seed)
        theorem = local_equivalence_theorem(task)
        remaining = (1 << task.query_count) - 1
        plain = CountingPlainPlanner(task, MATCHED_BUDGET).result(
            task.full_mask,
            remaining,
        )
        quotient = CountingQuotientPlanner(task, MATCHED_BUDGET).result(
            task.full_mask,
            remaining,
        )
        matched = plan_metrics(plain.plan) == plan_metrics(quotient.plan)
        if not matched:
            mismatches.append({
                "seed": seed,
                "plain": plan_metrics(plain.plan),
                "quotient": plan_metrics(quotient.plan),
            })
        rows.append({
            "seed": seed,
            "hypotheses": task.candidate_count,
            "queries": task.query_count,
            "theorem": theorem,
            "plan_matched": matched,
        })
    return {
        "task_count": len(rows),
        "theorem_pass_count": sum(
            int(row["theorem"]["passed"]) for row in rows
        ),
        "plan_match_count": sum(
            int(row["plan_matched"]) for row in rows
        ),
        "violations": [
            row for row in rows if not row["theorem"]["passed"]
        ],
        "plan_mismatches": mismatches,
        "passed": (
            len(rows) == RANDOM_THEOREM_TASKS
            and not mismatches
            and all(row["theorem"]["passed"] for row in rows)
        ),
    }


def collect_policy_states(task: object) -> set[tuple[int, int]]:
    states: set[tuple[int, int]] = set()
    for objective in v34.OBJECTIVE_NAMES:
        stack = [(
            task.full_mask,
            (1 << task.query_count) - 1,
        )]
        seen = set()
        while stack:
            allowed, remaining = stack.pop()
            key = (allowed, remaining)
            if key in seen:
                continue
            seen.add(key)
            if v34.base.pure_label(task, allowed) is not None:
                continue
            states.add(key)
            try:
                query = v34.base.select_query(
                    task,
                    allowed,
                    remaining,
                    v34.OBJECTIVES[objective],
                )
            except RuntimeError:
                continue
            next_remaining = remaining & ~(1 << query)
            for mask in task.masks_for(query).values():
                child = allowed & mask
                if child:
                    stack.append((child, next_remaining))
    return states


def deterministic_rank(
    task_name: str,
    allowed: int,
    remaining: int,
) -> str:
    return hashlib.sha256(
        f"{task_name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()


def evaluate_matched_state(
    task: object,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    plain = CountingPlainPlanner(task, MATCHED_BUDGET).result(
        allowed,
        remaining,
    )
    quotient = CountingQuotientPlanner(task, MATCHED_BUDGET).result(
        allowed,
        remaining,
    )
    matched = plan_metrics(plain.plan) == plan_metrics(quotient.plan)
    expansion_ratio = (
        plain.stats.query_expansions
        / max(1, quotient.stats.query_expansions)
    )
    state_ratio = (
        plain.stats.cache_states
        / max(1, quotient.stats.cache_states)
    )
    return {
        "matched": matched,
        "candidate_count": allowed.bit_count(),
        "raw_remaining_queries": remaining.bit_count(),
        "quotient_representatives": quotient_representative_count(
            task, allowed, remaining
        ),
        "plain_plan": plan_metrics(plain.plan),
        "quotient_plan": plan_metrics(quotient.plan),
        "plain_stats": plain.stats.__dict__,
        "quotient_stats": quotient.stats.__dict__,
        "expansion_ratio": expansion_ratio,
        "state_ratio": state_ratio,
    }


def evaluate_challenge_state(
    task: object,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    quotient_solved = False
    plain_solved = False
    quotient_result = None
    plain_result = None
    try:
        quotient_result = CountingQuotientPlanner(
            task, CHALLENGE_BUDGET
        ).result(allowed, remaining)
        quotient_solved = True
    except BudgetExceeded:
        pass
    try:
        plain_result = CountingPlainPlanner(
            task, CHALLENGE_BUDGET
        ).result(allowed, remaining)
        plain_solved = True
    except BudgetExceeded:
        pass
    matched = (
        quotient_solved
        and plain_solved
        and plan_metrics(quotient_result.plan)
        == plan_metrics(plain_result.plan)
    )
    return {
        "candidate_count": allowed.bit_count(),
        "raw_remaining_queries": remaining.bit_count(),
        "quotient_representatives": quotient_representative_count(
            task, allowed, remaining
        ),
        "quotient_solved": quotient_solved,
        "plain_solved": plain_solved,
        "matched_if_both": matched,
        "quotient_stats": (
            quotient_result.stats.__dict__
            if quotient_result is not None else None
        ),
        "plain_stats": (
            plain_result.stats.__dict__
            if plain_result is not None else None
        ),
    }


def load_all_opened_tasks() -> tuple[list[object], dict[str, object]]:
    base_tasks, v39_verification = v40.opened_tasks()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        v41_verification = v41.download_and_verify(root)
        v41_tasks = v41.load_tasks(root)
    tasks = {task.name: task for task in base_tasks + v41_tasks}
    return [tasks[name] for name in sorted(tasks)], {
        "v39": v39_verification,
        "v41": v41_verification,
    }


def real_state_benchmark(tasks: list[object]) -> dict[str, object]:
    matched_rows = []
    challenge_rows = []
    task_summaries = []
    for task in tasks:
        states = collect_policy_states(task)
        matched_candidates = []
        challenge_candidates = []
        for allowed, remaining in states:
            size = allowed.bit_count()
            raw = remaining.bit_count()
            representatives = quotient_representative_count(
                task, allowed, remaining
            )
            if 2 <= size <= 14 and raw <= 18:
                matched_candidates.append((allowed, remaining))
            if (
                4 <= size <= 16
                and raw >= 16
                and representatives <= 12
                and raw - representatives >= 4
            ):
                challenge_candidates.append((allowed, remaining))
        matched_candidates.sort(
            key=lambda row: deterministic_rank(task.name, *row)
        )
        challenge_candidates.sort(
            key=lambda row: deterministic_rank(
                "challenge-" + task.name, *row
            )
        )
        matched_count = 0
        for allowed, remaining in matched_candidates[:MAX_MATCHED_PER_TASK]:
            try:
                row = evaluate_matched_state(task, allowed, remaining)
            except BudgetExceeded:
                continue
            row["task"] = task.name
            matched_rows.append(row)
            matched_count += 1
        challenge_count = 0
        for allowed, remaining in challenge_candidates[:MAX_CHALLENGE_PER_TASK]:
            row = evaluate_challenge_state(task, allowed, remaining)
            row["task"] = task.name
            challenge_rows.append(row)
            challenge_count += 1
        task_summaries.append({
            "task": task.name,
            "harvested_states": len(states),
            "matched_candidates": len(matched_candidates),
            "matched_evaluated": matched_count,
            "challenge_candidates": len(challenge_candidates),
            "challenge_evaluated": challenge_count,
        })
    ratios = [row["expansion_ratio"] for row in matched_rows]
    state_ratios = [row["state_ratio"] for row in matched_rows]
    challenge_quotient_only = sum(
        int(row["quotient_solved"] and not row["plain_solved"])
        for row in challenge_rows
    )
    both_solved = [
        row for row in challenge_rows
        if row["quotient_solved"] and row["plain_solved"]
    ]
    return {
        "task_count": len(tasks),
        "task_summaries": task_summaries,
        "matched_state_count": len(matched_rows),
        "matched_plan_count": sum(
            int(row["matched"]) for row in matched_rows
        ),
        "plan_mismatch_count": sum(
            int(not row["matched"]) for row in matched_rows
        ),
        "expansion_ratio_median": (
            float(np.median(ratios)) if ratios else None
        ),
        "expansion_ratio_p90": (
            float(np.quantile(ratios, 0.9)) if ratios else None
        ),
        "state_ratio_median": (
            float(np.median(state_ratios)) if state_ratios else None
        ),
        "challenge_state_count": len(challenge_rows),
        "challenge_quotient_only_solved": challenge_quotient_only,
        "challenge_both_solved": len(both_solved),
        "challenge_both_plan_matches": sum(
            int(row["matched_if_both"]) for row in both_solved
        ),
        "matched_rows": matched_rows,
        "challenge_rows": challenge_rows,
    }


def duplicate_query_task(seed: int):
    rng = np.random.default_rng(seed)
    hypotheses = int(rng.integers(10, 17))
    base_queries = int(rng.integers(6, 9))
    repeats = int(rng.integers(5, 8))
    base_columns = [
        tuple(str(int(value)) for value in rng.integers(0, 3, size=hypotheses))
        for _ in range(base_queries)
    ]
    columns = []
    names = []
    for index, column in enumerate(base_columns):
        for repeat in range(repeats):
            columns.append(column)
            names.append(f"q{index}-copy{repeat}")
    rows = [
        tuple(column[row] for column in columns)
        for row in range(hypotheses)
    ]
    labels = tuple(
        str(int(value))
        for value in rng.integers(0, 3, size=hypotheses)
    )
    return v34.base.make_task(
        f"duplicate-challenge-{seed}",
        tuple(names),
        rows,
        labels,
    )


def synthetic_challenge_benchmark() -> dict[str, object]:
    rows = []
    for seed in range(5201, 5201 + SYNTHETIC_CHALLENGES):
        task = duplicate_query_task(seed)
        row = evaluate_challenge_state(
            task,
            task.full_mask,
            (1 << task.query_count) - 1,
        )
        row["seed"] = seed
        row["task"] = task.name
        rows.append(row)
    quotient_only = sum(
        int(row["quotient_solved"] and not row["plain_solved"])
        for row in rows
    )
    both = [
        row for row in rows
        if row["quotient_solved"] and row["plain_solved"]
    ]
    return {
        "task_count": len(rows),
        "quotient_solved_count": sum(
            int(row["quotient_solved"]) for row in rows
        ),
        "plain_solved_count": sum(
            int(row["plain_solved"]) for row in rows
        ),
        "quotient_only_solved": quotient_only,
        "both_solved_count": len(both),
        "both_plan_matches": sum(
            int(row["matched_if_both"]) for row in both
        ),
        "rows": rows,
    }


def run() -> dict[str, object]:
    theorem = random_theorem_certificate()
    tasks, verification = load_all_opened_tasks()
    real = real_state_benchmark(tasks)
    synthetic = synthetic_challenge_benchmark()
    gate = (
        theorem["passed"]
        and verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and real["matched_state_count"] >= 500
        and real["plan_mismatch_count"] == 0
        and real["matched_plan_count"] == real["matched_state_count"]
        and real["expansion_ratio_median"] >= 1.25
        and real["expansion_ratio_p90"] >= 2.0
        and real["challenge_quotient_only_solved"] >= 5
        and real["challenge_both_plan_matches"] == real["challenge_both_solved"]
        and synthetic["quotient_solved_count"] == SYNTHETIC_CHALLENGES
        and synthetic["quotient_only_solved"] >= 24
        and synthetic["both_plan_matches"] == synthetic["both_solved_count"]
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "budgets": {
                    "matched": MATCHED_BUDGET,
                    "challenge": CHALLENGE_BUDGET,
                },
                "sampling": {
                    "matched_per_task": MAX_MATCHED_PER_TASK,
                    "challenge_per_task": MAX_CHALLENGE_PER_TASK,
                    "random_theorem_tasks": RANDOM_THEOREM_TASKS,
                    "synthetic_challenges": SYNTHETIC_CHALLENGES,
                },
                "task_names": [task.name for task in tasks],
                "verification": verification,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "exact_local_quotient_certificate_candidate"
            if gate else "not_yet"
        ),
        "claim_scope": (
            "state-local response-partition quotienting is exhaustively checked "
            "for hereditary equivalence on random finite decision tables and is "
            "benchmarked against a matched unquotiented global exact solver on "
            "real UCI-derived states under deterministic operation budgets; a "
            "pass is an exact algorithmic certificate, but novelty relative to "
            "all prior identification-tree solvers still requires external review"
        ),
        "development_gate": gate,
        "protocol": {
            "matched_budget": MATCHED_BUDGET,
            "challenge_budget": CHALLENGE_BUDGET,
            "max_matched_per_task": MAX_MATCHED_PER_TASK,
            "max_challenge_per_task": MAX_CHALLENGE_PER_TASK,
            "random_theorem_tasks": RANDOM_THEOREM_TASKS,
            "synthetic_challenges": SYNTHETIC_CHALLENGES,
        },
        "theorem_certificate": theorem,
        "real_state_benchmark": real,
        "synthetic_challenge_benchmark": synthetic,
        "archive_verification": verification,
        "frozen_benchmark_digest": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    real = report["real_state_benchmark"]
    synthetic = report["synthetic_challenge_benchmark"]
    print(json.dumps({
        "status": report["status"],
        "theorem_passed": report["theorem_certificate"]["passed"],
        "matched_states": real["matched_state_count"],
        "plan_mismatches": real["plan_mismatch_count"],
        "median_expansion_ratio": real["expansion_ratio_median"],
        "p90_expansion_ratio": real["expansion_ratio_p90"],
        "real_quotient_only": real["challenge_quotient_only_solved"],
        "synthetic_quotient_only": synthetic["quotient_only_solved"],
    }, indent=2))


if __name__ == "__main__":
    main()
