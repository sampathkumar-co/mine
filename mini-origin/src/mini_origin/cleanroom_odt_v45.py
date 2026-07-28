from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# Data loading and the preregistered state harvester are reused as corpus
# infrastructure only. No exact or greedy planner from an earlier version is
# imported or called by this module.
from . import exact_quotient_certificate_v42 as corpus


EXPANSION_BUDGET = 250_000
MAX_STATES_PER_TASK = 15
MIN_CANDIDATES = 8
MAX_CANDIDATES = 24
MIN_RAW_QUERIES = 18
MAX_PARTITIONS = 18
MIN_REDUNDANCY = 6
BUDGET_LADDER = (10_000, 50_000, 250_000)


class SearchBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class IndependentPlan:
    diagnosed: int
    total_cost: int
    maximum_depth: int
    first_test: int | None

    def ordering_key(self) -> tuple[int, int, int, int]:
        return (
            self.diagnosed,
            -self.total_cost,
            -self.maximum_depth,
            -(self.first_test if self.first_test is not None else 10**9),
        )

    def metrics(self) -> tuple[int, int, int]:
        return self.diagnosed, self.total_cost, self.maximum_depth


@dataclass(frozen=True)
class IndependentStats:
    recursive_calls: int
    memo_entries: int
    expanded_tests: int
    memo_hits: int
    raw_tests_scanned: int
    partition_classes_scanned: int


@dataclass(frozen=True)
class IndependentResult:
    plan: IndependentPlan
    stats: IndependentStats


def is_label_pure(task: object, candidates: int) -> bool:
    if not candidates:
        return True
    containing_labels = 0
    for label_mask in task.label_mask_dict().values():
        if candidates & label_mask:
            containing_labels += 1
            if containing_labels > 1:
                return False
    return True


def response_partition(
    task: object,
    candidates: int,
    test: int,
) -> tuple[int, ...]:
    blocks = []
    for outcome_mask in task.masks_for(test).values():
        block = candidates & outcome_mask
        if block:
            blocks.append(block)
    blocks.sort()
    return tuple(blocks)


def canonical_test_mask(
    task: object,
    candidates: int,
    tests: int,
) -> tuple[int, int, int]:
    """Return one smallest-index test per non-trivial response partition."""
    representative_for: dict[tuple[int, ...], int] = {}
    raw_scanned = 0
    pending = tests
    while pending:
        bit = pending & -pending
        test = bit.bit_length() - 1
        pending ^= bit
        raw_scanned += 1
        partition = response_partition(task, candidates, test)
        if len(partition) <= 1:
            continue
        previous = representative_for.get(partition)
        if previous is None or test < previous:
            representative_for[partition] = test
    canonical = 0
    for test in representative_for.values():
        canonical |= 1 << test
    return canonical, raw_scanned, len(representative_for)


class IndependentQuotientSolver:
    """Clean-room dynamic program for the uniform-prior identification cost."""

    def __init__(self, task: object, expansion_budget: int) -> None:
        self.task = task
        self.expansion_budget = expansion_budget
        self.memo: dict[tuple[int, int], IndependentPlan] = {}
        self.recursive_calls = 0
        self.expanded_tests = 0
        self.memo_hits = 0
        self.raw_tests_scanned = 0
        self.partition_classes_scanned = 0

    def _solve(self, candidates: int, tests: int) -> IndependentPlan:
        self.recursive_calls += 1
        canonical, raw, classes = canonical_test_mask(
            self.task, candidates, tests
        )
        self.raw_tests_scanned += raw
        self.partition_classes_scanned += classes
        key = candidates, canonical
        known = self.memo.get(key)
        if known is not None:
            self.memo_hits += 1
            return known

        population = candidates.bit_count()
        if is_label_pure(self.task, candidates):
            answer = IndependentPlan(population, 0, 0, None)
            self.memo[key] = answer
            return answer

        alternatives: list[IndependentPlan] = []
        pending = canonical
        while pending:
            bit = pending & -pending
            test = bit.bit_length() - 1
            pending ^= bit
            self.expanded_tests += 1
            if self.expanded_tests > self.expansion_budget:
                raise SearchBudgetExceeded(
                    "clean-room exact-search expansion budget exceeded"
                )
            children = response_partition(self.task, candidates, test)
            child_tests = canonical & ~(1 << test)
            child_plans = [
                self._solve(child, child_tests) for child in children
            ]
            alternatives.append(IndependentPlan(
                diagnosed=sum(plan.diagnosed for plan in child_plans),
                total_cost=population + sum(
                    plan.total_cost for plan in child_plans
                ),
                maximum_depth=1 + max(
                    plan.maximum_depth for plan in child_plans
                ),
                first_test=test,
            ))

        answer = (
            max(alternatives, key=IndependentPlan.ordering_key)
            if alternatives
            else IndependentPlan(0, 0, 0, None)
        )
        self.memo[key] = answer
        return answer

    def solve(self, candidates: int, tests: int) -> IndependentResult:
        plan = self._solve(candidates, tests)
        return IndependentResult(
            plan=plan,
            stats=IndependentStats(
                recursive_calls=self.recursive_calls,
                memo_entries=len(self.memo),
                expanded_tests=self.expanded_tests,
                memo_hits=self.memo_hits,
                raw_tests_scanned=self.raw_tests_scanned,
                partition_classes_scanned=self.partition_classes_scanned,
            ),
        )


def expected_eliminations(
    task: object,
    candidates: int,
    test: int,
) -> int:
    population = candidates.bit_count()
    return sum(
        block.bit_count() * (population - block.bit_count())
        for block in response_partition(task, candidates, test)
    )


def expected_elimination_test(
    task: object,
    candidates: int,
    tests: int,
) -> int:
    choices = []
    pending = tests
    while pending:
        bit = pending & -pending
        test = bit.bit_length() - 1
        pending ^= bit
        if len(response_partition(task, candidates, test)) > 1:
            choices.append(test)
    if not choices:
        raise RuntimeError("no separating test")
    return max(
        choices,
        key=lambda test: (expected_eliminations(task, candidates, test), -test),
    )


class IndependentGreedySolver:
    def __init__(self, task: object) -> None:
        self.task = task
        self.memo: dict[tuple[int, int], IndependentPlan] = {}

    def solve(self, candidates: int, tests: int) -> IndependentPlan:
        key = candidates, tests
        known = self.memo.get(key)
        if known is not None:
            return known
        population = candidates.bit_count()
        if is_label_pure(self.task, candidates):
            answer = IndependentPlan(population, 0, 0, None)
            self.memo[key] = answer
            return answer
        try:
            test = expected_elimination_test(self.task, candidates, tests)
        except RuntimeError:
            answer = IndependentPlan(0, 0, 0, None)
            self.memo[key] = answer
            return answer
        child_tests = tests & ~(1 << test)
        children = [
            self.solve(block, child_tests)
            for block in response_partition(self.task, candidates, test)
        ]
        answer = IndependentPlan(
            diagnosed=sum(plan.diagnosed for plan in children),
            total_cost=population + sum(plan.total_cost for plan in children),
            maximum_depth=1 + max(plan.maximum_depth for plan in children),
            first_test=test,
        )
        self.memo[key] = answer
        return answer


def deterministic_state_rank(
    task_name: str,
    candidates: int,
    tests: int,
    partition_classes: int,
) -> tuple[int, int, int, str]:
    return (
        -candidates.bit_count(),
        -partition_classes,
        -tests.bit_count(),
        hashlib.sha256(
            f"{task_name}:{candidates}:{tests}".encode("utf-8")
        ).hexdigest(),
    )


def selected_states(task: object) -> list[tuple[int, int, int]]:
    eligible = []
    for candidates, tests in corpus.collect_policy_states(task):
        population = candidates.bit_count()
        raw = tests.bit_count()
        _, _, classes = canonical_test_mask(task, candidates, tests)
        if (
            MIN_CANDIDATES <= population <= MAX_CANDIDATES
            and raw >= MIN_RAW_QUERIES
            and classes <= MAX_PARTITIONS
            and raw - classes >= MIN_REDUNDANCY
        ):
            eligible.append((candidates, tests, classes))
    eligible.sort(
        key=lambda row: deterministic_state_rank(
            task.name, row[0], row[1], row[2]
        )
    )
    return eligible[:MAX_STATES_PER_TASK]


def reproduce_state(
    task: object,
    candidates: int,
    tests: int,
) -> dict[str, object]:
    result = IndependentQuotientSolver(
        task, EXPANSION_BUDGET
    ).solve(candidates, tests)
    greedy = IndependentGreedySolver(task).solve(candidates, tests)
    saving = (
        greedy.total_cost - result.plan.total_cost
        if greedy.diagnosed == result.plan.diagnosed
        else None
    )
    state_digest = hashlib.sha256(
        f"{task.name}:{candidates}:{tests}".encode("utf-8")
    ).hexdigest()
    return {
        "task": task.name,
        "state_digest": state_digest,
        "candidate_count": candidates.bit_count(),
        "raw_remaining_queries": tests.bit_count(),
        "partition_classes": canonical_test_mask(
            task, candidates, tests
        )[2],
        "exact_plan": result.plan.metrics(),
        "greedy_plan": greedy.metrics(),
        "strict_exact_gain": (
            result.plan.ordering_key() > greedy.ordering_key()
        ),
        "total_query_saving": saving,
        "stats": asdict(result.stats),
        "budget_ladder": {
            str(budget): result.stats.expanded_tests <= budget
            for budget in BUDGET_LADDER
        },
    }


def run() -> dict[str, object]:
    tasks, verification = corpus.load_all_opened_tasks()
    rows = []
    for task in tasks:
        for candidates, tests, classes in selected_states(task):
            row = reproduce_state(task, candidates, tests)
            assert row["partition_classes"] == classes
            rows.append(row)
    rows.sort(key=lambda row: (row["task"], row["state_digest"]))
    savings = [
        int(row["total_query_saving"])
        for row in rows
        if row["total_query_saving"] is not None
    ]
    strict = sum(int(row["strict_exact_gain"]) for row in rows)
    ladder = {
        str(budget): sum(
            int(row["budget_ladder"][str(budget)]) for row in rows
        )
        for budget in BUDGET_LADDER
    }
    gate = (
        verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(rows) == 65
        and strict >= 58
        and sum(savings) >= 666
        and ladder["50000"] == 65
    )
    return {
        "status": "cleanroom_reproduced" if gate else "not_reproduced",
        "cleanroom_gate": gate,
        "solver_independence": {
            "imports_v44": False,
            "imports_prior_plan_type": False,
            "uses_prior_exact_planner": False,
            "shared_infrastructure": [
                "frozen dataset loaders",
                "task interface",
                "preregistered policy-state harvester",
            ],
        },
        "state_count": len(rows),
        "strict_exact_gain_count": strict,
        "aggregate_total_query_saving_vs_greedy": sum(savings),
        "budget_ladder_exact_solved": ladder,
        "archive_verification": verification,
        "rows": rows,
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
        "states": report["state_count"],
        "strict_gains": report["strict_exact_gain_count"],
        "query_saving": report[
            "aggregate_total_query_saving_vs_greedy"
        ],
        "budget_ladder": report["budget_ladder_exact_solved"],
    }, indent=2))


if __name__ == "__main__":
    main()
