from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from mini_origin import cleanroom_odt_v45 as v45
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "cleanroom-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def exhaustive_metrics(task: object, candidates: int, tests: int):
    @lru_cache(maxsize=None)
    def solve(allowed: int, remaining: int):
        population = allowed.bit_count()
        if v45.is_label_pure(task, allowed):
            return population, 0, 0
        alternatives = []
        pending = remaining
        while pending:
            bit = pending & -pending
            test = bit.bit_length() - 1
            pending ^= bit
            children = v45.response_partition(task, allowed, test)
            if len(children) <= 1:
                continue
            child_rows = [
                solve(child, remaining & ~(1 << test))
                for child in children
            ]
            alternatives.append((
                sum(row[0] for row in child_rows),
                population + sum(row[1] for row in child_rows),
                1 + max(row[2] for row in child_rows),
                test,
            ))
        if not alternatives:
            return 0, 0, 0
        chosen = max(
            alternatives,
            key=lambda row: (row[0], -row[1], -row[2], -row[3]),
        )
        return chosen[:3]

    return solve(candidates, tests)


def test_cleanroom_source_does_not_import_prior_planners() -> None:
    source_path = Path(v45.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
    forbidden_modules = {
        "mini_origin.average_odt_frontier_v44",
        "mini_origin.exact_tail_v36",
        "mini_origin.local_quotient_v40",
    }
    assert not (imported_modules & forbidden_modules)
    for forbidden_symbol in (
        "AverageQuotientPlanner",
        "AveragePlainPlanner",
        "ExpectedEliminationGreedy",
        "v36.Plan",
    ):
        assert forbidden_symbol not in source


def test_cleanroom_solver_matches_exhaustive_optimum() -> None:
    task = toy_task()
    tests = (1 << task.query_count) - 1
    clean = v45.IndependentQuotientSolver(task, 10_000).solve(
        task.full_mask, tests
    )
    assert clean.plan.metrics() == exhaustive_metrics(
        task, task.full_mask, tests
    )


def test_duplicate_tests_collapse_to_one_partition_class() -> None:
    task = toy_task()
    tests = (1 << task.query_count) - 1
    canonical, raw, classes = v45.canonical_test_mask(
        task, task.full_mask, tests
    )
    assert raw == 3
    assert classes == 2
    assert canonical & 1
    assert not (canonical & (1 << 1))


def test_cleanroom_greedy_selects_maximum_elimination() -> None:
    task = toy_task()
    tests = (1 << task.query_count) - 1
    selected = v45.expected_elimination_test(
        task, task.full_mask, tests
    )
    values = [
        v45.expected_eliminations(task, task.full_mask, test)
        for test in range(task.query_count)
        if len(v45.response_partition(task, task.full_mask, test)) > 1
    ]
    assert v45.expected_eliminations(
        task, task.full_mask, selected
    ) == max(values)
