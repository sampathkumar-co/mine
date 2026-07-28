from mini_origin import exact_quotient_certificate_v42 as v42
from mini_origin import state_policy_v34 as v34


def duplicate_task():
    return v34.base.make_task(
        "duplicate-proof",
        ("a", "a-copy", "b", "b-copy"),
        (
            ("0", "0", "0", "0"),
            ("0", "0", "1", "1"),
            ("1", "1", "0", "0"),
            ("1", "1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def test_partition_equivalence_is_hereditary() -> None:
    task = duplicate_task()
    certificate = v42.local_equivalence_theorem(task)
    assert certificate["passed"]
    assert certificate["comparisons"] > 0
    assert certificate["descendants_checked"] > certificate["comparisons"]


def test_plain_and_quotient_global_optima_match() -> None:
    task = duplicate_task()
    remaining = (1 << task.query_count) - 1
    plain = v42.CountingPlainPlanner(task, 10000).result(
        task.full_mask, remaining
    )
    quotient = v42.CountingQuotientPlanner(task, 10000).result(
        task.full_mask, remaining
    )
    assert v42.plan_metrics(plain.plan) == v42.plan_metrics(quotient.plan)
    assert quotient.stats.query_expansions < plain.stats.query_expansions
    assert quotient.stats.cache_states < plain.stats.cache_states


def test_budget_is_operation_count_not_wall_clock() -> None:
    task = v42.duplicate_query_task(5201)
    remaining = (1 << task.query_count) - 1
    quotient = v42.CountingQuotientPlanner(task, 50000).result(
        task.full_mask, remaining
    )
    assert quotient.plan.diagnosed >= 0
    assert quotient.stats.query_expansions <= 50000
    try:
        v42.CountingPlainPlanner(task, 10).result(
            task.full_mask, remaining
        )
    except v42.BudgetExceeded:
        pass
    else:
        raise AssertionError("plain solver unexpectedly stayed within tiny budget")


def test_random_theorem_sample_has_zero_mismatches() -> None:
    for seed in range(4201, 4206):
        task = v42.random_task(seed)
        theorem = v42.local_equivalence_theorem(task)
        assert theorem["passed"]
        remaining = (1 << task.query_count) - 1
        plain = v42.CountingPlainPlanner(task, 250000).result(
            task.full_mask, remaining
        )
        quotient = v42.CountingQuotientPlanner(task, 250000).result(
            task.full_mask, remaining
        )
        assert v42.plan_metrics(plain.plan) == v42.plan_metrics(quotient.plan)
