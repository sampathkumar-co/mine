from mini_origin import average_odt_frontier_v44 as v44
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "average-frontier-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def test_expected_elimination_selects_maximum_value() -> None:
    task = toy_task()
    remaining = (1 << task.query_count) - 1
    selected = v44.select_expected_elimination_query(
        task, task.full_mask, remaining
    )
    values = [
        v44.expected_elimination_value(
            task, task.full_mask, query
        )
        for query in range(task.query_count)
        if len(v44.partition_signature(
            task, task.full_mask, query
        )) > 1
    ]
    assert v44.expected_elimination_value(
        task, task.full_mask, selected
    ) == max(values)


def test_average_plain_and_quotient_optima_match() -> None:
    task = toy_task()
    remaining = (1 << task.query_count) - 1
    plain = v44.AveragePlainPlanner(task, 10000).result(
        task.full_mask, remaining
    )
    quotient = v44.AverageQuotientPlanner(task, 10000).result(
        task.full_mask, remaining
    )
    assert v44.plan_metrics(plain.plan) == v44.plan_metrics(
        quotient.plan
    )
    assert quotient.stats.query_expansions < plain.stats.query_expansions


def test_exact_average_plan_dominates_greedy() -> None:
    task = toy_task()
    remaining = (1 << task.query_count) - 1
    exact = v44.AverageQuotientPlanner(task, 10000).result(
        task.full_mask, remaining
    ).plan
    greedy = v44.ExpectedEliminationGreedy(task).solve(
        task.full_mask, remaining
    )
    assert v44.average_plan_score(exact) >= v44.average_plan_score(greedy)


def test_frontier_state_reports_matched_exact_result() -> None:
    task = toy_task()
    row = v44.solve_state(
        task,
        task.full_mask,
        (1 << task.query_count) - 1,
    )
    assert row["quotient_solved"]
    assert row["plain_solved"]
    assert row["matched_if_both"]
    assert row["exact_dominates_greedy"]
    assert row["expected_elimination_root_certificate"]["passed"]
