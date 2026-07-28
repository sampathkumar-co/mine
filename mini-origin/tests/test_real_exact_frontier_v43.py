from mini_origin import real_exact_frontier_v43 as v43
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "frontier-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def test_uniform_expected_elimination_equals_gini_order() -> None:
    task = toy_task()
    certificate = v43.gini_approximation_certificate(
        task,
        task.full_mask,
        (1 << task.query_count) - 1,
    )
    assert certificate["checked"]
    assert certificate["passed"]
    assert certificate["selected_value"] == certificate["maximum_value"]


def test_structural_rank_is_data_result_independent() -> None:
    first = v43.structural_rank("task", 0b1111, 0b111111, 3)
    second = v43.structural_rank("task", 0b1111, 0b111111, 3)
    assert first == second
    assert len(first[-1]) == 64


def test_frontier_solver_preserves_exact_plan_when_both_finish() -> None:
    task = toy_task()
    row = v43.solve_frontier_state(
        task,
        task.full_mask,
        (1 << task.query_count) - 1,
    )
    assert row["quotient_solved"]
    assert row["plain_solved"]
    assert row["matched_if_both"]
    assert row["gini_approximation_certificate"]["passed"]


def test_budget_ladder_is_monotone() -> None:
    task = toy_task()
    row = v43.solve_frontier_state(
        task,
        task.full_mask,
        (1 << task.query_count) - 1,
    )
    quotient = [
        row["budget_ladder"][str(budget)]["quotient_solved"]
        for budget in v43.BUDGET_LADDER
    ]
    plain = [
        row["budget_ladder"][str(budget)]["plain_solved"]
        for budget in v43.BUDGET_LADDER
    ]
    assert quotient == sorted(quotient)
    assert plain == sorted(plain)
