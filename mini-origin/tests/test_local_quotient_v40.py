from mini_origin import exact_tail_v36 as v36
from mini_origin import local_quotient_v40 as v40
from mini_origin import safe_portfolio_v37 as v37
from mini_origin import state_policy_v34 as v34


def duplicate_task():
    return v34.base.make_task(
        "duplicate-query-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def test_local_quotient_removes_duplicate_partitions() -> None:
    task = duplicate_task()
    planner = v40.LocalQuotientPlanner(task)
    remaining = (1 << task.query_count) - 1
    canonical = planner.canonical_remaining(
        task.full_mask,
        remaining,
    )
    assert canonical.bit_count() == 2
    assert canonical & 1
    assert not (canonical & (1 << 1))


def test_quotient_exact_matches_unquotiented_optimum() -> None:
    task = duplicate_task()
    remaining = (1 << task.query_count) - 1
    quotient = v40.LocalQuotientPlanner(task).solve(
        task.full_mask,
        remaining,
    )
    original = v36.ExactPlanner(task).solve(
        task.full_mask,
        remaining,
    )
    assert quotient.diagnosed == original.diagnosed
    assert quotient.worst_queries == original.worst_queries
    assert quotient.total_queries == original.total_queries


def test_accepted_exact_subtree_is_executed_completely() -> None:
    task = duplicate_task()
    exact = v40.LocalQuotientPlanner(task)
    fallback = v37.FallbackPlanner(task, "first")
    result = v40.evaluate(
        task,
        v40.QuotientPolicy(8, "first"),
        exact,
        fallback,
    )
    plan = exact.solve(
        task.full_mask,
        (1 << task.query_count) - 1,
    )
    assert result.diagnosed_fraction == plan.diagnosed / task.candidate_count
    assert result.worst_queries == plan.worst_queries
    assert round(result.mean_queries * task.candidate_count) == plan.total_queries
    assert result.exact_query_uses > 0


def test_partition_quotient_is_monotone_on_subsets() -> None:
    task = duplicate_task()
    planner = v40.LocalQuotientPlanner(task)
    full = planner.partition_signature(task.full_mask, 0)
    copied = planner.partition_signature(task.full_mask, 1)
    assert full == copied
    subset = task.full_mask & ~(1 << 3)
    assert (
        planner.partition_signature(subset, 0)
        == planner.partition_signature(subset, 1)
    )
