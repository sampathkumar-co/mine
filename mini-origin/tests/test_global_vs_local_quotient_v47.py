from mini_origin import global_vs_local_quotient_v47 as v47
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "global-local-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def test_one_time_global_preprocessing_removes_root_duplicates() -> None:
    task = toy_task()
    remaining = (1 << task.query_count) - 1
    mask = v47.one_time_global_mask(task, task.full_mask, remaining)
    assert mask.bit_count() == 2
    assert mask & 1
    assert not (mask & (1 << 1))


def test_global_and_local_exact_plans_match_on_toy() -> None:
    task = toy_task()
    remaining = (1 << task.query_count) - 1
    global_result = v47.GlobalOnlyQuotientPlanner(
        task, task.full_mask, remaining, 10_000
    ).result(task.full_mask, remaining)
    from mini_origin import average_odt_frontier_v44 as v44
    local_result = v44.AverageQuotientPlanner(task, 10_000).result(
        task.full_mask, remaining
    )
    assert v44.plan_metrics(global_result.plan) == v44.plan_metrics(
        local_result.plan
    )


def test_root_equivalence_is_hereditary() -> None:
    task = toy_task()
    full_left = v47.partition(task, task.full_mask, 0)
    full_right = v47.partition(task, task.full_mask, 1)
    assert full_left == full_right
    subset = task.full_mask & ~(1 << 3)
    assert v47.partition(task, subset, 0) == v47.partition(
        task, subset, 1
    )
