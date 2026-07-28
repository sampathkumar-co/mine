from mini_origin import exact_tail_runner_v36 as runner
from mini_origin import exact_tail_v36 as v36
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "xor-toy",
        ("a", "b"),
        (
            ("0", "0"),
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
        ),
        ("0", "1", "1", "0"),
    )


def test_exact_planner_solves_xor_optimally() -> None:
    task = toy_task()
    planner = v36.ExactPlanner(task)
    plan = planner.solve(
        task.full_mask,
        (1 << task.query_count) - 1,
    )
    assert plan.diagnosed == 4
    assert plan.worst_queries == 2
    assert plan.total_queries == 8
    assert plan.query in (0, 1)


def test_exact_policy_executes_the_certified_plan() -> None:
    task = toy_task()
    planner = v36.ExactPlanner(task)
    policy = v36.TailPolicy(4, 2, "gini")
    score = v36.evaluate(task, policy, planner)
    assert score.diagnosed_fraction == 1.0
    assert score.mean_queries == 2.0
    assert score.worst_queries == 2
    assert score.exact_query_uses == 8


def test_zero_threshold_matches_constant_fallback() -> None:
    task = toy_task()
    tail = v36.evaluate(
        task,
        v36.TailPolicy(0, 0, "gini"),
        v36.ExactPlanner(task),
    )
    constant = v34.evaluate(
        task,
        v34.StatePolicy(None, "gini", "gini"),
    )
    assert tail.diagnosed_fraction == constant.diagnosed_fraction
    assert tail.mean_queries == constant.mean_queries
    assert tail.worst_queries == constant.worst_queries


def test_bounded_protocol_has_150_programs() -> None:
    v36.THRESHOLDS = runner.THRESHOLDS
    v36.FEATURE_LIMITS = runner.FEATURE_LIMITS
    assert len(v36.grammar()) == 150
