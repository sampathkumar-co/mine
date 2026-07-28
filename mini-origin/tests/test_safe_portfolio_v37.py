from mini_origin import exact_tail_v36 as v36
from mini_origin import safe_portfolio_v37 as v37
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "toy",
        ("a", "b", "c"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("0", "1", "0"),
            ("1", "0", "0"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "x", "y", "y", "z", "z"),
    )


def test_fallback_plan_matches_constant_policy() -> None:
    task = toy_task()
    for objective in v34.OBJECTIVE_NAMES:
        planner = v37.FallbackPlanner(task, objective)
        plan = planner.solve(
            task.full_mask,
            (1 << task.query_count) - 1,
        )
        score = v34.evaluate(
            task,
            v34.StatePolicy(None, objective, objective),
        )
        assert plan.diagnosed == round(
            score.diagnosed_fraction * task.candidate_count
        )
        assert plan.worst_queries == score.worst_queries
        assert plan.total_queries == round(
            score.mean_queries * task.candidate_count
        )


def test_safe_repair_is_lexicographically_no_worse() -> None:
    tasks = [
        toy_task(),
        v34.base.load_monk(1, "train"),
        v34.base.load_zoo(),
    ]
    for task in tasks:
        exact = v36.ExactPlanner(task)
        for objective in v34.OBJECTIVE_NAMES:
            fallback = v37.FallbackPlanner(task, objective)
            policy = v37.SafePolicy(16, 10, objective)
            repaired = v37.evaluate(
                task,
                policy,
                exact,
                fallback,
            )
            original = v34.evaluate(
                task,
                v34.StatePolicy(None, objective, objective),
            )
            repaired_score = (
                repaired.diagnosed_fraction,
                -repaired.worst_queries,
                -repaired.mean_queries,
            )
            original_score = (
                original.diagnosed_fraction,
                -original.worst_queries,
                -original.mean_queries,
            )
            assert repaired_score >= original_score


def test_dominance_requires_strict_improvement() -> None:
    a = v36.Plan(4, 2, 8, 0)
    b = v36.Plan(4, 2, 9, 1)
    assert v37.strictly_dominates(a, b)
    assert not v37.strictly_dominates(a, a)
