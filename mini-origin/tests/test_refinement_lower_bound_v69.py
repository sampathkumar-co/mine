from mini_origin import refinement_lower_bound_v69 as refined
from mini_origin import response_cost_lower_bound_v65 as lower
from mini_origin import response_cost_pareto_v56 as response
from mini_origin import state_policy_v34 as state


def make_refinement_task():
    task = state.base.make_task(
        "refinement-test",
        ("coarse", "fine"),
        (
            ("a", "x"),
            ("a", "y"),
            ("b", "z"),
        ),
        ("L0", "L1", "L2"),
    )
    return task


def profile(coarse_costs, fine_costs):
    return response.ResponseCostProfile(
        hypothesis_mass=(1, 2, 3),
        hypothesis_cost_by_query=(coarse_costs, fine_costs),
        seed=69,
    )


def test_strict_refinement_removes_costlier_coarse_query():
    task = make_refinement_task()
    mask, equivalent, refinement = refined.refinement_mask(
        task,
        profile((2, 2, 2), (1, 1, 1)),
        task.full_mask,
        0b11,
    )
    assert mask == 0b10
    assert equivalent == 0
    assert refinement == 1


def test_equal_cost_refinement_is_retained_for_tie_breaking():
    task = make_refinement_task()
    mask, _, refinement = refined.refinement_mask(
        task,
        profile((1, 1, 1), (1, 1, 1)),
        task.full_mask,
        0b11,
    )
    assert mask == 0b11
    assert refinement == 0


def test_more_expensive_fine_query_does_not_dominate():
    task = make_refinement_task()
    mask, _, refinement = refined.refinement_mask(
        task,
        profile((1, 1, 1), (2, 1, 1)),
        task.full_mask,
        0b11,
    )
    assert mask == 0b11
    assert refinement == 0


def test_refined_planner_matches_lower_bound_on_random_task():
    task, cost_profile = response.random_task_and_profile(69_999)
    remaining = (1 << task.query_count) - 1
    current = lower.LowerBoundParetoPlanner(
        task, cost_profile, response.BUDGET
    ).result(task.full_mask, remaining)
    candidate = refined.RefinementLowerBoundPlanner(
        task, cost_profile, response.BUDGET
    ).result(task.full_mask, remaining)
    assert lower.exact_plan_tuple(current.plan) == lower.exact_plan_tuple(
        candidate.plan
    )
    assert candidate.stats.query_expansions <= current.stats.query_expansions
