from mini_origin import response_cost_lower_bound_v65 as bounded
from mini_origin import response_cost_pareto_v56 as response


def test_random_exact_plan_match() -> None:
    task, profile = response.random_task_and_profile(65001)
    remaining = (1 << task.query_count) - 1
    current = response.ParetoPlanner(
        task, profile, response.BUDGET
    ).result(task.full_mask, remaining)
    candidate = bounded.LowerBoundParetoPlanner(
        task, profile, response.BUDGET
    ).result(task.full_mask, remaining)
    assert bounded.exact_plan_tuple(candidate.plan) == bounded.exact_plan_tuple(
        current.plan
    )
    assert candidate.stats.query_expansions <= current.stats.query_expansions


def test_impossible_impure_child_is_prunable_after_full_incumbent() -> None:
    row = bounded.CandidateBound(
        query=7,
        children=(1, 2),
        full_diagnosis_possible=False,
        expected_cost_lower_bound=1,
        worst_cost_lower_bound=1,
    )
    incumbent = response.Plan(
        diagnosed_mass=10,
        expected_cost_numerator=100,
        worst_cost=50,
        query=3,
    )
    assert bounded.incumbent_dominates_bound(incumbent, 10, row) == (
        True,
        True,
    )


def test_cost_bound_requires_full_mass_incumbent() -> None:
    row = bounded.CandidateBound(
        query=4,
        children=(1, 2),
        full_diagnosis_possible=True,
        expected_cost_lower_bound=1000,
        worst_cost_lower_bound=1000,
    )
    incumbent = response.Plan(
        diagnosed_mass=9,
        expected_cost_numerator=1,
        worst_cost=1,
        query=2,
    )
    assert bounded.incumbent_dominates_bound(incumbent, 10, row) == (
        False,
        False,
    )
