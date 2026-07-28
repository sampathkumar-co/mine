from mini_origin import response_cost_pareto_v56 as v56
from mini_origin import state_policy_v34 as state


def dominated_toy():
    task = state.base.make_task(
        "response-cost-dominated-toy",
        ("expensive", "cheap", "other"),
        (
            ("0", "x", "0"),
            ("0", "x", "1"),
            ("1", "y", "0"),
            ("1", "y", "1"),
        ),
        ("a", "b", "b", "a"),
    )
    profile = v56.ResponseCostProfile(
        hypothesis_mass=(1, 3, 5, 7),
        hypothesis_cost_by_query=(
            (8, 8, 9, 9),
            (2, 2, 4, 4),
            (5, 5, 5, 5),
        ),
        seed=1,
    )
    return task, profile


def test_componentwise_dominated_equivalent_test_is_removed() -> None:
    task, profile = dominated_toy()
    remaining = (1 << task.query_count) - 1
    signature = v56.partition(task, task.full_mask, 0)
    assert signature == v56.partition(task, task.full_mask, 1)
    assert v56.cost_vector(
        task, profile, task.full_mask, 1
    ) == (2, 4)
    assert v56.cost_vector(
        task, profile, task.full_mask, 0
    ) == (8, 9)
    representatives = v56.pareto_representatives(
        task, profile, task.full_mask, remaining
    )
    assert representatives[signature] == (1,)
    certificate = v56.pareto_certificate(
        task, profile, task.full_mask, remaining
    )
    assert certificate["passed"]
    assert certificate["dominated_queries_removed"] == 1


def test_plain_and_pareto_exact_optima_match() -> None:
    task, profile = dominated_toy()
    remaining = (1 << task.query_count) - 1
    plain = v56.PlainPlanner(task, profile, 10000).result(
        task.full_mask, remaining
    )
    pareto = v56.ParetoPlanner(task, profile, 10000).result(
        task.full_mask, remaining
    )
    assert v56.plan_metrics(plain.plan) == v56.plan_metrics(
        pareto.plan
    )
    assert pareto.stats.query_expansions < plain.stats.query_expansions


def test_response_cost_dominance_is_hereditary() -> None:
    task, profile = dominated_toy()
    theorem = v56.hereditary_pareto_theorem(task, profile)
    assert theorem["passed"]
    assert theorem["comparisons"] > 0
    assert theorem["descendant_checks"] > 0


def test_incomparable_equivalent_tests_must_both_survive() -> None:
    certificate = v56.incomparable_counterexample()
    assert certificate["passed"]
    assert certificate["cost_vectors"] == [(1, 9), (9, 1)]
    assert certificate["first_optimal_query"] == 0
    assert certificate["second_optimal_query"] == 1


def test_random_response_cost_example_preserves_exact_optimum() -> None:
    task, profile = v56.random_task_and_profile(56002)
    remaining = (1 << task.query_count) - 1
    theorem = v56.hereditary_pareto_theorem(task, profile)
    plain = v56.PlainPlanner(task, profile, v56.BUDGET).result(
        task.full_mask, remaining
    )
    pareto = v56.ParetoPlanner(task, profile, v56.BUDGET).result(
        task.full_mask, remaining
    )
    assert theorem["passed"]
    assert v56.plan_metrics(plain.plan) == v56.plan_metrics(
        pareto.plan
    )


def test_profile_cost_is_constant_within_each_response() -> None:
    task, profile = v56.random_task_and_profile(56003)
    for query in range(task.query_count):
        observed = {}
        for hypothesis in range(task.candidate_count):
            response = task.rows[hypothesis][query]
            cost = profile.hypothesis_cost_by_query[query][hypothesis]
            if response in observed:
                assert observed[response] == cost
            else:
                observed[response] = cost
