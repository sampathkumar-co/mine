from mini_origin import cost_prior_quotient_v48 as v48
from mini_origin import state_policy_v34 as state


def toy_task():
    return state.base.make_task(
        "weighted-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def toy_profile():
    return v48.WeightedProfile(
        hypothesis_mass=(1, 3, 5, 7),
        query_cost=(9, 2, 5),
        seed=1,
    )


def test_local_class_retains_cheapest_equivalent_query() -> None:
    task = toy_task()
    profile = toy_profile()
    remaining = (1 << task.query_count) - 1
    representatives = v48.representative_map(
        task, profile, task.full_mask, remaining
    )
    duplicate_signature = v48.partition(task, task.full_mask, 0)
    assert duplicate_signature == v48.partition(
        task, task.full_mask, 1
    )
    assert representatives[duplicate_signature] == 1
    certificate = v48.representative_certificate(
        task, profile, task.full_mask, remaining
    )
    assert certificate["passed"]
    assert certificate["cost_sensitive_representative_changes"] == 1


def test_weighted_plain_and_local_exact_optima_match() -> None:
    task = toy_task()
    profile = toy_profile()
    remaining = (1 << task.query_count) - 1
    plain = v48.PlainWeightedPlanner(task, profile, 10000).result(
        task.full_mask, remaining
    )
    local = v48.LocalWeightedPlanner(task, profile, 10000).result(
        task.full_mask, remaining
    )
    assert v48.plan_metrics(plain.plan) == v48.plan_metrics(local.plan)
    assert local.stats.query_expansions < plain.stats.query_expansions


def test_equivalent_query_relation_is_hereditary() -> None:
    theorem = v48.hereditary_cost_theorem(toy_task(), toy_profile())
    assert theorem["passed"]
    assert theorem["comparisons"] > 0
    assert theorem["descendant_checks"] > 0


def test_weighted_exact_plan_dominates_greedy() -> None:
    task = toy_task()
    profile = toy_profile()
    remaining = (1 << task.query_count) - 1
    exact = v48.LocalWeightedPlanner(task, profile, 10000).result(
        task.full_mask, remaining
    ).plan
    greedy = v48.WeightedGreedy(task, profile).solve(
        task.full_mask, remaining
    )
    assert v48.plan_score(exact) >= v48.plan_score(greedy)


def test_profile_generation_is_deterministic_and_positive() -> None:
    task = toy_task()
    first = v48.profile_for_task(task, 4801)
    second = v48.profile_for_task(task, 4801)
    assert first == second
    assert min(first.hypothesis_mass) > 0
    assert min(first.query_cost) > 0
    assert v48.profile_digest(task, first) == v48.profile_digest(
        task, second
    )


def test_random_weighted_exact_certificate_example() -> None:
    task, profile = v48.random_weighted_task(48001)
    theorem = v48.hereditary_cost_theorem(task, profile)
    remaining = (1 << task.query_count) - 1
    plain = v48.PlainWeightedPlanner(task, profile, 500000).result(
        task.full_mask, remaining
    )
    local = v48.LocalWeightedPlanner(task, profile, 500000).result(
        task.full_mask, remaining
    )
    assert theorem["passed"]
    assert v48.plan_metrics(plain.plan) == v48.plan_metrics(local.plan)
