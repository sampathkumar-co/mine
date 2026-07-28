from __future__ import annotations

import inspect

import numpy as np

from mini_origin import noise_transfer_v29 as v29


def test_all_noise_generators_are_centered_and_finite() -> None:
    rng = np.random.default_rng(123)
    for family in (
        "gaussian",
        "laplace",
        "student5",
        "student3",
        "contaminated2",
        "contaminated5",
        "skewed",
        "variance_mixture",
    ):
        values = v29._standard_noise(rng, family, 20_000)
        assert np.isfinite(values).all()
        assert abs(float(np.mean(values))) < 0.12


def test_robust_aggregators_are_not_aliases_of_the_mean() -> None:
    values = np.array([0.0] * 31 + [100.0])
    mean = v29.aggregate(values, "mean")
    trimmed = v29.aggregate(values, "trimmed_mean_10")
    median_of_means = v29.aggregate(values, "median_of_means_8")
    assert trimmed < mean
    assert median_of_means < mean


def test_candidate_search_includes_robust_estimators() -> None:
    policies = v29.candidate_policies(123)
    assert {policy.aggregator for policy in policies} == set(v29.AGGREGATORS)
    assert all(policy.base_replicates == 256 for policy in policies)
    assert all(policy.extra_replicates == 512 for policy in policies)


def test_known_robust_policy_is_exact_without_noise() -> None:
    rule = next(rule for rule in v29.position_rules() if rule.name == "lower_midpoint")
    policy = v29.RobustPolicy(rule, "median_of_means_8", 0.14, 0.14, 0.08)
    assert v29.exact_training_success(policy)


def test_development_and_hidden_noise_laws_are_disjoint() -> None:
    development = {stratum.family for stratum in v29.development_strata()}
    hidden = {stratum.family for stratum in v29.hidden_strata()}
    assert development.isdisjoint(hidden)
    assert hidden == {"student3", "contaminated5", "skewed", "variance_mixture"}


def test_search_uses_terminal_reward_without_step_labels() -> None:
    source = inspect.getsource(v29.search_policy)
    assert ".action" not in source
    assert "worst_seen_noise_then_mean_then_samples" in source
    assert "selected_aggregator" in source


def test_hidden_noise_is_created_only_after_policy_freeze() -> None:
    source = inspect.getsource(v29.run)
    freeze = source.index("frozen_digest = digest(policy)")
    hidden = source.index("strata = hidden_strata()")
    candidate = source.index("candidate = evaluate_strata")
    assert freeze < hidden < candidate


def test_gate_requires_unseen_noise_transfer_and_sample_efficiency() -> None:
    source = inspect.getsource(v29.run)
    assert 'policy.aggregator != "mean"' in source
    assert "candidate.minimum_accuracy >= 0.98" in source
    assert "mean_gap >= 0.02" in source
    assert "robust_fixed_gap >= -0.01" in source
    assert "mean_fixed_gap >= 0.0" in source
    assert "sample_saving >= 0.25" in source
