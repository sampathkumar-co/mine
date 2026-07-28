from __future__ import annotations

import inspect

import numpy as np

from mini_origin import estimator_switch_v30 as v30


def test_diagnostics_separate_skew_from_symmetric_outliers() -> None:
    rng = np.random.default_rng(123)
    symmetric = rng.normal(size=2_000)
    symmetric[:20] = rng.normal(scale=30.0, size=20)
    skewed = rng.exponential(size=2_000) - 1.0
    symmetric_skew, symmetric_tail = v30.sample_diagnostics(symmetric)
    skewed_skew, _ = v30.sample_diagnostics(skewed)
    assert symmetric_tail > 4.0
    assert skewed_skew > symmetric_skew


def test_switch_uses_mean_for_skew_and_robust_for_symmetric_tails() -> None:
    rule = next(rule for rule in v30.position_rules() if rule.name == "lower_midpoint")
    policy = v30.SwitchPolicy(rule, "trimmed_mean_10", 0.25, 4.5, 0.14, 0.14, 0.08)
    rng = np.random.default_rng(456)
    symmetric = rng.normal(size=2_000)
    symmetric[:30] = rng.normal(scale=25.0, size=30)
    skewed = rng.exponential(size=2_000) - 1.0
    _, symmetric_robust = v30.switched_estimate(symmetric, policy)
    _, skewed_robust = v30.switched_estimate(skewed, policy)
    assert symmetric_robust
    assert not skewed_robust


def test_candidate_search_contains_both_robust_estimators() -> None:
    policies = v30.candidate_policies(123)
    assert {policy.robust_aggregator for policy in policies} == set(v30.ROBUST_AGGREGATORS)
    assert {policy.skew_threshold for policy in policies} == {0.15, 0.25, 0.35}
    assert {policy.tail_threshold for policy in policies} == {4.5, 6.0, 8.0}


def test_known_switch_policy_is_exact_without_noise() -> None:
    rule = next(rule for rule in v30.position_rules() if rule.name == "lower_midpoint")
    policy = v30.SwitchPolicy(rule, "median_of_means_8", 0.25, 6.0, 0.14, 0.14, 0.08)
    assert v30.exact_training_success(policy)


def test_hidden_noise_laws_are_new_and_sealed() -> None:
    development = {stratum.family for stratum in v30.development_strata()}
    hidden = {stratum.family for stratum in v30.hidden_strata()}
    assert development.isdisjoint(hidden)
    assert hidden == {
        "student2_5",
        "contaminated1x40",
        "centered_lognormal",
        "asymmetric_mixture",
    }
    source = inspect.getsource(v30.run)
    freeze = source.index("frozen_digest = digest(policy)")
    hidden_creation = source.index("strata = hidden_strata()")
    evaluation = source.index("candidate = evaluate_strata")
    assert freeze < hidden_creation < evaluation


def test_search_uses_terminal_success_and_switching_cost() -> None:
    source = inspect.getsource(v30.search_policy)
    assert ".action" not in source
    assert "worst_seen_law_then_mean_then_samples" in source
    assert "mean_replicates_per_query" in source


def test_gate_requires_benefit_over_both_fixed_estimators() -> None:
    source = inspect.getsource(v30.run)
    assert "portfolio_gap >= 0.02" in source
    assert "fixed_gap >= -0.01" in source
    assert "switch_range >= 0.20" in source
    assert "sample_saving >= 0.25" in source
    assert "candidate.minimum_accuracy >= 0.97" in source
