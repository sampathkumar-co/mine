from __future__ import annotations

import inspect

from mini_origin import adaptive_measurement_v28 as v28


def test_adaptive_policy_is_exact_on_locked_training_worlds() -> None:
    rule = next(rule for rule in v28.position_rules() if rule.name == "lower_midpoint")
    policy = v28.AdaptivePolicy(rule, 0.15, 0.15, 0.05, 256, 512)
    assert v28.exact_training_success(policy)


def test_resampling_is_triggered_only_near_a_decision_boundary() -> None:
    rule = next(rule for rule in v28.position_rules() if rule.name == "lower_midpoint")
    policy = v28.AdaptivePolicy(rule, 0.15, 0.15, 0.05, 256, 512)
    far = v28.adaptive_response(1, 5, 2, 0.90, 2, policy)
    near = v28.adaptive_response(2, 5, 2, 0.24, 2, policy)
    assert far[2] in (256, 768)
    assert near[2] in (256, 768)
    assert far[2] == 256 or far[3]
    assert near[2] == 768 or not near[3]


def test_combined_measurement_uses_all_samples() -> None:
    assert v28._combine(0.2, 0.4, 256, 512) == (256 * 0.2 + 512 * 0.4) / 768
    assert v28._combine(None, None, 256, 512) is None


def test_candidate_grammar_contains_real_adaptive_policies() -> None:
    observations = [(0.0, 0.0), (0.25, 0.25), (0.0, 0.25), (0.25, 0.0)] * 20
    policies = v28.candidate_policies(observations)
    assert policies
    assert all(policy.base_replicates == 256 for policy in policies)
    assert {policy.extra_replicates for policy in policies} == {256, 512}
    assert all(policy.trigger_margin > 0.0 for policy in policies)


def test_search_uses_terminal_reward_and_sample_aware_selection() -> None:
    source = inspect.getsource(v28.search_policy)
    assert ".action" not in source
    assert "worst_support_accuracy_then_mean_then_samples" in source
    assert "mean_replicates_per_query" in source


def test_hidden_support_is_created_only_after_policy_freeze() -> None:
    source = inspect.getsource(v28.run)
    freeze = source.index("frozen_digest = digest(policy)")
    hidden = source.index("strata = hidden_strata()")
    evaluation = source.index("candidate = evaluate_support")
    assert freeze < hidden < evaluation
    assert v28.hidden_strata()[0].dimensions == (17, 31, 63, 127, 255)


def test_gate_counts_samples_and_uses_fixed_budget_controls() -> None:
    source = inspect.getsource(v28.run)
    assert "candidate.maximum_mean_replicates_per_query <= 600.0" in source
    assert "fixed_256_gap >= 0.03" in source
    assert "fixed_512_gap >= 0.005" in source
    assert "specialist_gap >= -0.01" in source
    assert "sample_saving >= 0.20" in source
    assert "random_gap >= 0.45" in source
