import inspect

import numpy as np

from mini_origin.intervention_genesis_v22 import (
    decode_relation,
    information_lower_bound,
    observational_equivalence_certificate,
    optimal_query_offset,
    optimal_worst_case_queries,
    policy_digest,
    position_rules,
    root_chain_structural_covariance,
    ar1_covariance,
    synthesize_position_rule,
)


def test_all_root_orientations_have_the_same_observational_covariance() -> None:
    for dimension in (3, 5, 8):
        for rho in (0.25, 0.55, 0.88):
            expected = ar1_covariance(dimension, rho)
            for root in range(dimension):
                observed = root_chain_structural_covariance(
                    dimension, root, rho
                )
                assert np.allclose(observed, expected, atol=1e-10)


def test_equivalence_certificate_is_exact() -> None:
    certificate = observational_equivalence_certificate()
    assert certificate["exact_within_tolerance"]
    assert certificate["maximum_covariance_difference"] <= 1e-10
    assert certificate["checked_models"] > 0


def test_exhaustive_tree_depth_matches_information_lower_bound() -> None:
    for candidate_count in range(1, 129):
        assert optimal_worst_case_queries(candidate_count) == information_lower_bound(
            candidate_count
        )


def test_tie_broken_optimal_offsets_induce_lower_midpoint() -> None:
    selected, evidence = synthesize_position_rule()
    assert selected.name == "lower_midpoint"
    assert evidence["all_training_depths_meet_lower_bound"]
    for size in range(2, 8):
        assert optimal_query_offset(size) == (size - 1) // 2
        assert selected.offset(size) == optimal_query_offset(size)


def test_intervention_outcomes_encode_ordered_root_comparison() -> None:
    threshold = 0.2
    # Interior query: right child only means the root lies to the left.
    assert decode_relation(0.0, 0.6, 2, 5, threshold) == "left"
    # Interior query: left child only means the root lies to the right.
    assert decode_relation(0.6, 0.0, 2, 5, threshold) == "right"
    assert decode_relation(0.6, 0.6, 2, 5, threshold) == "equal"
    # Boundary queries still distinguish equality from the remaining interval.
    assert decode_relation(None, 0.6, 0, 5, threshold) == "equal"
    assert decode_relation(None, 0.0, 0, 5, threshold) == "right"
    assert decode_relation(0.6, None, 4, 5, threshold) == "equal"
    assert decode_relation(0.0, None, 4, 5, threshold) == "left"


def test_policy_digest_changes_with_rule_or_threshold() -> None:
    rules = {rule.name: rule for rule in position_rules()}
    first = policy_digest(rules["lower_midpoint"], 0.2)
    assert first == policy_digest(rules["lower_midpoint"], 0.2)
    assert first != policy_digest(rules["upper_midpoint"], 0.2)
    assert first != policy_digest(rules["lower_midpoint"], 0.21)


def test_hidden_worlds_are_generated_after_policy_freeze() -> None:
    from mini_origin import intervention_genesis_v22 as module

    source = inspect.getsource(module.run)
    freeze = source.index("frozen_digest = policy_digest")
    hidden = source.index("candidate = evaluate_policy")
    assert freeze < hidden
