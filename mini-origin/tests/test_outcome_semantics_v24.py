from __future__ import annotations

import inspect
import itertools

from mini_origin import outcome_semantics_v24 as v24


def test_semantic_universe_contains_exactly_the_reachable_cases() -> None:
    states = v24.canonical_semantic_universe()
    assert [state.name for state in states] == [
        "left_boundary_equal",
        "left_boundary_right",
        "interior_left",
        "interior_equal",
        "interior_right",
    ]
    assert all(state.right_exists for state in states)
    assert {state.action for state in states} == {
        v24.LEFT,
        v24.ACCEPT,
        v24.RIGHT,
    }


def test_machine_checked_lower_bound_is_three_features() -> None:
    certificate = v24.semantic_lower_bound_certificate()
    assert certificate["minimum_feature_count"] == 3
    assert certificate["unique_minimal_feature_set"]
    assert certificate["minimal_feature_sets"] == [
        ["left_exists", "left_active", "right_active"]
    ]
    assert certificate["all_smaller_subsets_refuted"]


def test_every_smaller_alphabet_has_an_explicit_collision_witness() -> None:
    certificate = v24.semantic_lower_bound_certificate()
    smaller = [
        row
        for row in certificate["rows"]
        if row["feature_count"] < certificate["minimum_feature_count"]
    ]
    assert len(smaller) == 15
    for row in smaller:
        witness = row["conflict_witness"]
        assert witness is not None
        assert witness["first_state"] != witness["second_state"]
        assert witness["first_action"] != witness["second_action"]


def test_selected_alphabet_is_action_consistent_on_all_semantic_states() -> None:
    features = ("left_exists", "left_active", "right_active")
    mapping, conflict = v24.semantic_lookup(features)
    assert conflict is None
    assert mapping is not None
    for state in v24.canonical_semantic_universe():
        code = v24.encode_values(state.values(), features)
        assert mapping[code] == state.action


def test_no_one_or_two_feature_subset_is_action_consistent() -> None:
    for count in (1, 2):
        for features in itertools.combinations(v24.FEATURE_NAMES, count):
            mapping, conflict = v24.semantic_lookup(features)
            assert mapping is None
            assert conflict is not None


def test_balanced_generator_represents_every_semantic_state_equally() -> None:
    examples = v24.generate_balanced_examples(
        seed=123,
        per_state=4,
        dimensions=(5, 7),
        replicates=64,
        rho_low=0.4,
        rho_high=0.8,
    )
    counts = {
        state.name: sum(example.state == state.name for example in examples)
        for state in v24.canonical_semantic_universe()
    }
    assert set(counts.values()) == {4}


def test_hidden_evidence_is_created_after_the_decoder_is_frozen() -> None:
    source = inspect.getsource(v24.run)
    freeze = source.index("frozen_digest = decoder_digest")
    hidden = source.index("hidden_examples = generate_balanced_examples")
    semantic_evaluation = source.index(
        "candidate_semantic = evaluate_classifier"
    )
    assert freeze < hidden < semantic_evaluation
    assert "dimensions=(9, 13, 21, 37, 63)" in source


def test_candidate_gate_requires_semantic_and_closed_loop_controls() -> None:
    source = inspect.getsource(v24.run)
    assert "certificate[\"minimum_feature_count\"] == 3" in source
    assert "certificate[\"all_smaller_subsets_refuted\"]" in source
    assert "candidate_semantic.minimum_state_accuracy >= 0.97" in source
    assert "semantic_gap >= 0.15" in source
    assert "candidate_loop.accuracy >= 0.985" in source
    assert "random_gap >= 0.45" in source
