from __future__ import annotations

import inspect

from mini_origin import semantic_alphabet_v24 as v24
from mini_origin.outcome_alphabet_v23 import ACCEPT, LEFT, RIGHT


def test_semantic_state_set_covers_boundaries_and_interiors() -> None:
    states = {state.name: state.action for state in v24.semantic_states()}
    assert states == {
        "left_boundary_equal": ACCEPT,
        "left_boundary_root_right": RIGHT,
        "right_boundary_equal": ACCEPT,
        "right_boundary_root_left": LEFT,
        "interior_root_left": LEFT,
        "interior_equal": ACCEPT,
        "interior_root_right": RIGHT,
    }


def test_exhaustive_certificate_proves_four_bit_minimum() -> None:
    selected, smaller, certificate = v24.exhaustive_semantic_certificate(0.2)
    assert certificate.exhaustive_subsets == 31
    assert certificate.state_count == 7
    assert certificate.selected_accuracy == 1.0
    assert certificate.selected_feature_count == 4
    assert certificate.perfect_smaller_alphabets == 0
    assert certificate.best_smaller_accuracy == 6.0 / 7.0
    assert selected.features == (
        "left_exists",
        "right_exists",
        "left_active",
        "right_active",
    )
    assert all(len(program.features) < 4 for program in smaller)


def test_selected_alphabet_classifies_every_ideal_state() -> None:
    selected, _, _ = v24.exhaustive_semantic_certificate(0.2)
    for state in v24.semantic_states():
        assert v24.decode_action(selected, state.left, state.right) == state.action


def test_lower_and_upper_tie_breaks_reach_opposite_boundaries() -> None:
    assert v24.query_position("lower", 0, 1, 0) == 0
    assert v24.query_position("upper", 0, 1, 0) == 1
    assert v24.query_position("alternating", 0, 1, 0) == 0
    assert v24.query_position("alternating", 0, 1, 1) == 1


def test_boundary_heavy_root_sampler_includes_both_edges() -> None:
    import numpy as np

    rng = np.random.default_rng(24)
    roots = [v24.sample_root(rng, 17) for _ in range(1000)]
    assert roots.count(0) > 150
    assert roots.count(16) > 150
    assert any(root not in (0, 1, 15, 16) for root in roots)


def test_hidden_family_opens_after_program_freeze() -> None:
    source = inspect.getsource(v24.run)
    freeze = source.index("frozen_digest = program_digest")
    hidden = source.index("hidden_dimensions = (10, 15, 26, 41, 70)")
    candidate = source.index("candidate = evaluate_program")
    assert freeze < hidden < candidate
    assert "candidate_smaller_gap" in source


def test_gate_requires_all_policy_families_and_semantic_proof() -> None:
    source = inspect.getsource(v24.run)
    assert "certificate.selected_feature_count == 4" in source
    assert "certificate.perfect_smaller_alphabets == 0" in source
    assert "policy_floor >= 0.98" in source
    assert "reduced_gap >= 0.06" in source
