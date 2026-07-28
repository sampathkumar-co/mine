from __future__ import annotations

import inspect

from mini_origin import outcome_alphabet_v23 as v23
from mini_origin.intervention_genesis_v22 import synthesize_position_rule


def test_response_encoding_uses_existence_and_activity_bits() -> None:
    features = (
        "left_exists",
        "right_exists",
        "left_active",
        "right_active",
    )
    threshold = 0.2
    assert v23.encode_response(None, 0.0, threshold, features) == 2
    assert v23.encode_response(None, 0.7, threshold, features) == 10
    assert v23.encode_response(0.0, None, threshold, features) == 1
    assert v23.encode_response(0.7, None, threshold, features) == 5
    assert v23.encode_response(0.0, 0.7, threshold, features) == 11
    assert v23.encode_response(0.7, 0.0, threshold, features) == 7
    assert v23.encode_response(0.7, 0.7, threshold, features) == 15


def test_optimal_lookup_uses_majority_action_for_each_code() -> None:
    examples = [
        v23.DecoderExample(0.0, 0.7, v23.LEFT),
        v23.DecoderExample(0.0, 0.7, v23.LEFT),
        v23.DecoderExample(0.0, 0.7, v23.ACCEPT),
        v23.DecoderExample(0.7, 0.0, v23.RIGHT),
    ]
    accuracy, mapping, default = v23.optimal_lookup(
        examples,
        ("left_active", "right_active"),
        threshold=0.2,
    )
    left_code = v23.encode_response(
        0.0, 0.7, 0.2, ("left_active", "right_active")
    )
    right_code = v23.encode_response(
        0.7, 0.0, 0.2, ("left_active", "right_active")
    )
    assert mapping[left_code] == v23.LEFT
    assert mapping[right_code] == v23.RIGHT
    assert accuracy == 0.75
    assert default == v23.LEFT


def test_invalid_interval_transition_is_a_failure() -> None:
    query_rule, _ = synthesize_position_rule()
    # Always moving left from the first element of a two-element interval makes
    # the interval invalid. The runner must not reinterpret this contradiction
    # as a lucky correct answer.
    program = v23.OutcomeProgram(
        features=("left_exists",),
        threshold=0.2,
        mapping=((0, v23.LEFT), (1, v23.LEFT)),
        default_action=v23.LEFT,
        training_accuracy=0.0,
        development_accuracy=0.0,
        mapping_entries=2,
    )
    correct, _, invalid, remaining = v23.run_closed_loop(
        seed=123,
        dimension=2,
        root=1,
        rho=0.7,
        query_rule=query_rule,
        program=program,
        replicates=512,
        query_budget=1,
    )
    assert not correct
    assert invalid
    assert remaining == 0


def test_human_decoder_matches_the_intended_boundary_and_interior_actions() -> None:
    program = v23.hand_program(0.2)
    assert v23.decode_action(program, None, 0.0) == v23.RIGHT
    assert v23.decode_action(program, None, 0.7) == v23.ACCEPT
    assert v23.decode_action(program, 0.0, None) == v23.LEFT
    assert v23.decode_action(program, 0.7, None) == v23.ACCEPT
    assert v23.decode_action(program, 0.0, 0.7) == v23.LEFT
    assert v23.decode_action(program, 0.7, 0.0) == v23.RIGHT
    assert v23.decode_action(program, 0.7, 0.7) == v23.ACCEPT


def test_hidden_dimensions_are_irregular_and_created_after_freeze() -> None:
    source = inspect.getsource(v23.run)
    freeze = source.index("frozen_digest = decoder_digest")
    hidden = source.index("hidden_dimensions = (9, 13, 21, 37, 63)")
    evaluation = source.index("candidate = evaluate_program")
    assert freeze < hidden < evaluation
    assert "(8, 16, 32, 64)" not in source


def test_candidate_gate_requires_reduced_decoder_gap() -> None:
    source = inspect.getsource(v23.run)
    assert "reduced_gap >= 0.02" in source
    assert "random_gap >= 0.45" in source
    assert "candidate.invalid_transition_rate <= 0.01" in source
