from __future__ import annotations

import inspect

from mini_origin import terminal_reward_v26 as v26
from mini_origin.outcome_alphabet_v23 import ACCEPT, LEFT, RIGHT


def test_noiseless_responses_match_chain_semantics() -> None:
    assert v26.noiseless_response(5, 0, 0.7, 0) == (None, 0.7)
    assert v26.noiseless_response(5, 4, 0.7, 0) == (None, 0.0)
    assert v26.noiseless_response(5, 1, 0.7, 2) == (0.0, 0.7)
    assert v26.noiseless_response(5, 2, 0.7, 2) == (0.7, 0.7)
    assert v26.noiseless_response(5, 4, 0.7, 2) == (0.7, 0.0)


def test_mapping_space_contains_every_four_code_controller() -> None:
    mappings = list(v26._mapping_space(2))
    assert len(mappings) == 81
    assert (RIGHT, ACCEPT, RIGHT, LEFT) in mappings


def test_known_two_predicate_controller_solves_exact_training_worlds() -> None:
    program = v26.specialist_program()
    assert all(
        v26.run_noiseless_trial(program, world)
        for world in v26.exact_worlds()
    )


def test_search_uses_terminal_reward_not_step_action_labels() -> None:
    source = inspect.getsource(v26.search_terminal_programs)
    assert ".action" not in source
    assert "terminal_root_success_only" in source
    assert "_all_exact_two_predicate_programs" in source
    assert "fit_table" not in source


def test_unsupervised_grammar_does_not_receive_root_actions() -> None:
    source = inspect.getsource(v26.terminal_predicate_grammar)
    assert "action" not in source
    assert "observations" in source


def test_hidden_worlds_are_created_after_full_program_freeze() -> None:
    source = inspect.getsource(v26.run)
    freeze = source.index("frozen_digest = digest(selected)")
    hidden = source.index("hidden_dimensions = (13, 21, 37, 63, 127)")
    evaluation = source.index("candidate = evaluate_noisy")
    assert freeze < hidden < evaluation


def test_gate_requires_joint_query_predicate_and_terminal_controls() -> None:
    source = inspect.getsource(v26.run)
    assert 'selected.rule.name == "lower_midpoint"' in source
    assert 'search["step_action_labels_used"] is False' in source
    assert "selected.training_accuracy == 1.0" in source
    assert "single_gap >= 0.20" in source
    assert "query_gap >= 0.20" in source
    assert "random_gap >= 0.45" in source
