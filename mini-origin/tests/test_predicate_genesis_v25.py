from __future__ import annotations

import inspect

from mini_origin import predicate_genesis_v25 as v25
from mini_origin.outcome_semantics_v24 import generate_balanced_examples


def test_relational_predicate_uses_raw_response_difference() -> None:
    predicate = v25.Predicate("difference", "right-left", 0.15, 3)
    assert predicate.evaluate(0.0, 0.7)
    assert not predicate.evaluate(0.7, 0.7)
    assert not predicate.evaluate(0.7, 0.0)
    assert not predicate.evaluate(None, 0.7)


def test_two_predicates_separate_every_reachable_action_state() -> None:
    predicates = (
        v25.Predicate("greater", "right", 0.15, 2),
        v25.Predicate("difference", "right-left", 0.15, 3),
    )
    assert v25.conflict(predicates) is None
    assert v25.conflict((predicates[0],)) is not None
    assert v25.conflict((predicates[1],)) is not None


def test_small_search_invents_the_two_predicate_relational_alphabet() -> None:
    training = generate_balanced_examples(
        411,
        per_state=160,
        dimensions=(3, 4, 5, 6, 7),
        replicates=384,
        rho_low=0.30,
        rho_high=0.90,
    )
    development = generate_balanced_examples(
        887,
        per_state=100,
        dimensions=(4, 6, 9),
        replicates=384,
        rho_low=0.28,
        rho_high=0.92,
    )
    selected, reduced, _, grammar = v25.synthesize(training, development)
    proof = v25.certificate(grammar, selected)
    assert len(selected.predicates) == 2
    assert any(
        p.kind == "greater" and p.channel == "right"
        for p in selected.predicates
    )
    assert any(
        p.kind == "difference" and p.channel == "right-left"
        for p in selected.predicates
    )
    assert len(reduced.predicates) == 1
    assert proof["minimum_predicate_count"] == 2
    assert proof["all_single_predicates_refuted"]


def test_named_v024_feature_vocabulary_is_not_imported() -> None:
    source = inspect.getsource(v25)
    assert "FEATURE_NAMES" not in source
    assert "left_exists" not in source
    assert "right_active" not in source


def test_hidden_evidence_is_created_after_program_freeze() -> None:
    source = inspect.getsource(v25.run)
    freeze = source.index("frozen_digest = digest")
    hidden = source.index("hidden = generate_balanced_examples")
    evaluation = source.index("candidate_semantic")
    assert freeze < hidden < evaluation
    assert "hidden_dimensions = (9, 13, 21, 37, 63)" in source


def test_candidate_gate_requires_compression_and_strong_controls() -> None:
    source = inspect.getsource(v25.run)
    assert "len(selected.predicates) == 2" in source
    assert 'proof["minimum_predicate_count"] == 2' in source
    assert 'proof["all_single_predicates_refuted"]' in source
    assert "candidate_semantic.minimum_state_accuracy >= 0.97" in source
    assert "candidate_loop.accuracy >= 0.985" in source
    assert "random_gap >= 0.45" in source
