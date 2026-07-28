from __future__ import annotations

import inspect
import numpy as np

from mini_origin import tree_compiler_v26 as v26


def test_random_tree_is_connected_and_has_n_minus_one_edges() -> None:
    tree = v26.random_tree(31, np.random.default_rng(7))
    edge_count = sum(len(neighbors) for neighbors in tree) // 2
    connected = v26.component(tree, set(range(len(tree))), 0, -1)
    assert edge_count == 30
    assert connected == set(range(31))


def test_sum_distance_expression_is_a_half_separator_on_adversarial_trees() -> None:
    program = v26.QueryProgram("sum_distance", 3)
    for tree in (v26.path_tree(31), v26.broom_tree(31), v26.comet_tree(31)):
        allowed = set(range(len(tree)))
        query = v26.select_query(tree, allowed, program)
        assert max(v26.component_sizes(tree, allowed, query), default=0) <= len(tree) // 2


def test_minimum_response_decoder_follows_parent_and_accepts_root() -> None:
    decoder = v26.DecoderProgram("minimum_below", 0.15, 2)
    assert v26.decode(decoder, [(1, 0.7), (2, 0.0), (3, 0.8)]) == 2
    assert v26.decode(decoder, [(1, 0.7), (2, 0.6), (3, 0.8)]) is None


def test_exact_policy_finds_every_root_in_multiple_tree_families() -> None:
    policy = v26.Policy(
        v26.QueryProgram("sum_distance", 3),
        v26.DecoderProgram("minimum_below", 0.15, 2),
    )
    trees = (
        v26.path_tree(17),
        v26.balanced_tree(17),
        v26.star_tree(17),
        v26.broom_tree(17),
        v26.comet_tree(17),
    )
    tasks = [(tree, root) for tree in trees for root in range(len(tree))]
    proof = v26.exact_certificate(tasks, policy)
    assert proof["passed"]
    assert proof["failures"] == 0
    assert proof["half_shrink_failures"] == 0


def test_hidden_tree_families_are_created_after_policy_freeze() -> None:
    source = inspect.getsource(v26.run)
    freeze = source.index("frozen_digest = digest")
    hidden = source.index("hidden_sizes = (17, 31, 63, 127)")
    evaluation = source.index("candidate = evaluate")
    assert freeze < hidden < evaluation


def test_candidate_gate_requires_separator_proof_and_controls() -> None:
    source = inspect.getsource(v26.run)
    assert 'proof["passed"]' in source
    assert "candidate.half_shrink_violation_rate == 0.0" in source
    assert "candidate.accuracy >= 0.985" in source
    assert "random_gap >= 0.50" in source
    assert "passive_gap >= 0.80" in source
