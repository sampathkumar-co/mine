from __future__ import annotations

import inspect
import numpy as np

from mini_origin import weighted_tree_transfer_v28 as v28
from mini_origin import tree_compiler_v26 as v26


def test_weighted_sum_distance_finds_half_mass_separator() -> None:
    rng = np.random.default_rng(7)
    instance = v28.make_weighted_tree(v26.broom_tree(31), rng)
    query = v28.QueryProgram("weighted_sum_distance", 2)
    proof = v28.separator_certificate([instance], query)
    assert proof["passed"]
    assert proof["violations"] == 0


def test_normalization_removes_edge_strength_scale() -> None:
    tree = v26.path_tree(3)
    instance = v28.WeightedTree(
        tree=tree,
        weights=np.ones(3),
        strengths={(0, 1): 0.25, (1, 2): 0.9},
    )
    allowed = {0, 1, 2}
    rows = v28.responses(11, instance, allowed, 1, 0, 100000, True)
    values = dict(rows)
    assert values[0] < 0.05
    assert values[2] > 0.9


def test_hidden_weighted_instances_follow_policy_freeze() -> None:
    source = inspect.getsource(v28.run)
    freeze = source.index("frozen_digest = digest")
    hidden = source.index("hidden_instances = make_instances")
    evaluation = source.index("candidate = evaluate")
    assert freeze < hidden < evaluation
    assert "(31, 63, 127, 255)" in source


def test_gate_requires_mass_certificate_and_normalization_advantage() -> None:
    source = inspect.getsource(v28.run)
    assert 'proof["passed"]' in source
    assert "candidate.half_mass_violation_rate == 0.0" in source
    assert "normalization_gap >= 0.05" in source
    assert "candidate.accuracy >= 0.985" in source
