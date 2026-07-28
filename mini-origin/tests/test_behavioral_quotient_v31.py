from __future__ import annotations

import inspect

from mini_origin import behavioral_quotient_v31 as v31
from mini_origin import cross_domain_compiler_v30 as base


def test_binary_objectives_share_one_exact_quotient_class() -> None:
    classes = v31.behaviour_quotient()
    assert classes[0].members == (
        "minimax_bucket",
        "gini",
        "max_outcomes",
    )
    assert classes[0].canonical == "minimax_bucket"
    certificate = v31.binary_equivalence_certificate(64)
    assert certificate["passed"]
    assert not certificate["violations"]


def test_multioutcome_domain_has_a_ranking_reversal() -> None:
    witness = v31.multioutcome_divergence_certificate(16)
    assert witness["passed"]
    first = witness["first_witness"]
    assert first["minimax_max"] < first["gini_max"]
    assert first["minimax_sum_squares"] > first["gini_sum_squares"]


def test_scoped_queries_preserve_the_worst_case_distinction() -> None:
    task = v31.divergent_task(12, 37, "test", 5)
    minimax = base.Policy("minimax_bucket", "argmin")
    gini = base.Policy("gini", "argmin")
    minimax_depth = max(
        v31.noiseless_depth(task, minimax, target)
        for target in range(task.candidate_count)
    )
    gini_depth = max(
        v31.noiseless_depth(task, gini, target)
        for target in range(task.candidate_count)
    )
    assert minimax_depth < gini_depth


def test_exact_certificate_checks_every_candidate_trajectory() -> None:
    minimax = base.Policy("minimax_bucket", "argmin")
    gini = base.Policy("gini", "argmin")
    proof = v31.exact_divergence_certificate(41, minimax, gini)
    assert proof["passed"]
    assert proof["root_divergence"]
    assert proof["certificate_type"] == (
        "exhaustive_frozen_policy_trajectories"
    )
    assert all(
        row["canonical_worst_depth"]
        <= row["alternative_worst_depth"]
        for row in proof["rows"]
    )


def test_hidden_domains_are_created_after_class_freeze() -> None:
    source = inspect.getsource(v31.run)
    freeze = source.index("frozen_digest = class_digest")
    hidden = source.index("hidden = hidden_tasks")
    evaluation = source.index("candidate = evaluate")
    assert freeze < hidden < evaluation


def test_gate_uses_worst_case_not_average_case_superiority() -> None:
    source = inspect.getsource(v31.run)
    assert "gini_worst_gain >= 0.40" in source
    assert "strict_gini_tasks >= 2" in source
    assert "outcomes_worst_gain >= 0.40" in source
    assert "strict_outcomes_tasks >= 2" in source
    assert "candidate.accuracy >= 0.985" in source
    assert "candidate.invalid_rate <= 0.01" in source
