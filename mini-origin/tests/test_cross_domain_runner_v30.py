from __future__ import annotations

import inspect

from mini_origin import cross_domain_compiler_v30 as base
from mini_origin import cross_domain_runner_v30 as v30


def test_minimax_query_selects_balanced_code_bit_over_distractors() -> None:
    task = base.bitcode_task(6)
    allowed = frozenset(range(task.candidate_count))
    remaining = frozenset(range(len(task.queries)))
    index = base.select_query(task, allowed, remaining, "minimax_bucket")
    assert task.queries[index].name.startswith("bit-")


def test_argmin_decoder_recovers_inactive_outcome_channel() -> None:
    assert base.decode("argmin", [0, 1, 2], [0.8, -0.1, 0.7]) == 1


def test_actual_query_count_is_not_the_full_budget() -> None:
    task = base.bitcode_task(4)
    policy = base.Policy("minimax_bucket", "argmin")
    correct, queries, invalid = v30.run_trial(17, task, 5, policy)
    assert correct
    assert not invalid
    assert queries == 4
    assert queries < 7


def test_small_cross_domain_certificate_is_near_optimal() -> None:
    policy = base.Policy("minimax_bucket", "argmin")
    proof = base.exact_certificate(v30.reduced_certificate_tasks(31), policy)
    assert proof["passed"]
    assert proof["maximum_gap"] <= 1


def test_hidden_domains_follow_policy_freeze() -> None:
    source = inspect.getsource(v30.run)
    freeze = source.index("frozen_digest = digest")
    hidden = source.index("hidden = base.hidden_tasks")
    evaluation = source.index("candidate = evaluate")
    assert freeze < hidden < evaluation


def test_gate_requires_true_cross_domain_advantage() -> None:
    source = inspect.getsource(v30.run)
    assert 'selected.query_rule == "minimax_bucket"' in source
    assert 'selected.decoder_rule == "argmin"' in source
    assert "candidate.mean_log2_ratio <= 1.20" in source
    assert "first_gap >= 0.20" in source
    assert "random_gap >= 0.20" in source
