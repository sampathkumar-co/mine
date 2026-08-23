from __future__ import annotations

import json

import pytest

from mini_origin import generic_bellman_quotient_v89 as v89


def test_preregistration_is_frozen_before_evaluation() -> None:
    payload = json.loads(v89.PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["status"] == "preregistered_before_v89_implementation_or_evaluation"
    assert payload["evaluation"]["seed_start"] == v89.SEED_START
    assert payload["evaluation"]["seeds_per_family"] == v89.SEEDS_PER_FAMILY
    assert payload["claim_boundary"]["breakthrough_claim"] is False


def test_componentwise_dominance_only_with_identical_successors() -> None:
    problem = v89.ShortestPathProblem(v89.SEED_START)
    state = problem.initial_state
    actions = problem.actions(state)
    kept, removed, classes = v89.quotient_actions(problem, actions)
    assert removed > 0
    assert classes > 0
    assert len(kept) + removed == len(actions)
    signatures = [v89.action_signature(problem, action) for action in kept]
    assert len(signatures) == len(set(signatures))


def test_negative_cost_is_rejected() -> None:
    problem = v89.ShortestPathProblem(v89.SEED_START)
    bad = v89.Action("bad", (v89.Outcome((1, 0), 1, -1),), 0)
    with pytest.raises(ValueError, match="negative immediate costs"):
        v89.quotient_actions(problem, (bad,))


@pytest.mark.parametrize(
    "factory",
    [
        v89.DiagnosisProblem,
        v89.SetCoverProblem,
        v89.ShortestPathProblem,
    ],
)
def test_raw_and_quotient_match_on_smoke_seeds(factory) -> None:
    for seed in range(v89.SEED_START, v89.SEED_START + 4):
        problem = factory(seed)
        raw = v89.ExactBellmanSolver(problem, use_quotient=False).result()
        quotient = v89.ExactBellmanSolver(problem, use_quotient=True).result()
        assert raw.objective == quotient.objective
        assert quotient.stats.action_expansions <= raw.stats.action_expansions


def test_small_transfer_certificate_passes() -> None:
    evidence = v89.evaluate(seed_start=v89.SEED_START, seeds_per_family=4)
    assert evidence["passed"] is True
    assert evidence["objective_mismatches"] == 0
    for summary in evidence["families"].values():
        assert summary["instances"] == 4
        assert summary["instances_with_nonzero_quotient_reduction"] >= 1
        assert summary["aggregate_action_expansion_reduction"] > 0
