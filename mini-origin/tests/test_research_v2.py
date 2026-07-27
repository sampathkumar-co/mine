import numpy as np

from mini_origin.research_v2 import (
    DirectionalGenome,
    DirectionalSubstrate,
    RelayCurriculumConfig,
    _identity_genome,
    _shift_fixed,
    discover_relay,
    evaluate_relay,
)


def test_fixed_shift_never_wraps() -> None:
    state = np.zeros((3, 5, 1))
    state[:, 0, 0] = 1.0
    shifted = _shift_fixed(state, 0, -1)
    assert np.allclose(shifted[:, -1, 0], 0.0)


def test_relay_balances_both_signs() -> None:
    channels = 4
    proposal = np.zeros((6, channels, channels))
    gate = np.zeros_like(proposal)
    always_negative = DirectionalGenome(
        proposal_weights=proposal,
        gate_weights=gate,
        proposal_bias=np.array([-3.0, 0.0, 0.0, 0.0]),
        gate_bias=np.array([4.0, -4.0, -4.0, -4.0]),
    )
    result = evaluate_relay(always_negative, width=6)
    assert result.negative_destination < 0.0
    assert result.positive_destination < 0.0
    assert result.score < 0.15


def test_identity_is_not_a_relay() -> None:
    result = evaluate_relay(_identity_genome(), width=8)
    assert result.score < 0.15


def test_substrate_shape_is_stable() -> None:
    rng = np.random.default_rng(3)
    genome = DirectionalGenome.random(rng)
    world = DirectionalSubstrate(genome, 6, 9)
    assert world.step().shape == (6, 9, 4)


def test_small_curriculum_is_deterministic() -> None:
    config = RelayCurriculumConfig(
        widths=(2, 3),
        generations_per_stage=(2, 2),
        population_size=10,
        elite_count=3,
        seed=19,
    )
    first = discover_relay(config)
    second = discover_relay(config)
    assert first.training_score == second.training_score
    assert first.hidden_scores == second.hidden_scores
