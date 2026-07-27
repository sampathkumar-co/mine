import numpy as np

from mini_origin.plasticity_v5 import (
    DistributedPlasticMemory,
    LearningScenario,
    PlasticityGenome,
    evaluate_learning,
    feedback_ablation,
    hand_delta_control,
    hand_hebb_control,
    no_learning_control,
)


def test_hand_delta_learns_from_correlated_examples() -> None:
    scenario = LearningScenario(
        dimension=4,
        cells=48,
        examples_per_dimension=12,
        repetitions=8,
        noise=0.025,
        damage_fraction=0.40,
        condition_number=8.0,
        seed=123,
    )
    result = evaluate_learning(hand_delta_control(), scenario)
    assert result.post_damage_score > 0.70
    assert result.retention > 0.93


def test_frozen_memory_cannot_adapt() -> None:
    scenario = LearningScenario(
        dimension=4,
        cells=48,
        examples_per_dimension=12,
        repetitions=8,
        noise=0.025,
        damage_fraction=0.40,
        condition_number=8.0,
        seed=123,
    )
    result = evaluate_learning(
        no_learning_control(hand_delta_control()),
        scenario,
    )
    assert result.post_damage_score < 0.20


def test_feedback_beats_hebbian_covariance_shortcut() -> None:
    scenario = LearningScenario(
        dimension=4,
        cells=56,
        examples_per_dimension=14,
        repetitions=9,
        noise=0.02,
        damage_fraction=0.35,
        condition_number=14.0,
        seed=311,
    )
    delta = evaluate_learning(hand_delta_control(), scenario)
    hebb = evaluate_learning(hand_hebb_control(), scenario)
    ablated = evaluate_learning(feedback_ablation(hand_delta_control()), scenario)
    assert delta.post_damage_score > hebb.post_damage_score + 0.08
    assert delta.post_damage_score > ablated.post_damage_score + 0.08


def test_damage_removes_memory_but_keeps_survivors() -> None:
    rng = np.random.default_rng(7)
    genome = hand_delta_control()
    memory = DistributedPlasticMemory(genome, cells=20, dimension=3, rng=rng)
    before = memory.memory.copy()
    killed = memory.damage(0.50, rng)
    assert int(np.sum(killed)) == 10
    assert np.allclose(memory.memory[killed], 0.0)
    assert np.allclose(memory.memory[~killed], before[~killed])


def test_learning_law_is_dimension_agnostic() -> None:
    genome = PlasticityGenome(
        learning_rate=0.15,
        error_coefficient=1.0,
        hebb_coefficient=0.0,
        prediction_coefficient=0.0,
        decay=0.0005,
        memory_clip=3.0,
        consensus_mix=0.0,
        observation_dropout=0.10,
    )
    small = evaluate_learning(
        genome,
        LearningScenario(3, 40, 10, 8, 0.02, 0.35, 6.0, 201),
    )
    large = evaluate_learning(
        genome,
        LearningScenario(7, 72, 16, 10, 0.05, 0.55, 18.0, 203),
    )
    assert small.post_damage_score > 0.70
    assert large.post_damage_score > 0.55
