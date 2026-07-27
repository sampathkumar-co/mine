import numpy as np

from mini_origin.plasticity_v5 import (
    DistributedPlasticMemory,
    LearningScenario,
    PlasticityGenome,
    evaluate_learning,
    hand_delta_control,
    no_learning_control,
)


def test_hand_delta_learns_unseen_mapping() -> None:
    result = evaluate_learning(
        hand_delta_control(),
        LearningScenario(
            dimension=4,
            cells=48,
            examples_per_dimension=7,
            repetitions=5,
            noise=0.025,
            damage_fraction=0.40,
            seed=123,
        ),
    )
    assert result.post_damage_score > 0.80
    assert result.retention > 0.95


def test_no_learning_control_cannot_adapt() -> None:
    genome = no_learning_control(hand_delta_control())
    result = evaluate_learning(
        genome,
        LearningScenario(
            dimension=4,
            cells=48,
            examples_per_dimension=7,
            repetitions=5,
            noise=0.025,
            damage_fraction=0.40,
            seed=123,
        ),
    )
    assert result.post_damage_score < 0.20


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
        learning_rate=0.20,
        error_coefficient=1.0,
        hebb_coefficient=0.0,
        prediction_coefficient=0.0,
        decay=0.001,
        memory_clip=3.0,
        consensus_mix=0.0,
        observation_dropout=0.10,
    )
    small = evaluate_learning(
        genome,
        LearningScenario(3, 40, 7, 5, 0.02, 0.35, 201),
    )
    large = evaluate_learning(
        genome,
        LearningScenario(7, 72, 9, 6, 0.05, 0.55, 203),
    )
    assert small.post_damage_score > 0.80
    assert large.post_damage_score > 0.70
