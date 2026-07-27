import numpy as np

from mini_origin import state_invention_v8 as v8
from mini_origin.state_repair_v9 import (
    RepairProgram,
    RepairingStateLearner,
    evaluate_program,
    one_state_control,
)


def _program() -> RepairProgram:
    return RepairProgram(
        base=v8.StateProgram(
            max_slots=3,
            spawn_logic="novelty",
            assignment="least_confident",
            read_mode="hard",
            novelty_threshold=0.20,
            surprise_threshold=0.20,
            temperature=8.0,
            weight_rate=0.14,
            prototype_rate=0.25,
            confidence_rate=0.12,
            decay=0.0,
            merge_threshold=0.15,
        ),
        spawn_init="nearest",
        reset_replacement=True,
        reconstruction_mix=0.5,
    )


def test_new_state_can_inherit_surviving_memory() -> None:
    rng = np.random.default_rng(5)
    learner = RepairingStateLearner(_program(), 3, 2, rng)
    learner.weights[0] = np.array([0.7, -0.2, 0.1])
    context = np.array([1.0, 0.0])
    selected = learner._merge_or_replace(context)
    assert selected == 1
    assert np.allclose(learner.weights[selected], learner.weights[0])


def test_full_memory_replacement_resets_stale_state() -> None:
    rng = np.random.default_rng(7)
    learner = RepairingStateLearner(_program(), 3, 2, rng)
    contexts = (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
    )
    for context in contexts:
        learner._merge_or_replace(context)
    learner.confidence = [0.9, 0.01, 0.8]
    old = learner.prototypes[1].copy()
    replacement = np.array([0.0, -1.0])
    selected = learner._merge_or_replace(replacement)
    assert selected == 1
    assert not np.allclose(learner.prototypes[selected], old)
    assert np.allclose(learner.prototypes[selected], replacement)


def test_one_state_control_removes_repair_language() -> None:
    control = one_state_control(_program())
    assert control.base.max_slots == 1
    assert control.base.spawn_logic == "never"
    assert control.spawn_init == "zero"
    assert not control.reset_replacement


def test_repair_evaluation_is_bounded() -> None:
    scenario = v8.ContextScenario(
        seed=11,
        contexts=3,
        dimension=5,
        context_dimension=3,
        steps=260,
        phase_length=40,
        context_noise=0.05,
        target_noise=0.025,
        damage_step=160,
        damage_fraction=0.60,
        drift=0.002,
    )
    result = evaluate_program(_program(), scenario)
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["recovery_skill"] <= 1.0
