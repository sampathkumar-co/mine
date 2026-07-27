import numpy as np

from mini_origin.state_invention_v8 import (
    ContextScenario,
    DynamicStateLearner,
    StateProgram,
    evaluate_program,
    one_state_programs,
)


def _dynamic_test_program() -> StateProgram:
    return StateProgram(
        max_slots=4,
        spawn_logic="novelty",
        assignment="least_used",
        read_mode="hard",
        novelty_threshold=0.25,
        surprise_threshold=0.20,
        temperature=8.0,
        weight_rate=0.12,
        prototype_rate=0.25,
        confidence_rate=0.10,
        decay=0.0,
        merge_threshold=0.18,
    )


def test_one_state_language_cannot_create_state() -> None:
    assert one_state_programs()
    assert all(program.max_slots == 1 for program in one_state_programs())
    assert all(program.spawn_logic == "never" for program in one_state_programs())


def test_dynamic_program_creates_distinct_internal_states() -> None:
    rng = np.random.default_rng(3)
    learner = DynamicStateLearner(_dynamic_test_program(), 3, 2, rng)
    feature = np.array([1.0, 0.0, 0.0])
    learner.learn(feature, np.array([1.0, 0.0]), 0.8)
    learner.learn(feature, np.array([0.0, 1.0]), -0.8)
    assert learner.created_slots >= 2
    assert len(learner.weights) >= 2


def test_damage_targets_confident_states_but_keeps_one() -> None:
    rng = np.random.default_rng(5)
    learner = DynamicStateLearner(_dynamic_test_program(), 3, 2, rng)
    feature = np.array([1.0, 0.0, 0.0])
    for context, target in (
        (np.array([1.0, 0.0]), 0.9),
        (np.array([0.0, 1.0]), -0.9),
        (np.array([-1.0, 0.0]), 0.4),
    ):
        for _ in range(4):
            learner.learn(feature, context, target)
    before = len(learner.weights)
    removed = learner.damage(0.75)
    assert removed >= 1
    assert len(learner.weights) < before
    assert len(learner.weights) >= 1


def test_evaluation_reports_bounded_skills() -> None:
    scenario = ContextScenario(
        seed=9,
        contexts=2,
        dimension=4,
        context_dimension=3,
        steps=180,
        phase_length=30,
        context_noise=0.04,
        target_noise=0.02,
        damage_step=110,
        damage_fraction=0.50,
        drift=0.0,
    )
    result = evaluate_program(_dynamic_test_program(), scenario)
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["skill"] <= 1.0
    assert result["created_slots"] >= 1.0
