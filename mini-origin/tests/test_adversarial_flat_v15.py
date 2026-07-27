import numpy as np

from mini_origin import rotating_sketch_v12 as v12
from mini_origin.adversarial_flat_v15 import (
    FlatStatistics,
    _candidate_subsets,
    _top_fraction,
    hand_flat_program,
    hidden_scenarios,
    select_cells,
    training_scenarios,
)


def test_attack_fraction_penalizes_concentration() -> None:
    uniform = np.ones(10)
    concentrated = np.array([9.0] + [1.0 / 9.0] * 9)
    assert _top_fraction(concentrated, 3) > _top_fraction(uniform, 3)


def test_candidate_sets_are_deterministic_and_sparse() -> None:
    scenario = v12.SketchScenario(17, 3, 4, 3.0, 8, 8.0, 0.01, 0.50)
    program = hand_flat_program()
    first = _candidate_subsets(program, scenario, 1, 2)
    second = _candidate_subsets(program, scenario, 1, 2)
    assert len(first) == program.candidate_sets
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    expected = max(1, int(np.floor(program.density * scenario.cells)))
    assert all(len(value) == expected for value in first)


def test_selection_is_deterministic_and_respects_budget() -> None:
    scenario = v12.SketchScenario(23, 3, 4, 3.0, 8, 8.0, 0.01, 0.50)
    program = hand_flat_program()
    statistics = FlatStatistics(
        gram=np.zeros((scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension)),
        response=np.zeros((scenario.cells, scenario.contexts, scenario.dimension)),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
        previous=[np.asarray([], dtype=int) for _ in range(scenario.contexts)],
    )
    feature = np.array([1.0, 0.0, 0.0, 0.0])
    first = select_cells(program, statistics, scenario, 0, 0, feature)
    second = select_cells(program, statistics, scenario, 0, 0, feature)
    assert np.array_equal(first, second)
    assert len(first) / scenario.cells <= 0.20 + 1e-12


def test_hidden_distribution_is_strictly_harder() -> None:
    training = training_scenarios(100)
    hidden = hidden_scenarios(100)
    assert min(value.contexts for value in hidden) > max(value.contexts for value in training)
    assert min(value.dimension for value in hidden) > max(value.dimension for value in training)
    assert min(value.damage_fraction for value in hidden) > max(value.damage_fraction for value in training)
